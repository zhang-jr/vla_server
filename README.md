# VLA Runtime

**模型无关的 VLA Serving Runtime** — 像 vLLM 服务 LLM 一样，统一服务任意 VLA 模型。

---

## 背景

当前 VLA（Vision-Language-Action）部署生态类似 LLM 2022 年（pre-vLLM 时代）：

- 没有统一推理框架，各团队独立实现相似的模块化异步 pipeline
- 优化工作分散在学术论文中，缺乏生产级整合
- vLLM/SGLang 的核心优化（KV cache、continuous batching、PagedAttention）在 VLA 场景大多失效或收益微弱

本项目填补"vLLM for VLA"的空白，提供可生产部署的 VLA serving runtime。

---

## 核心设计

### 三线程异步 Pipeline

```
Thread A — perception     1–5 Hz    heavy compute，可云端
Thread B — action gen     5–20 Hz   中频
Thread C — control loop   20–100 Hz real-time，必须 edge
```

三个线程频率与计算密度差异巨大，**异步解耦是核心架构，不是优化项**。同步推理会让机器人 idle，违反 `inference_latency < action_execution_time` 硬约束。

### 架构分层

```
Application      ← ActionRequest (image + instruction + robot_state)
Serving          ← API Server / Fleet Scheduler / Edge-Cloud Router
Pipeline         ← AsyncScheduler (perception → action → control)
Module           ← VLM Backend  |  Action Head Runtime  |  ROS2 Control
Infra            ← Triton/CUDA Kernel | Quantization | Observability
```

### 与 vLLM 的概念映射

| vLLM | 本项目 | 说明 |
|------|--------|------|
| LLM | `VLMBackbone` | 视觉-语言主干，低频 heavy compute |
| Token Generation | `ActionHead` | 离散 token / flow matching / diffusion |
| KV Cache | `EmbeddingCache` | 跨帧复用感知 embedding |
| Continuous Batching | `FleetServing` | 共享 VLM 计算，机器人 state 严格隔离 |
| API Server | `ActionEndpoint` | REST + WebSocket，`POST /v1/action` |
| Scheduler | `AsyncScheduler` | 三级 pipeline（perception/action/control） |

---

## 项目结构

```
vla_runtime/
├── vlm/                # VLM backbone 后端 (vLLM / SGLang / llama.cpp / LMDeploy)
├── action_head/        # Action head runtime (discrete_token / flow_matching / diffusion)
├── scheduler/          # 异步 pipeline (AsyncScheduler / ChunkBuffer / BudgetManager)
├── control/            # ROS2 集成 (Python bridge + C++ control node)
├── serving/            # API server + fleet scheduling + edge-cloud router
├── quantization/       # 量化配置 + 校准 + sim 验证
├── observability/      # 结构化 tracer + metrics
└── errors.py           # 异常体系
kernels/                # Triton / CUDA kernel (flow matching / diffusion / fused attention)
tests/unit/             # 单元测试
examples/               # 端到端示例
```

---

## 快速开始

### 安装

```bash
pip install -e ".[dev]"
```

### 最小示例（无 GPU）

```bash
python examples/openvla_minimal.py
```

示例使用 `MockVLMBackend`，无需 GPU 即可运行完整的三阶段 pipeline：量化配置 → VLM encode → action head decode → chunk buffer 消费。

### 切换真实模型（一行）

```python
# examples/openvla_minimal.py 第 97 行
# 将 MockOpenVLABackend() 替换为：
from vla_runtime.vlm.vllm_backend import VLLMBackend
vlm_backend = VLLMBackend(model_path="openvla/openvla-7b")
```

---

## 核心接口

### VLMBackend

```python
class VLMBackend(Protocol):
    async def encode(self, image: Tensor, text: str) -> Embedding: ...
    async def generate(self, image: Tensor, text: str) -> AsyncIterator[TokenChunk]: ...
    @property
    def latency_budget_ms(self) -> float: ...
```

支持后端：`vLLM`、`SGLang`、`LMDeploy`（云端）；`llama.cpp + GGUF`（边缘）。

### ActionHeadRuntime

```python
class ActionHeadRuntime(Protocol):
    def predict(self, embedding: Embedding, robot_state: RobotState) -> ActionChunk: ...
    @property
    def head_type(self) -> Literal["discrete_token", "flow_matching", "diffusion"]: ...
    @property
    def frequency_hz(self) -> float: ...
```

支持三种范式：离散 token（OpenVLA / RT-2）、flow matching（Pi0 / SmolVLA）、diffusion（Diffusion Policy）。

### AsyncScheduler

```python
scheduler = AsyncScheduler(vlm_backend=..., action_head=..., budget_config=...)
await scheduler.start()

handle = await scheduler.submit(request)
async for chunk in scheduler.stream(handle):
    robot.execute(chunk)
```

内置三个降级钩子：`on_vlm_timeout`（复用上一帧 embedding）、`on_action_head_failure`、`on_budget_exceeded`。

### QuantizationConfig

```python
# 云端激进量化预设
cfg = QuantizationConfig.cloud_aggressive()
# vision_encoder=w4a8, llm_backbone=w4a16, action_head=fp16

# flow matching / diffusion 强制 action_head=fp16，违反抛 PrecisionDegradationError
cfg.validate_for_head_type("flow_matching")
```

---

## 量化策略

| 组件 | 允许范围 | 禁止 | 原因 |
|------|---------|------|------|
| Vision Encoder | fp16 / w8a8 / w4a8 | - | - |
| LLM Backbone | fp16 / w4a16 / w8a16 / gptq / gguf_q4 | - | - |
| Action Head | **fp16 / w8a16 only** | INT4 / GPTQ | 迭代去噪误差累积，INT4 会导致任务成功率崩塌 |

验收标准：量化后 sim 端到端任务成功率下降 ≤ 2 pp，不只看 loss。

---

## 可观测性

每次 inference 产出结构化 trace：

```json
{
  "trace_id": "abc123",
  "request_id": "req-001",
  "module": "async_scheduler",
  "event": "perception_complete",
  "duration_ms": 48.3,
  "frame_id": 42,
  "stale_frames": 0,
  "quantization": "w4a16"
}
```

Phase 3 替换为 OpenTelemetry，对接 Langfuse / Jaeger。

---

## 性能目标

| 指标 | Phase 2 中期目标 | Phase 4 最终目标 |
|------|-----------------|-----------------|
| 端到端延迟 P50 | ≤ 500 ms | 基线 -10% |
| 端到端延迟 P95 | ≤ 600 ms | ≤ 550 ms |
| 吞吐 | ≥ 3 FPS | ≥ 5 FPS |
| 显存 | ≤ 90% 基线 | ≤ 80% 基线 |
| 量化后成功率下降 | ≤ 2 pp | ≤ 1 pp |

参考基线：LiteVLA-Edge 150.5 ms（llama.cpp + GGUF Q4 + ROS2，边缘部署）。

---

## 支持的 VLA 模型（规划）

| 模型 | Head 类型 | VLM Backend | 状态 |
|------|---------|------------|------|
| OpenVLA-7B | discrete_token | vLLM / SGLang | adapter 完成，待真实 GPU 验证 |
| SpatialVLA-4B | discrete_token | vLLM | adapter 完成 |
| SmolVLA / Pi0 | flow_matching | llama.cpp | adapter 完成 |
| Diffusion Policy | diffusion | - | head 完成 |

---

## 技术选型

| 组件 | 方案 |
|------|------|
| 云端 VLM | vLLM / SGLang / LMDeploy |
| 边缘 VLM | llama.cpp + GGUF |
| Action Head Kernel | Triton（主）+ TensorRT（fallback） |
| 序列化 | msgpack（比 JSON 快 2–5x） |
| 通信 | WebSocket（默认）/ gRPC（enterprise） |
| 控制集成 | ROS2 Humble+，C++ control node |
| 量化校准 | LLM-Compressor / TensorRT calibrator |
| 仿真验证 | LIBERO / RoboCasa |
| Trace | OpenTelemetry → Langfuse / Jaeger |
| 测试 | pytest + pytest-benchmark |

---

## 关键参考文献

| 文献 | 核心贡献 |
|------|---------|
| LiteVLA-Edge (2603.03380) | llama.cpp + ROS2 全链路边缘部署，150.5 ms baseline |
| VLA-Perf (2602.18397) | VLA 推理性能系统分析，15 条 takeaway |
| RoboECC (2603.20711) | 自动 edge-cloud 切分点，3.28x 加速 |
| OpenVLA-OFT (2503.12345) | 双向 attention 并行解码，25–50x |
| DeeR-VLA (NeurIPS 2024) | 动态早退出机制 |
| LeRobot (HuggingFace) | PolicyServer + RobotClient async，最接近的现有框架 |

---

## License

待定（计划开源）。
