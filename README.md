# VLA Runtime

**A model-agnostic serving runtime for Vision-Language-Action models** — serve any VLA model the way vLLM serves LLMs.

---

## Background

VLA deployment today looks a lot like LLM serving in 2022, before vLLM:

- There is no unified inference framework; each team builds its own modular async pipeline, and they all look alike
- Optimizations are scattered across research papers with no production-grade integration
- The core optimizations of vLLM/SGLang (KV cache, continuous batching, PagedAttention) mostly fail or yield marginal gains in VLA workloads

This project fills the "vLLM for VLA" gap with a production-deployable VLA serving runtime.

---

## Core Design

### Three-Stage Async Pipeline

```
Thread A — perception     1–5 Hz    heavy compute, can run in the cloud
Thread B — action gen     5–20 Hz   medium frequency
Thread C — control loop   20–100 Hz real-time, must run on the edge
```

The three stages differ enormously in frequency and compute density. **Async decoupling is the core architecture, not an optimization.** Synchronous inference leaves the robot idle and violates the hard constraint `inference_latency < action_execution_time`.

### Layered Architecture

```
Application      ← ActionRequest (image + instruction + robot_state)
Serving          ← API Server / Fleet Scheduler / Edge-Cloud Router
Pipeline         ← AsyncScheduler (perception → action → control)
Module           ← VLM Backend  |  Action Head Runtime  |  ROS2 Control
Infra            ← Triton/CUDA Kernel | Quantization | Observability
```

### Concept Mapping to vLLM

| vLLM | VLA Runtime | Notes |
|------|-------------|-------|
| LLM | `VLMBackbone` | Vision-language backbone, low-frequency heavy compute |
| Token Generation | `ActionHead` | Discrete token / flow matching / diffusion |
| KV Cache | `EmbeddingCache` | Reuse perception embeddings across frames |
| Continuous Batching | `FleetServing` | One shared VLM instance (weights); per-robot forward passes and strictly isolated state |
| API Server | `ActionEndpoint` | REST + WebSocket, `POST /v1/action` |
| Scheduler | `AsyncScheduler` | Three-stage pipeline (perception / action / control) |

---

## Project Layout

```
vla_runtime/
├── vlm/                # VLM backbone backends (vLLM / SGLang / llama.cpp / LMDeploy)
├── action_head/        # Action head runtimes (discrete_token / flow_matching / diffusion)
├── scheduler/          # Async pipeline (AsyncScheduler / ChunkBuffer / BudgetManager)
├── control/            # ROS2 integration (Python bridge + C++ control node)
├── serving/            # API server + fleet scheduling + edge-cloud router
├── quantization/       # Quantization config + calibration + sim validation
├── observability/      # Structured tracer + metrics
└── errors.py           # Exception hierarchy
kernels/                # Triton / CUDA kernels (flow matching / diffusion / fused attention)
tests/unit/             # Unit tests
examples/               # End-to-end examples
docs/                   # Subsystem documentation
```

See [`docs/`](./docs/README.md) for in-depth documentation of each subsystem.

---

## Quick Start

### Install

```bash
pip install -e ".[dev]"
```

### Minimal Example (no GPU)

```bash
python examples/openvla_minimal.py
```

The example uses a mock VLM backend and runs the full pipeline without a GPU: quantization config → VLM encode → action head decode → chunk buffer consumption.

### Real models and GPU work

No real model runs end to end yet: the vLLM / SGLang / LMDeploy / llama.cpp adapters are stubs that return correctly shaped placeholder tensors.

Tests that need a GPU are marked `@pytest.mark.gpu` and are skipped automatically where no CUDA device is visible. Run them on a CUDA machine with:

```bash
python -m pytest -m gpu
```

---

## Core Interfaces

### VLMBackend

```python
class VLMBackend(Protocol):
    async def encode(
        self, images: Mapping[str, Tensor], text: str, state: RobotState | None = None,
    ) -> Embedding: ...
    async def generate(
        self, images: Mapping[str, Tensor], text: str, state: RobotState | None = None,
    ) -> AsyncIterator[TokenChunk]: ...
    @property
    def latency_budget_ms(self) -> float: ...
```

Supported backends: `vLLM`, `SGLang`, `LMDeploy` (cloud); `llama.cpp + GGUF` (edge). The adapters are currently stubs that return correctly shaped placeholder tensors; no real model is wired end to end yet.

`Embedding` is a typed condition — `pooled`, `prefix_tokens`, `per_layer_kv`, or `projected_features` — because what a real action head reads is often per-layer prefix K/V or projected features rather than one vector. `state` reaches the backbone only for policies trained with state in the prefix.

### ActionHeadRuntime

```python
class ActionHeadRuntime(Protocol):
    def predict(self, embedding: Embedding, robot_state: RobotState) -> ActionChunk: ...
    @property
    def head_type(self) -> Literal["discrete_token", "flow_matching", "diffusion"]: ...
    @property
    def frequency_hz(self) -> float: ...
```

Three paradigms are supported: discrete tokens (OpenVLA / RT-2), flow matching (Pi0 / SmolVLA), and diffusion (Diffusion Policy).

### AsyncScheduler

```python
scheduler = AsyncScheduler(
    vlm_backend=...,
    action_head=...,
    capabilities=PolicyCapabilities(       # declared per model, not guessed (ADR-018)
        state_placement="suffix",
        condition_kind="per_layer_kv",
        head_structure="iterative",
        max_condition_staleness_ms=300.0,
    ),
    budget_config=...,
)
await scheduler.start()

handle = await scheduler.submit(request)
async for chunk in scheduler.stream(handle):
    robot.execute(chunk)
```

The policy's capabilities decide its frequency mode. A **decoupled** policy (head consumes fresh state, iterative, staleness calibrated) may re-run its head on the cached condition via `replan()`, and on a backbone timeout reuses the last condition within its tolerance. A **coupled** policy (e.g. state in the backbone prefix) never re-runs the head on an old condition; on a timeout the robot keeps executing the queued chunk tail.

Three built-in degradation hooks: `on_vlm_timeout` (decoupled only — reuse the previous condition within tolerance), `on_action_head_failure`, and `on_budget_exceeded`.

### QuantizationConfig

```python
# Aggressive cloud preset
cfg = QuantizationConfig.cloud_aggressive()
# vision_encoder=w8a8, llm_backbone=w4a16, action_head=fp16

# flow matching / diffusion require action_head=fp16; violations raise PrecisionDegradationError
cfg.validate_for_head_type("flow_matching")
```

---

## Quantization Policy

| Component | Allowed | Forbidden | Reason |
|-----------|---------|-----------|--------|
| Vision Encoder | fp16 / w8a8 / w4a8 | - | - |
| LLM Backbone | fp16 / w4a16 / w8a16 / gptq / gguf_q4 | - | - |
| Action Head | **fp16 / w8a16 only** | INT4 / GPTQ | Iterative denoising accumulates error; INT4 collapses task success rate |

Acceptance criterion: end-to-end sim task success rate drops by ≤ 2 pp after quantization — loss alone is not enough.

---

## Observability

Every inference emits a structured trace:

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

The JSON tracer will be replaced by OpenTelemetry, exporting to Langfuse / Jaeger.

---

## Performance Targets

| Metric | Mid-term Target | Final Target |
|--------|-----------------|--------------|
| End-to-end latency P50 | ≤ 500 ms | Baseline −10% |
| End-to-end latency P95 | ≤ 600 ms | ≤ 550 ms |
| Throughput | ≥ 3 FPS | ≥ 5 FPS |
| GPU memory | ≤ 90% of baseline | ≤ 80% of baseline |
| Success-rate drop after quantization | ≤ 2 pp | ≤ 1 pp |

Reference baseline: LiteVLA-Edge at 150.5 ms (llama.cpp + GGUF Q4 + ROS2, edge deployment).

---

## Supported VLA Models (Planned)

| Model | Head Type | VLM Backend | Status |
|-------|-----------|-------------|--------|
| OpenVLA-7B | discrete_token | vLLM / SGLang | Adapter done, pending real-GPU validation |
| SpatialVLA-4B | discrete_token | vLLM | Adapter done |
| SmolVLA / Pi0 | flow_matching | llama.cpp | Adapter done |
| Diffusion Policy | diffusion | - | Head done |

---

## Tech Stack

| Component | Choice |
|-----------|--------|
| Cloud VLM | vLLM / SGLang / LMDeploy |
| Edge VLM | llama.cpp + GGUF |
| Action Head Kernel | Triton (primary) + TensorRT (fallback) |
| Serialization | msgpack (2–5x faster than JSON) |
| Transport | WebSocket (default) / gRPC (enterprise) |
| Control Integration | ROS2 Humble+, C++ control node |
| Quantization Calibration | LLM-Compressor / TensorRT calibrator |
| Sim Validation | LIBERO / RoboCasa |
| Tracing | OpenTelemetry → Langfuse / Jaeger |
| Testing | pytest + pytest-benchmark |

---

## TODO

**Baselines & Validation**
- [ ] GPU server (A100 / H100) and edge device (Jetson Orin) environments
- [ ] Containerized sim environments (LIBERO + RoboCasa)
- [ ] Native inference baselines for OpenVLA / SpatialVLA
- [ ] Reproduce LiteVLA-Edge (target ≈ 150.5 ms)
- [ ] Unoptimized end-to-end sim task success rate
- [ ] Sim success rate for every quantization combination

**Performance**
- [ ] Triton kernel profiling (Nsight Compute)
- [ ] Triton vs TensorRT vs ONNX Runtime comparison
- [ ] Resident action head weights + on-demand VLM loading
- [ ] Hit the latency / throughput / memory targets on real GPUs

**System Integration**
- [ ] ROS2 integration: Python bridge + C++ control node, rosbag record & replay
- [ ] Edge-cloud split scheduling over WebSocket + msgpack, with offline fallback
- [ ] Real-robot robustness: safe takeover ramp, observation freshness, `ObservationStaleError`
- [ ] API server: `POST /v1/action`, WebSocket streaming, OpenPI wire-schema compatibility
- [ ] Model hot-swap, Prometheus metrics, health checks
- [ ] OpenTelemetry tracer

**Model & Embodiment Generalization**
- [ ] Real-Time Chunking (RTC) support for flow matching heads
- [ ] Single-arm, bimanual, and mobile manipulation embodiments
- [ ] Action space abstraction (joint / cartesian / delta) with structured actuator groups
- [ ] Fleet serving with ≥ 5 independent robots

**Release**
- [ ] PyPI package
- [ ] Docker images (server + edge)
- [ ] HuggingFace Hub integration
- [ ] Public benchmark suite

---

## Key References

| Reference | Key Contribution |
|-----------|------------------|
| [LiteVLA-Edge: Quantized On-Device Multimodal Control for Embedded Robotics](https://arxiv.org/abs/2603.03380) | Full edge deployment with llama.cpp + ROS2, 150.5 ms baseline |
| [How Fast Can I Run My VLA? Demystifying VLA Inference Performance with VLA-Perf](https://arxiv.org/abs/2602.18397) | Systematic analysis of VLA inference performance, 15 key takeaways |
| [RoboECC: Multi-Factor-Aware Edge-Cloud Collaborative Deployment for VLA Models](https://arxiv.org/abs/2603.20711) | Automatic edge-cloud split point selection, up to 3.28x speedup |
| [Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success (OpenVLA-OFT)](https://arxiv.org/abs/2502.19645) | Parallel decoding with bidirectional attention, 26x action throughput |
| [DeeR-VLA: Dynamic Inference of Multimodal Large Language Models for Efficient Robot Execution](https://arxiv.org/abs/2411.02359) | Dynamic early-exit mechanism (NeurIPS 2024) |
| [LeRobot](https://github.com/huggingface/lerobot) | PolicyServer + RobotClient async inference, the closest existing framework |

---

## License

This project is licensed under the [Apache License 2.0](./LICENSE).
