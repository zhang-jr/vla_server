# VLA Runtime

A model-agnostic serving runtime for Vision-Language-Action (VLA) models.

VLA deployment looks superficially like LLM serving, but the workload is shaped differently. A VLA policy is two models with wildly different cost profiles glued together: a heavy vision-language backbone that runs a few times per second, and a lightweight action head that must feed a control loop running one to two orders of magnitude faster. The context is short, so paging attention buys nothing. Every robot carries independent, real-time state, so requests cannot be batched together the way LLM requests can. And the thing that actually breaks is not throughput — it is a control loop that runs out of actions to execute while inference is still in flight.

This runtime is built around that shape. It keeps the backbone and the action head behind separate interfaces so they can be scheduled, quantized, and deployed independently; it runs them in a staged asynchronous pipeline so the robot never idles waiting on a forward pass; and it treats latency compensation and graceful degradation as part of the core execution path rather than as error handling bolted on afterwards.

## How the pieces map to LLM serving

If you know vLLM or SGLang, this table locates the analogous components.

| LLM serving | VLA Runtime | Why it differs |
|---|---|---|
| The LLM itself | `VLMBackend` | Vision-language backbone; heavy, low frequency, often remote |
| Token generation | `ActionHeadRuntime` | Three paradigms — discrete tokens, flow matching, diffusion |
| KV cache | `KVCacheManager` | Context is short; image tokens are pinned and text slides |
| Continuous batching | `FleetScheduler` | Only the backbone is shared; per-robot state stays isolated |
| Paged attention | *(not applicable)* | Short context means no fragmentation to solve |
| Scheduler | `AsyncScheduler` | Three stages at three different rates, not one queue |
| *(no analogue)* | `ChunkBuffer` | Stitching successive action chunks into one continuous trajectory |

## Layers

```
Application     ActionRequest: image + instruction + robot state
Serving         FleetScheduler — many robots over one shared backbone
Pipeline        AsyncScheduler — perception -> action generation -> control
Module          VLMBackend  |  ActionHeadRuntime
Infrastructure  Triton kernels  |  Quantization  |  Observability
```

## Documentation

| Document | Contents |
|---|---|
| [Architecture](./architecture.md) | The staged pipeline, the rates each stage runs at, data flow, and degradation paths |
| [VLM backends](./vlm-backends.md) | The `VLMBackend` protocol, the four adapters, embeddings, and KV cache management |
| [Action heads](./action-heads.md) | The `ActionHeadRuntime` protocol, the three action paradigms, and the TensorRT wrapper |
| [Scheduler](./scheduler.md) | `AsyncScheduler`, chunk stitching and latency compensation, and latency budgets |
| [Fleet serving](./fleet-serving.md) | Serving many robots from one backbone with isolated per-robot state |
| [Quantization](./quantization.md) | Per-module precision policy, calibration, and task-success-rate validation |
| [Kernels](./kernels.md) | The Triton kernels, their CPU fallbacks, and autotuning |
| [Observability](./observability.md) | Structured tracing, metrics, and the error hierarchy |

## Repository map

```
vla_runtime/
  vlm/                 VLMBackend protocol, four adapters, KV cache manager
    base.py            Protocol, Embedding, ActionRequest, TokenStream
    vllm_backend.py    vLLM AsyncLLMEngine adapter
    sglang_backend.py  SGLang Runtime adapter
    llamacpp_backend.py  llama.cpp / GGUF adapter for edge devices
    lmdeploy_backend.py  LMDeploy TurboMind adapter
    kv_cache.py        Fixed-window KV cache with pinned image tokens
  action_head/         ActionHeadRuntime protocol and the three paradigms
    base.py            Protocol, RobotState, ActionChunk, conditioning checks
    discrete_token.py  FAST tokenizer and parallel decoding
    flow_matching.py   Euler and RK4 integration of a velocity field
    diffusion.py       DDPM and DDIM sampling on a cosine schedule
    tensorrt_backend.py  TensorRT engine wrapper for any head
  scheduler/           The asynchronous pipeline
    async_scheduler.py Three-stage pipeline and degradation hooks
    chunk_buffer.py    Chunk stitching, latency trim, underrun tracking
    budget_manager.py  Rolling P50/P95 latency budgets
  serving/
    fleet.py           Multi-robot scheduling over a shared backbone
  quantization/
    config.py          Per-module precision schemes and their constraints
    calibration.py     Reproducible per-module calibration pipeline
    validation.py      Task success rate validation in simulation
  observability/
    tracer.py          Structured JSON spans and events
    metrics.py         Counters, gauges, histograms
  errors.py            Exception hierarchy with trace context

kernels/               Triton kernels, each with a PyTorch CPU fallback
  flow_matching_step.py  Euler and RK4 integration steps
  diffusion_step.py      DDPM and DDIM denoising steps
  fused_attention.py     Tiled multi-head attention
  layernorm_linear.py    Fused LayerNorm and Linear

benchmarks/            Kernel and pipeline benchmarks
examples/              End-to-end runnable examples
tests/                 Unit tests, CPU only
```

## Running the example

`examples/openvla_minimal.py` drives a complete inference loop against a mock backend, which makes it a fast way to see the pipeline, the chunk buffer, and the trace output working together without a GPU or model weights.

```
PYTHONPATH=. python examples/openvla_minimal.py
```

It prints per-cycle chunk shapes and a final snapshot of buffer state and latency budgets. Swapping the mock for a real backend is a one-line change — every backend satisfies the same protocol described in [VLM backends](./vlm-backends.md).

## Tests

```
python -m pytest tests/ -q
```

The unit suite is CPU-only and mocks the VLM backends, but exercises the action heads, kernels, scheduler, and chunk buffer for real rather than mocking them.
