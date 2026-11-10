# VLA Runtime

A multi-robot policy server for action-chunking Vision-Language-Action (VLA) models: one GPU serves several robots, each within its own latency budget.

The target deployment is a GPU server on the same site as the robots (a lab, a production line, a data-collection or evaluation farm), serving fixed-base arms over the local network. Closed-loop control from a public cloud region is not a goal, and neither is on-robot inference for a single humanoid; both have their own stacks.

The workload shapes the design. A VLA inference is one fixed-size pass: a backbone prefix over the camera views and instruction, then an action expert that produces a chunk of future actions. The context is short and does not grow, so paged attention and KV-cache management buy nothing. Concurrency is low, so throughput tricks matter less than each robot's deadline. What breaks in practice is the control loop running out of actions while inference is in flight, and a shared GPU making every robot late at once.

The runtime is built around three things that single-model policy servers do not provide:

- **Deadline-aware fleet serving.** One loaded model serves many robots. Each robot's backbone pass runs separately, and action heads that share weights are batched across robots in one pass. The head is launch-bound, so a batch of four costs little more than one. With head batching plus CUDA Graphs, π0.5 on one RTX 4090 D serves three robots at a P95 of about 230 ms with the same LIBERO-10 success rate as one robot; serial dispatch serves one.
- **Chunk execution under latency.** Latency trimming, blend policies and Real-Time Chunking (RTC) guidance keep successive chunks continuous. The sim benchmark measures how much latency a policy tolerates (π0.5 × LIBERO-10: about 300 ms without RTC, about 500 ms with RTC `zeros`).
- **Capability declarations.** Where a model reads robot state, what crosses its backbone → head boundary and whether its head is iterative are declared per model (`PolicyCapabilities`), and the schedulers act on the declaration instead of assuming.

## How the pieces relate to LLM serving

LLM serving is a useful map of components, not of techniques: vLLM's gains come from high-concurrency throughput and memory management, which do not transfer.

| LLM serving | VLA Runtime | Why it differs |
|---|---|---|
| The LLM itself | `VLMBackend` | Vision-language prefix; recomputed every frame because the images change |
| Token generation | `ActionHeadRuntime` | An action expert producing a chunk, usually by iterative denoising |
| Continuous batching | `FleetScheduler` head batching | Batches the launch-bound head across robots; the compute-bound backbone runs per robot |
| KV cache / paged attention | *(not applicable)* | The prefix K/V lives for one inference and is consumed by the head |
| Scheduler | `AsyncScheduler` / `FleetScheduler` | Deadlines per robot, not one throughput queue |
| *(no analogue)* | `ChunkBuffer` | Stitching successive action chunks into one continuous trajectory |

## Layers

```
Application     ActionRequest: camera views + instruction + robot state
Serving         FleetScheduler — many robots over one loaded model, head batching, admission
Pipeline        AsyncScheduler — one robot: encode -> head -> chunk buffer
Module          VLMBackend  |  ActionHeadRuntime  (+ PolicyCapabilities, Policy)
Infrastructure  RTC guidance  |  Observability  |  Precision policy
```

## Documentation

| Document | Contents |
|---|---|
| [Architecture](./architecture.md) | The pipeline, data flow, coupled vs decoupled policies, and degradation paths |
| [VLM backends](./vlm-backends.md) | The `VLMBackend` protocol, typed conditions, and the prefix-KV backbone |
| [Action heads](./action-heads.md) | The `ActionHeadRuntime` protocol, its optional capabilities, and the shipped heads |
| [Scheduler](./scheduler.md) | `AsyncScheduler`, chunk stitching and latency compensation, and latency budgets |
| [Fleet serving](./fleet-serving.md) | Serving many robots from one model with isolated per-robot state |
| [Quantization](./quantization.md) | Per-module precision policy and task-success-rate acceptance |
| [Observability](./observability.md) | Structured tracing, metrics, and the error hierarchy |
| [Benchmark methodology](./benchmark_methodology.md) | Closed-loop LIBERO / ManiSkill benchmarks: execution modes, metrics, and how to run them remotely |

## Repository map

```
vla_runtime/
  vlm/
    base.py              VLMBackend protocol, Embedding (typed condition), LayerKV, ActionRequest
  action_head/
    base.py              ActionHeadRuntime protocol and its optional capabilities
                         (WeightSharingHead, RealTimeChunkingHead, BatchedActionHead),
                         RobotState, ActionChunk, conditioning checks
    flow_matching.py     Reference flow-matching head (MLP velocity net) for tests and mocks
    rtc.py               Real-Time Chunking guidance for any flow-matching expert
  scheduler/
    async_scheduler.py   Single-robot pipeline and degradation hooks
    chunk_buffer.py      Chunk stitching, latency trim, underrun tracking
    budget_manager.py    Rolling P50/P95 latency budgets
  serving/
    fleet.py             Multi-robot scheduling over one model, head batching
    admission.py         Predicted round latency → advised replan threshold / admission
  capabilities.py        PolicyCapabilities: state placement, condition kind, frequency mode
  policy.py              Policy: backbone + head prototype + capability declaration
  quantization/
    config.py            Per-module precision schemes and their constraints
    validation.py        Task success rate validation in simulation
  observability/         Structured JSON spans and events, metrics
  errors.py              Exception hierarchy with trace context

benchmarks/
  sim/                   Closed-loop LIBERO / ManiSkill benchmark, and the real model
                         adapters: π0.5 and SmolVLA split at the prefix-KV boundary,
                         CUDA Graph capture for the expert loop
  latency/               Model-level latency / FPS benchmark
  closed_loop/           ARX X5 real-robot closed-loop latency benchmark
examples/                End-to-end runnable examples
tests/                   Unit tests, CPU only
```

The model adapters and CUDA Graph capture currently live under `benchmarks/sim/`; moving them into the package is planned.

## Running the example

`examples/minimal_pipeline.py` drives a complete inference loop against a mock backbone and the reference flow-matching head, so you can watch the pipeline, the chunk buffer and the trace output work together without a GPU or model weights.

```
PYTHONPATH=. python examples/minimal_pipeline.py
```

It emits one structured event per chunk and a final snapshot of buffer state and latency budgets. Real models run through the sim benchmark; see [Benchmark methodology](./benchmark_methodology.md).

## Tests

```
python -m pytest tests/ -q
```

The unit suite is CPU-only. It mocks the VLM backend but exercises the reference head, RTC guidance, schedulers, fleet and chunk buffer for real. The GPU tests for the real adapters (split vs native alignment, RTC vs LeRobot, batched vs per-robot heads) are in `benchmarks/benchmark_sim.py`, marked `@pytest.mark.gpu`, and skip without CUDA.
