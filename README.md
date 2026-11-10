# VLA Runtime

**A multi-robot policy server for action-chunking VLA models** — one GPU serves several robots, each within its own latency budget.

---

## Scope

The target is a GPU server on the same site as the robots (a lab, a production line, a data-collection or evaluation farm) serving fixed-base arms over the local network.

Out of scope:
- closed-loop control from a public cloud region, because its round-trip latency and jitter do not fit a chunk's budget;
- on-robot inference for a single humanoid, which has its own stacks (TensorRT pipelines on Jetson Thor).

Official policy servers (openpi, LeRobot) already serve one model to one robot well. This project adds what their structure does not have:

- **Deadline-aware fleet serving.** One loaded model serves N robots. Backbone passes run per robot. Action heads that share weights are batched across robots in one pass: the head is launch-bound, so a batch of four costs little more than one. Optional admission control predicts the round latency for the registered robots.
- **Chunk execution under latency.** Latency trimming, blend policies and Real-Time Chunking (RTC) guidance for flow-matching experts. The closed-loop benchmark measures how much latency a policy tolerates.
- **Capability declarations.** `PolicyCapabilities` declares, per model, where robot state enters, what crosses the backbone → head boundary and whether the head is iterative. The schedulers act on the declaration instead of assuming.
- **Closed-loop benchmarks.** Task success, latency percentiles, buffer underruns and smoothness are measured together, in simulation (LIBERO / ManiSkill) and on a real arm (ARX X5).

## Results so far

π0.5 (`pi05_libero_finetuned_v044`) × LIBERO-10 on one RTX 4090 D, in simulation. Comparisons are paired by seed, 100–500 episodes per cell; network round trip not included.

| Question | Result |
|---|---|
| How much latency does closed loop tolerate? | 0 ms (sync): 90.2%. Trim blending: 92.4% at 300 ms. RTC with the `zeros` prefix schedule: 94.4% at 300 ms and 89.0% at 500 ms (no significant drop from 0 ms). Every configuration drops significantly from 750 ms. |
| Cross-fading chunks? | `linear_overlap` is 15–19 pp below plain trimming at 150–500 ms, so trimming is the default. |
| RTC prefix schedule? | Guiding only the frozen delay steps (`zeros`) beats the paper's exponential schedule by +7.8 / +12.2 pp at 300 / 500 ms. |
| Robots per GPU at P95 ≤ 300 ms | Serial dispatch: 1. Head batching + CUDA Graph + bf16 vision tower + sdpa: 3 robots, P95 229 ms, 95% success (same as 1 robot). |
| Load-adaptive replan threshold? | Not the default: at best about 2 pp over a fixed threshold, and only at light load. |

Model-level latency on the same GPU (P50, raw frames → actions on host, single robot):
- OpenVLA-7B: 162 ms
- OpenVLA-OFT: 113 ms
- π0.5 (bf16): 161 ms
- SmolVLA: 110 ms

ARX X5 real-arm closed loop with π0.5: end-to-end P50 is 237 ms, of which about 65 ms is outside inference (camera frame age, transport, command).

These numbers hold for this checkpoint, suite, GPU and acceleration stack. They are not a claim of beating the official servers on single-robot latency.

## Status

Research prototype.

| Model | Integration | State |
|---|---|---|
| π0.5 (LeRobot) | Split at the prefix-KV boundary, `vla_runtime/models/pi05.py` | Matches native inference (1e-3); closed-loop LIBERO; fleet batching; CUDA Graph expert |
| SmolVLA (LeRobot) | Split at the prefix-KV boundary, `vla_runtime/models/smolvla.py` | Matches native inference; closed-loop LIBERO; fleet batching |
| OpenVLA, OpenVLA-OFT, π0 | Model-level latency benchmark only (`benchmarks/latency`) | Not integrated into the runtime |

The policy server (`vla-serve`) puts the fleet behind a WebSocket endpoint that speaks openpi's wire protocol. openpi's own client drives it unmodified (tested with openpi-client 0.1.2 against the mock policy). It has not yet served a real model to a real robot: the ARX X5 benchmark still uses its own single-model server.

Not built yet:
- a client-side chunk executor for real robots;
- serving several models on one GPU;
- serving models that cannot be split at the backbone → head boundary (OpenVLA-OFT, fused TensorRT pipelines).

## Design

```
Application      robot client (openpi protocol) or in-process ActionRequest
Serving          PolicyServer (WebSocket) → FleetScheduler: N robots over one loaded model,
                 head batching, admission
Pipeline         AsyncScheduler: one robot, encode → head → ChunkBuffer
Module           VLMBackend | ActionHeadRuntime (+ PolicyCapabilities, Policy)
Infra            RTC guidance | Observability | Precision policy
```

A policy is split at the boundary between its vision-language prefix and its action expert. The split is not for running the two at different rates or on different machines: the per-layer prefix K/V that crosses it is about 17.8 MB per frame for π0.5, so both halves run in the same process on the GPU server. The split exists so that the two halves can be scheduled differently:
- the compute-bound backbone runs per robot;
- the launch-bound head is batched across robots;
- the head's denoising loop is where CUDA Graph replay and RTC guidance plug in.

Robot state goes where the model was trained to read it. For π0.5 and SmolVLA that is the backbone prefix, which makes them **coupled**: backbone and head run once per request. A **decoupled** policy, whose head reads fresh state, may re-run its head on a cached condition. The policy declares which kind it is in `PolicyCapabilities`.

See [`docs/`](./docs/README.md) for the subsystems.

## Quick start

```bash
pip install -e ".[dev]"
python examples/minimal_pipeline.py    # mock backbone + reference head, CPU only
python -m pytest                       # CPU unit tests
vla-serve --policy mock --port 8000    # policy server with the mock policy
```

Serving a real model to robots:

```bash
vla-serve --policy pi05 --policy-opt repo=<checkpoint> \
    --policy-opt cuda_graph=1 --policy-opt vision_bf16=1 --policy-opt expert_sdpa=2 \
    --image-key agentview=observation/image --image-key wrist=observation/wrist_image \
    --max-robots 4
```

```python
from openpi_client.websocket_client_policy import WebsocketClientPolicy
actions = WebsocketClientPolicy(host="gpu-server", port=8000).infer(observation)["actions"]
```

See [`docs/serving-api.md`](./docs/serving-api.md) for the protocol.

Closed-loop benchmarks need a CUDA host. In this repository, GPU work runs on a remote host through `scripts/remote.py`:

```bash
# CPU plumbing check, any machine
python -m benchmarks.sim.run --env mock --policy mock --modes all --episodes 2

# π0.5 fleet on LIBERO-10, 1–4 robots sharing one GPU
python scripts/remote.py run -- python -m benchmarks.sim.run \
    --env libero --env-opt suite=libero_10 --policy pi05 \
    --policy-opt cuda_graph=1 --policy-opt vision_bf16=1 --policy-opt expert_sdpa=2 \
    --modes realtime --robots 1,2,3,4 --chunk-threshold 0.9
```

See [`docs/benchmark_methodology.md`](./docs/benchmark_methodology.md) for the execution modes, metrics and options.

## Core interfaces

```python
class VLMBackend(Protocol):
    async def encode(self, images: Mapping[str, Tensor], text: str,
                     state: RobotState | None = None) -> Embedding: ...

class ActionHeadRuntime(Protocol):
    def predict(self, embedding: Embedding, robot_state: RobotState) -> ActionChunk: ...
    # Optional capabilities: WeightSharingHead (spawn), RealTimeChunkingHead (context),
    # BatchedActionHead (predict_batch)

class PolicyCapabilities(BaseModel):
    state_placement: Literal["prefix", "suffix", "head_encoder", "none"]
    condition_kind: Literal["pooled", "prefix_tokens", "per_layer_kv", "projected_features"]
    head_structure: Literal["iterative", "single_pass", "monolithic"]
    max_condition_staleness_ms: float | None
    supports_rtc: bool

fleet = FleetScheduler.from_policy(policy, FleetConfig(max_robots=4))
await fleet.start()
await fleet.register_robot("arm_0")
handle = await fleet.submit("arm_0", request)
async for chunk in fleet.stream(handle):
    ...
```

`Embedding` is a typed condition: `pooled`, `prefix_tokens`, `per_layer_kv` or `projected_features`. Its payload is checked against its kind at construction, and its `nbytes` is reported in traces.

## Precision policy

| Component | Allowed | Reason |
|---|---|---|
| Vision encoder / backbone | fp16 or lower, model by model | Runs once per frame; small perturbations are absorbed |
| Iterative action head | 16-bit float (fp16 / bf16); no INT4 | Each denoising step feeds the next, so error compounds |
| Discrete-token head | additionally W8A16 | One decision per token; error does not compound |

A precision change is accepted only on closed-loop task success, paired against the unchanged model. Loss and action error alone are not enough.

## References

| Reference | Relevance |
|---|---|
| [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339) | RTC guidance implemented here |
| [Training-Time Action Conditioning for Efficient Real-Time Chunking](https://arxiv.org/abs/2512.05964) | RTC moved into training; same context interface |
| [LeRobot](https://github.com/huggingface/lerobot) | PolicyServer + RobotClient async inference; source of the π0.5 / SmolVLA checkpoints |
| [openpi](https://github.com/Physical-Intelligence/openpi) | Official π0 / π0.5 remote-inference server and wire protocol |
| [How Fast Can I Run My VLA? (VLA-Perf)](https://arxiv.org/abs/2602.18397) | VLA inference performance analysis |
| [KerColle](https://arxiv.org/abs/2609.22335) | GPU concurrency between VLA backbone and action stages |
| [vla.cpp](https://arxiv.org/abs/2606.08094) | Unified C++ inference engine for 11 VLA models |
| [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) | Parallel decoding for OpenVLA |

## License

This project is licensed under the [Apache License 2.0](./LICENSE).
