# Action Heads

The action head turns a vision-language embedding and the robot's current state into a chunk of future actions. It is the half of a VLA policy that touches the control loop, and the half where precision matters more than compression.

Source: `vla_runtime/action_head/`

## The protocol

`ActionHeadRuntime` in `vla_runtime/action_head/base.py`:

```python
class ActionHeadRuntime(Protocol):
    @property
    def head_type(self) -> Literal["discrete_token", "flow_matching", "diffusion"]: ...
    @property
    def frequency_hz(self) -> float: ...
    @property
    def chunk_size(self) -> int: ...
    @property
    def action_dim(self) -> int: ...

    def predict(self, embedding: Embedding, robot_state: RobotState) -> ActionChunk: ...
    def load_weights(self, checkpoint_path: str) -> None: ...
    def to_device(self, device: torch.device) -> None: ...
```

`predict()` is synchronous by design. Timing is the scheduler's responsibility, and a head that also tried to manage its own cadence would duplicate that logic three times over. The scheduler runs `predict()` in an executor so a slow head does not block the event loop.

Each robot gets its own head instance. State isolation is guaranteed by the caller giving each robot a separate object, not by locking inside the head, which keeps the heads free of synchronization entirely.

Separate instances do not mean separate weights; see `WeightSharingHead` below.

A head must be able to read the condition kind its policy declares (`accepted_condition_kinds`). The reference head conditions an MLP on a pooled vector, so it accepts `kind="pooled"` only and rejects anything else with a message naming the kind it received. The prefix-KV expert accepts `per_layer_kv` only.

`chunk_size` is the main lever on the latency-versus-freshness tradeoff. A larger chunk amortizes per-call overhead and gives the control loop more runway before it can underrun, but it also means actions at the end of the chunk were computed from an older observation.

## Data types

### `RobotState`

```python
@dataclass
class RobotState:
    joint_positions: Tensor      # [dof]
    joint_velocities: Tensor     # [dof]
    gripper_state: Tensor        # [] or [n_fingers]
    ee_pose: Tensor | None       # [7] — xyz + quaternion
    extra: dict[str, Tensor]
```

Two derived properties matter when configuring a head. `dof` is the number of joints. `flat_dim` is the width of `as_flat_tensor()`, which is what a head's `state_dim` must equal.

These are not the same number, and the gap is the most common configuration mistake. `flat_dim` is `len(joint_positions) + len(joint_velocities) + gripper_state.numel()`, plus seven if `ee_pose` is present, plus the size of everything in `extra`. A seven-joint arm with one gripper channel has `dof == 7` but `flat_dim == 15`, not 14. Size `state_dim` from `flat_dim` on a representative `RobotState`, never from `2 * dof`.

`check_conditioning_dims()` in the same module enforces both the embedding width and the state width before the conditioning network sees them. Without it, a mismatch surfaces as a matrix-shape error from inside an MLP, which says nothing about which input was wrong; with it, the error names both the configured and the actual width and points at the likely cause.

### `ActionChunk`

```python
@dataclass
class ActionChunk:
    actions: Tensor                  # [chunk_size, action_dim]
    head_type: ActionHeadType
    confidence: Tensor | None        # [chunk_size]
    metadata: dict
```

`chunk_size` and `action_dim` are read off the tensor shape, and `step(i)` returns a single action. `metadata` carries head-specific detail — the number of denoising steps used, the sampler, whether the chunk was produced by a fallback path — which is what makes a trace readable after the fact.

## Optional capabilities

Three capabilities sit beside the core protocol rather than in it, so a third-party head that has none of them still satisfies `ActionHeadRuntime`. The schedulers check for each one and use it only when present.

| Protocol | Methods | Used by |
|---|---|---|
| `WeightSharingHead` | `spawn()`, `shares_weights_with()` | `Policy.head_for_robot()` and `FleetScheduler.from_policy()`: every robot gets its own instance over one copy of the weights |
| `RealTimeChunkingHead` | `supports_rtc`, `predict(..., context)` | Schedulers pass an `ActionHeadContext` (queued actions, inference delay) only when the policy declares `supports_rtc`; the head guides its denoising toward the committed prefix (`vla_runtime/action_head/rtc.py`) |
| `BatchedActionHead` | `predict_batch(embeddings, robot_states, contexts)` | `FleetScheduler` runs every ready row of heads that share weights in one pass; each row keeps its own noise, condition, RTC context and output |

`spawn()` instances read the same weights, frozen in eval mode with gradients off. `load_weights()` and `to_device()` act on that shared copy, so they apply to every spawned instance and must not run while any of them is serving.

## The shipped heads

### Reference flow-matching head

Source: `vla_runtime/action_head/flow_matching.py`

`FlowMatchingHead` integrates a velocity field from noise to an action chunk with a small MLP (`VelocityMLP`) conditioned on the current sample, time, a pooled embedding and the flat robot state. It is the head the unit tests and the mock sim policy use; it implements weight sharing and RTC, so scheduling, chunk stitching and guidance can be exercised without a model.

Two integrators are available. Euler takes `n_steps` fixed steps. RK4 evaluates the field four times per step for a higher-order estimate. Select with `use_rk4`.

### Prefix-KV action expert

Source: `benchmarks/sim/lerobot_smolvla.py` (`PrefixKVExpertHead`)

The π0.5 and SmolVLA action experts, run over the `per_layer_kv` condition from the prefix-KV backbone ([VLM backends](./vlm-backends.md)). The expert keeps no state between calls, so spawned instances share the one loaded model and any instance can serve a batch. It implements all three optional capabilities:

- batched denoising: per-layer K/V stacked along the batch, shorter prefixes right-padded and masked;
- RTC guidance inside its Euler loop;
- for π0.5, optional replay of the unguided loop from CUDA Graphs captured per batch size (`benchmarks/sim/cuda_graph.py`).

The head is launch-bound. The π0.5 expert's 10 denoising steps take 84 ms eager and 33 ms replayed from a graph at batch 1, and a batch of four costs about the same as one row eager. That is why batching it across robots is nearly free.

## Paradigms

`head_type` names the paradigm; it decides which `head_structure` a policy may declare (`vla_runtime/policy.py`) and which precision policy applies ([Quantization](./quantization.md)).

| `head_type` | Example models | Structure | Cost driver |
|---|---|---|---|
| `flow_matching` | π0, π0.5, SmolVLA, GR00T N1.x | iterative | Integration steps × expert size |
| `diffusion` | Diffusion Policy | iterative | Denoising steps |
| `discrete_token` | OpenVLA, π0-FAST | single pass or monolithic | Decode steps (autoregressive action tokens) |

Only `flow_matching` heads are integrated today. Iterative heads are the ones the runtime can guide (RTC) and batch.

## Related documents

- [Architecture](./architecture.md) — where action generation sits in the pipeline
- [Scheduler](./scheduler.md) — how chunks are stitched into a continuous trajectory
- [Fleet serving](./fleet-serving.md) — how heads are batched across robots
- [Quantization](./quantization.md) — precision constraints and how they are validated
