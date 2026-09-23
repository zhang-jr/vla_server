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

Separate instances do not mean separate weights. The built-in heads and `TensorRTBackend` implement `WeightSharingHead`: `spawn()` returns a new instance that reads the same `SharedWeights` — one copy of the network, frozen in eval mode with gradients off. `load_weights()` and `to_device()` act on that shared copy, so they apply to every spawned instance and must not run while any of them is serving. `TensorRTBackend.spawn()` shares the compiled engine but creates a new execution context, since a context holds per-inference state.

A head must be able to read the condition kind its policy declares. The three built-in heads condition an MLP on a pooled vector, so they accept `kind="pooled"` only and reject anything else with a message naming the kind they received. A π0-style expert that attends over per-layer prefix K/V, or a GR00T-style head reading projected features, needs a head that implements that attention — the built-in heads are not stand-ins for it.

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

## The three paradigms

The three action paradigms differ enough computationally that they are implemented independently rather than behind a shared abstraction. A discrete-token head decodes autoregressively; a flow matching head integrates an ODE; a diffusion head iteratively denoises. They share only the protocol.

### Discrete tokens

Source: `vla_runtime/action_head/discrete_token.py`

Actions are quantized into a token vocabulary and predicted like text. Two components make this practical for control rates.

`FASTTokenizer` compresses an action chunk through a DCT basis before quantization, so a chunk becomes a short token sequence rather than one token per dimension per timestep. `encode()` maps `[chunk_size, action_dim]` to a token list and `decode()` inverts it.

`OFTParallelDecoder` predicts the whole chunk in one forward pass instead of stepping autoregressively. Autoregressive decoding of an action chunk is the single largest latency contributor in a discrete-token VLA, and removing the sequential dependency is what brings this paradigm into control-loop range. Enable it with `use_oft=True`.

The head supports both directions. With `use_oft=True`, `predict()` decodes the chunk directly from the embedding. Without it, the tokens come from the backbone's token stream, and the caller uses `predict_from_stream(token_stream, robot_state)` — which is the path that pairs with `VLMBackend.generate()`.

### Flow matching

Source: `vla_runtime/action_head/flow_matching.py`

A velocity field is integrated from noise to an action chunk. `VelocityMLP` conditions on the concatenation of the current sample, time, the embedding, and the flat robot state.

Two integrators are available. Euler takes `n_steps` fixed steps and is the cheaper option. RK4 evaluates the field four times per step for a higher-order estimate, trading compute for accuracy at a given step count — worth it when `n_steps` is small enough that Euler's truncation error shows up in the trajectory. Select with `use_rk4`.

Both call into the kernels in `kernels/flow_matching_step.py`, which are Triton on GPU and PyTorch elsewhere. See [Kernels](./kernels.md).

### Diffusion

Source: `vla_runtime/action_head/diffusion.py`

Reverse diffusion from noise to an action chunk, with `NoisePredictorMLP` conditioning on sample, time, embedding, and flat state.

Two samplers are available. DDPM injects stochastic noise at each step and follows the training-time reverse process. DDIM is deterministic and reaches comparable quality in fewer steps, which is why it is the default — denoising step count is the dominant cost in this paradigm, and every step removed is latency returned to the control loop.

The noise schedule is cosine. Note that `ᾱ` is floored at `_ALPHA_BAR_MIN` rather than being allowed to reach its analytic value of zero at `t = 1`: every sampler divides the residual by `sqrt(ᾱ_t)` to recover the clean sample, so a true zero makes that amplification unbounded and overflows float16 to infinity on the very first reverse step. The floor is the same guard the improved-DDPM schedule gets from clipping β.

## TensorRT wrapper

Source: `vla_runtime/action_head/tensorrt_backend.py`

`TensorRTBackend` wraps any head that satisfies the protocol, exporting it to ONNX and compiling a TensorRT engine. It satisfies the protocol itself, so it substitutes for the head it wraps without any change to the scheduler.

`build_or_load_engine(example_inputs)` compiles on first use and loads a cached engine afterwards. The wrapper carries a precision guard: it will not compile a flow matching or diffusion head at a precision below what those paradigms tolerate, so an aggressive engine configuration fails at build time rather than producing a policy that quietly misses its target. See [Quantization](./quantization.md) for the precision policy this enforces.

## Choosing a configuration

| | Discrete token | Flow matching | Diffusion |
|---|---|---|---|
| Cost driver | Decode steps, unless parallel decoding is on | Integration steps | Denoising steps |
| Main lever | `use_oft` | `n_steps`, `use_rk4` | `n_steps`, `sampler` |
| Precision tolerance | Tolerates modest quantization | Float16 | Float16 |
| Needs the token path | Yes, unless using parallel decoding | No | No |

The precision column is the one that constrains deployment. Flow matching and diffusion are iterative: each step's error feeds the next, so quantization error compounds over the trajectory rather than averaging out. A discrete-token head makes one classification decision per token and is far more forgiving. This is why the runtime allows aggressive compression of the backbone but not of an iterative head — the details are in [Quantization](./quantization.md).

## Related documents

- [Architecture](./architecture.md) — where action generation sits in the pipeline
- [Scheduler](./scheduler.md) — how chunks are stitched into a continuous trajectory
- [Kernels](./kernels.md) — the Triton kernels the iterative heads call
- [Quantization](./quantization.md) — precision constraints and how they are validated
