# VLM Backends

The vision-language backbone is the half of a VLA policy that reads the camera views and the instruction. The runtime defines a narrow protocol for it; everything downstream depends only on the protocol, so backbones are interchangeable.

Source: `vla_runtime/vlm/base.py` (protocol and data types); `vla_runtime/models/prefix_kv.py` (the prefix-KV backbone).

## The protocol

`VLMBackend` in `vla_runtime/vlm/base.py` is a runtime-checkable `Protocol`. Implementations may expose more, but callers rely only on this.

```python
class VLMBackend(Protocol):
    @property
    def model_id(self) -> str: ...
    @property
    def latency_budget_ms(self) -> float: ...
    @property
    def kv_cache_enabled(self) -> bool: ...

    async def encode(
        self, images: Mapping[str, Tensor], text: str, state: RobotState | None = None,
    ) -> Embedding: ...
    async def generate(
        self, images: Mapping[str, Tensor], text: str, state: RobotState | None = None,
    ) -> TokenStream: ...
    async def warmup(self) -> None: ...
    async def shutdown(self) -> None: ...
```

Every method is async and must not block the event loop. A backbone wrapping synchronous GPU work dispatches it to a thread internally; the prefix-KV backbone uses one dedicated thread, which also serializes concurrent fleet requests on the shared weights.

`encode()` is the path used by flow matching and diffusion heads: it returns the condition the action head reads. `generate()` is the token-level path used only by discrete-action-token models, returning an async iterator the caller must drain. A backend serving a flow matching policy does not need a meaningful `generate()`.

`images` holds every camera view the policy takes, keyed by camera name, in the order the model expects — most VLA policies take more than one (a base camera plus a wrist camera, for instance). `state` is passed only when the loaded policy was trained with robot state inside the backbone prefix (`PolicyCapabilities.state_placement == "prefix"`, e.g. SmolVLA and π0.5). The scheduler makes that decision from the policy's declaration; a backend never pulls state in on its own, and for every other policy it receives `None`.

`warmup()` exists because the first forward pass through a JIT-compiled engine can cost orders of magnitude more than steady-state, which would otherwise show up as a spurious timeout on the very first request. `AsyncScheduler.start()` and `FleetScheduler.start()` both call it before accepting work.

## Data types

### `Embedding`

The single object crossing the boundary between the backbone and the action head. The name is historical: what a real action head reads is often much more than one embedding vector, so the payload is tagged with a `kind`.

| `kind` | Payload | Example models |
|---|---|---|
| `pooled` | `data` `[hidden_dim]` | Simple MLP heads (the reference `FlowMatchingHead`) |
| `prefix_tokens` | `data` `[seq_len, hidden_dim]` | Heads that cross-attend over the prefix sequence |
| `per_layer_kv` | `layer_kv`: one `LayerKV(key, value)` per layer, each `[num_kv_heads, seq_len, head_dim]` | π0-style joint attention, where the action expert attends over every layer's prefix K/V |
| `projected_features` | `features`: named tensors | GR00T-style heads that read projected backbone features |

| Field | Meaning |
|---|---|
| `data` | Hidden states; required for `pooled` and `prefix_tokens` |
| `kind` | One of the four above; inferred from the rank of `data` when omitted |
| `layer_kv` | Per-layer prefix K/V; required for `per_layer_kv`, all layers the same length |
| `features` | Named conditioning tensors; required for `projected_features` |
| `attention_mask` | Optional `[seq_len]` bool, `True` = visible; masks padded language tokens |
| `created_at` | Creation timestamp, used to reason about staleness |
| `frame_id` | Monotonic counter within one robot's request stream |
| `model_id` | Which backend produced it |
| `metadata` | Backend-specific extras |

Construction validates that the payload matches `kind`, so a malformed condition fails where it is built rather than inside a head. `seq_len`, `num_layers`, and `nbytes` are derived. `nbytes` matters for deployment: it is what an edge-cloud split placed at this boundary would have to ship every frame. For a π0-scale model (about 800 prefix tokens, 18 layers, one KV head of width 256, fp16) that is roughly 15 MB per frame — which is why the split point cannot simply sit between backbone and head for `per_layer_kv` models (ADR-018).

`age_ms` and `is_stale(max_age_ms)` are derived from `created_at`. On the degradation path a decoupled policy may reuse the previous condition, and whether it still may is decided by comparing its age with the policy's calibrated tolerance; see [Scheduler](./scheduler.md).

### `ActionRequest`

The input to the pipeline.

| Field | Meaning |
|---|---|
| `request_id` | Unique per request |
| `images` | Camera views keyed by name, each `[C, H, W]`; at least one; insertion order is the order handed to the backbone |
| `instruction` | Natural language task instruction |
| `robot_state` | A structured `RobotState` (joint positions, velocities, gripper, optional pose and extras) — a tensor channel, never text |
| `frame_id` | Monotonic frame counter for this robot |
| `timestamp` | Creation time |

Robot state travels as a tensor on its own channel and is handed to the action head directly. Serializing it into the prompt would make the backbone's input change on every control tick, which defeats any cross-frame caching and wastes backbone capacity on numbers the action head can consume directly.

### `TokenChunk` and `TokenStream`

`TokenStream` is an `AsyncIterator[TokenChunk]`. Each chunk carries `token_ids`, optional `log_probs`, and an `is_final` flag. Only discrete-token models (autoregressive action tokens) use this path; none is integrated yet.

## The prefix-KV backbone

Source: `vla_runtime/models/prefix_kv.py` (`PrefixKVBackbone`), with model classes for SmolVLA (`vla_runtime/models/smolvla.py`) and π0.5 (`vla_runtime/models/pi05.py`); `vla_runtime/models/registry.py` loads them by name

π0.5 and SmolVLA are each one network: a VLM runs over the prefix (camera views, instruction, and for these two models the robot state), and a flow-matching action expert attends over every layer's prefix K/V while it denoises. `PrefixKVBackbone.encode()` runs the LeRobot model's own prefix pass in-process and returns those K/V as a `per_layer_kv` condition. The matching head is `PrefixKVExpertHead` ([Action heads](./action-heads.md)). Together they reproduce the model's native `sample_actions` step for step; the GPU tests check the split against LeRobot's own inference with shared noise (max deviation 1e-3).

The condition is large: about 17.8 MB per frame for π0.5 (18 layers × 968 prefix tokens). It never leaves the GPU; backbone and head run in the same process, and what crosses the network is the observation going up and the action chunk coming down.

General-purpose LLM engines (vLLM, SGLang, LMDeploy, llama.cpp) are not used for this backbone. Their public interfaces do not hand per-layer K/V to an external consumer, and they do not run the expert's joint attention over the prefix, so integrating one would mean writing the expert into the engine's model code. They fit models whose output is text or action tokens, such as a high-level planner that issues language subtasks.

## Implementing a backend

Satisfy the protocol and the rest of the runtime works unchanged. Three things are worth getting right:

Report an honest `latency_budget_ms`. The scheduler uses it to decide when to fall back to a stale embedding, so a value set optimistically produces spurious degradation and one set generously delays a fallback the robot needed.

Produce the condition kind the policy declares. The scheduler compares every `Embedding.kind` against `PolicyCapabilities.condition_kind` and raises `PolicyCapabilityError` on a mismatch, because the head would otherwise receive a payload it cannot read.

Accept `state` and use it only if the model consumes state in its prefix. A prefix-state model must place it where the model was trained to find it; the prefix-KV backbone hands it to the model's own preprocessing.

Never block the event loop. If the underlying library is synchronous, dispatch through an executor.

Raise the runtime's own exception types. `VLMTimeoutError` and `BackendError` carry trace context, so a failure is attributable to a request and a module instead of appearing as an anonymous traceback. See [Observability](./observability.md).

## Related documents

- [Architecture](./architecture.md) — where the backbone sits in the pipeline
- [Action heads](./action-heads.md) — what consumes the embedding
- [Quantization](./quantization.md) — precision policy and how precision changes are accepted
- [Fleet serving](./fleet-serving.md) — sharing one backbone across many robots
