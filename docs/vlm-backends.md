# VLM Backends

The vision-language backbone is the expensive half of a VLA policy and the half with the most mature serving ecosystem behind it. Rather than reimplementing inference, the runtime defines a narrow protocol and adapts existing engines to it. Everything downstream depends only on the protocol, so backends are interchangeable.

Source: `vla_runtime/vlm/`

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

    async def encode(self, image: Tensor, text: str) -> Embedding: ...
    async def generate(self, image: Tensor, text: str) -> TokenStream: ...
    async def warmup(self) -> None: ...
    async def shutdown(self) -> None: ...
```

Every method is async and must not block the event loop — backends wrapping synchronous libraries are expected to dispatch through a thread pool internally, which is what the llama.cpp adapter does.

`encode()` is the path used by flow matching and diffusion heads: it returns a feature tensor for the action head to condition on. `generate()` is the token-level path used only by discrete-action-token models, returning an async iterator the caller must drain. A backend serving a flow matching policy does not need a meaningful `generate()`.

`warmup()` exists because the first forward pass through a JIT-compiled engine can cost orders of magnitude more than steady-state, which would otherwise show up as a spurious timeout on the very first request. `AsyncScheduler.start()` and `FleetScheduler.start()` both call it before accepting work.

## Data types

### `Embedding`

The single object crossing the boundary between the backbone and the action head.

| Field | Meaning |
|---|---|
| `data` | Feature tensor, `[seq_len, hidden_dim]` or `[hidden_dim]` |
| `created_at` | Creation timestamp, used to reason about staleness |
| `frame_id` | Monotonic counter within one robot's request stream |
| `model_id` | Which backend produced it |
| `metadata` | Backend-specific extras such as an attention mask |

`age_ms` and `is_stale(max_age_ms)` are derived from `created_at`. These matter on the degradation path: when the backbone times out, the scheduler reuses the previous embedding, and the trace records how old it was so a stale-embedding episode is visible after the fact rather than silent.

### `ActionRequest`

The input to the pipeline.

| Field | Meaning |
|---|---|
| `request_id` | Unique per request |
| `image` | Preprocessed image tensor `[C, H, W]` |
| `instruction` | Natural language task instruction |
| `robot_state` | Joint positions, velocities, gripper — a tensor, never text |
| `frame_id` | Monotonic frame counter for this robot |
| `timestamp` | Creation time |

Robot state travels as a tensor on its own channel and is handed to the action head directly. Serializing it into the prompt would make the backbone's input change on every control tick, which defeats any cross-frame caching and wastes backbone capacity on numbers the action head can consume directly.

### `TokenChunk` and `TokenStream`

`TokenStream` is an `AsyncIterator[TokenChunk]`. Each chunk carries `token_ids`, optional `log_probs`, and an `is_final` flag. Only discrete-token models use this path — see `DiscreteTokenHead.predict_from_stream()` in [Action heads](./action-heads.md).

## Adapters

Four adapters ship with the runtime, covering server-class and embedded deployment.

| Adapter | Engine | Intended for | Key constructor arguments |
|---|---|---|---|
| `VLLMBackend` | vLLM `AsyncLLMEngine` | Cloud and datacenter GPUs | `model_path`, `tensor_parallel_size`, `quantization`, `kv_cache_enabled` |
| `SGLangBackend` | SGLang Runtime | Cloud, structured decoding workloads | `model_path`, `tp_size`, `quantization` |
| `LMDeployBackend` | LMDeploy TurboMind | Cloud, quantized serving | `model_path`, `tp`, `cache_max_entry_count`, `quant_policy` |
| `LlamaCppBackend` | llama-cpp-python, GGUF | Edge and embedded devices | `model_path`, `n_gpu_layers`, `n_ctx`, `n_threads`, `embedding_dim` |

All four take `latency_budget_ms` and forward unrecognized keyword arguments to the underlying engine, so engine-specific tuning stays available without the adapter having to enumerate every option.

The edge adapter differs in more than which library it calls. `n_ctx` defaults to 512 because VLA context genuinely is short, and it wraps its synchronous llama.cpp calls in a thread pool so the async contract holds. Its default `latency_budget_ms` is higher than the server adapters', reflecting that an edge device is slower but still has to stay inside the control loop's tolerance.

## KV cache management

Source: `vla_runtime/vlm/kv_cache.py`

Cross-frame reuse is worth having in VLA serving, but the mechanism differs from LLM serving. A VLA context is an image plus a short instruction — typically well under a thousand tokens — so there is no fragmentation problem to solve and no reason to page. What there is instead is a clear structural split: the image tokens are the expensive, reusable part, and the text tokens are what accumulate.

`KVCacheManager` exploits that split directly.

- The pool is pre-allocated at construction, so allocation and release during serving are O(1) free-list operations with no allocator pressure in the hot path.
- Image tokens are pinned. They are never evicted, because they are the part worth keeping.
- Text tokens evict under a sliding window, oldest first, shifting the retained tail in place across every layer.

`KVCacheConfig` describes the backbone's shape — `num_layers`, `num_heads`, `head_dim` — plus `max_context_tokens`, `num_image_tokens`, and `pool_size`, which should be at least the number of concurrent robots. `bytes_per_slot` and `pool_memory_gb` are computed from those, so the memory footprint is known before anything is allocated rather than discovered when it runs out.

The API is `allocate()`, `write_kv()`, `evict_oldest_text_tokens()`, `free()`, and `get()`, with `stats()` and `memory_usage_gb()` for introspection. Setting `eviction_policy="none"` turns overflow into an error instead of silently dropping context, which is the right behavior while establishing how much context a given model actually needs.

## Implementing a backend

Satisfy the protocol and the rest of the runtime works unchanged. Three things are worth getting right:

Report an honest `latency_budget_ms`. The scheduler uses it to decide when to fall back to a stale embedding, so a value set optimistically produces spurious degradation and one set generously delays a fallback the robot needed.

Never block the event loop. If the underlying library is synchronous, dispatch through an executor.

Raise the runtime's own exception types. `VLMTimeoutError` and `BackendError` carry trace context, so a failure is attributable to a request and a module instead of appearing as an anonymous traceback. See [Observability](./observability.md).

## Related documents

- [Architecture](./architecture.md) — where the backbone sits in the pipeline
- [Action heads](./action-heads.md) — what consumes the embedding
- [Quantization](./quantization.md) — how aggressively the backbone can be compressed
- [Fleet serving](./fleet-serving.md) — sharing one backbone across many robots
