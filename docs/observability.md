# Observability

Performance problems in a VLA pipeline are not diagnosable by inspection. A robot that moves jerkily might be suffering backbone latency, a starved chunk buffer, a quantization regression, or an action head that is fine but arriving late. These look identical from outside and call for opposite responses, so every inference produces a structured record of what each stage cost and which degradation paths were taken.

Source: `vla_runtime/observability/`, `vla_runtime/errors.py`

## Tracing

`vla_runtime/observability/tracer.py` provides module-scoped tracers emitting structured JSON. There are two record types: spans, which have a duration, and events, which are instantaneous.

```python
from vla_runtime import get_tracer

tracer = get_tracer("scheduler")

with tracer.span("vlm_encode", request_id=rid, model="openvla") as span:
    embedding = await backend.encode(images, text)
    span.set_attribute("tokens", embedding.dim)

tracer.event("chunk_pushed", robot_id="arm_left", chunk_size=16)
```

`get_tracer(module)` returns a singleton per module name. A span records its own duration and emits on exit; `set_attribute()` adds fields, `set_error(exc)` marks it failed, and `set_degraded(reason)` marks it as having taken a fallback path — which is what separates "this request was slow" from "this request was served from a stale embedding" in the log.

Every record carries a `trace_id`. `tracer.child(module)` produces a tracer for a different module that shares the current trace id, which is how one request stays correlated as it moves from perception into action generation. `tracer.new_trace()` starts a fresh one.

Output is one JSON object per line through the standard `logging` module, so it composes with existing log shipping. `configure_logging()` sets up a sensible default handler.

```json
{"type": "span", "trace_id": "…", "span_id": "…", "module": "scheduler",
 "name": "perception_encode", "request_id": "…", "duration_ms": 61.2, "status": "ok"}
```

`_emit_span()` and `_emit_event()` are the extension points. Overriding them in a `Tracer` subclass redirects records to any backend without touching a single call site.

## Metrics

`vla_runtime/observability/metrics.py` is a thread-safe in-process store with three instrument types.

```python
from vla_runtime.observability.metrics import get_metrics

metrics = get_metrics()
metrics.increment("chunk_pushed")
metrics.gauge("fleet_queue_depth", 4)
metrics.histogram("action_buffer_gap_ms", 12.4)
```

`increment(name, delta=1, labels=None)` counts, `gauge(name, value)` records a current level, and `histogram(name, value)` accumulates a distribution with count, min, max, and mean. All three accept `labels`, which are folded into the key in Prometheus style. `snapshot()` returns everything at once.

Tracing and metrics answer different questions and both are needed. A trace explains one request in full; a metric shows whether a behavior is rare or routine. A single stale-embedding fallback is an unremarkable event in a trace, and a fallback counter climbing steadily is a backbone that no longer fits its budget.

## What the runtime emits

These are produced without any instrumentation on your part.

| Event | Emitted when | Notable fields |
|---|---|---|
| `chunk_pushed` | An action chunk is integrated into the buffer | `chunk_size`, `blend_policy`, `trimmed_steps`, `blended_steps`, `dropped`, `contended`, `remaining` |
| `action_buffer_gap` | The control loop underruns, and again when it recovers | `state` (`start` / `end`), `duration_ms`, `empty_polls` |
| `latency_trim_updated` | Measured latency changes the compensation trim | `latency_ms`, `trim_steps` |
| `chunk_buffer_seeded` | The buffer is anchored to a measured pose | `robot_id` |
| `vlm_timeout_fallback` | A decoupled policy reuses its last condition | `stale_frames`, `embedding_age_ms` |
| `vlm_timeout_hold_tail` / `fleet_vlm_timeout_hold_tail` | A coupled policy's backbone timed out; no chunk, the queued tail keeps running | `stale_frames` |
| `scheduler_started` / `fleet_started` | A scheduler starts | `frequency_mode`, `staleness_uncalibrated` |
| `action_head_failure_fallback` | A zero-action chunk is substituted | `error` |
| `rolling_budget_violation` | Budget violations exceed the tolerated fraction | `latency_ms`, `budget_ms`, `stats` |
| `inference_complete` | A request finishes | `action_ms`, `total_ms`, `chunk_size`, `latency_trim_steps` |
| `fleet_request_failed` | One robot's request fails in the shared dispatch loop | `robot_id`, `error_type`, `error` |

Spans cover `perception_encode`, `action_gen_predict`, `scheduler_replan`, `flow_matching_predict`, `diffusion_predict`, `discrete_token_predict`, and `fleet_process`. `perception_encode` carries `frequency_mode`, `condition_kind`, and `condition_bytes` — the last is the per-frame payload an edge-cloud split at the backbone boundary would have to ship.

Two of these deserve attention when reading a trace. `action_buffer_gap` is the one that corresponds directly to physical symptoms — it means the control loop had nothing to execute, which is what a stutter actually is — and it is recorded as an interval rather than a counter because the tail duration is the part that matters. `dropped` on `chunk_pushed` means a chunk arrived so late that the entire thing was already stale, which is a stronger signal than latency percentiles: the pipeline is no longer keeping up at all.

## Error hierarchy

`vla_runtime/errors.py` defines one tree rooted at `VLARuntimeError`. Every exception carries an `ErrorContext` with `trace_id`, `module_name`, an optional `request_id`, and arbitrary extras, so a failure is attributable to a request and a module instead of arriving as a bare traceback.

```
VLARuntimeError
├── BackendError
│   ├── VLMTimeoutError                 backbone exceeded its latency budget
│   └── ActionHeadFailureError          action head raised
├── PipelineError
│   ├── BudgetExceededError             latency exceeded the configured budget
│   ├── EmbeddingStaleError             reused condition older than the policy tolerates
│   └── PolicyCapabilityError           operation or payload the declared capabilities rule out
├── QuantizationError
│   └── PrecisionDegradationError       precision constraint violated, or success rate dropped
├── KernelError                         Triton or TensorRT compilation or launch failure
└── DeploymentError
    ├── EdgeResourceError               insufficient device resources or queue capacity
    └── NetworkPartitionError           remote backbone unreachable
```

Each subclass carries the numbers that describe its own failure. `VLMTimeoutError` holds `timeout_ms` and `budget_ms`; `EmbeddingStaleError` holds the condition's `age_ms` and the `max_age_ms` it exceeded; `PrecisionDegradationError` holds baseline and quantized success rates and the threshold. The message is readable and the fields are machine-readable, so alerting does not have to parse strings.

The types are re-exported at package level:

```python
from vla_runtime import VLMTimeoutError, EmbeddingStaleError, EdgeResourceError
```

Two conventions govern their use. Never suppress an exception without context — catch, add an `ErrorContext`, re-raise. And never use `print()`; every diagnostic goes through the tracer, so it is structured, correlated by trace id, and routable.

## Reading a degradation

A worked example. The control loop reports stuttering.

Start with `action_buffer_gap` events. If none are present, the buffer never underran and the problem is elsewhere — likely trajectory smoothness rather than availability, in which case check `blend_policy` and whether `chunk_pushed` shows `blended_steps` at zero.

If gaps are present, check `duration_ms` against the control period. Gaps of one or two ticks are the buffer running to its last action before the next chunk lands; long gaps mean chunks are not arriving in time.

Then check whether chunks are arriving at all. `dropped: true` on `chunk_pushed` means chunks are arriving fully stale. `vlm_timeout_fallback` events mean the backbone is missing its budget and the pipeline is running on reused embeddings.

Finally read the budget snapshot from `stats()`. If P95 for `vlm` exceeds its budget while `action_head` is comfortable, the backbone is the constraint — quantize it further, move it to a faster device, or lower the perception rate. If `action_head` is the one over, reduce the step count or the chunk size. See [Quantization](./quantization.md) and [Action heads](./action-heads.md).

## Related documents

- [Architecture](./architecture.md) — the stages these records describe
- [Scheduler](./scheduler.md) — budgets, degradation hooks, and buffer underruns
- [Fleet serving](./fleet-serving.md) — per-robot failure isolation
