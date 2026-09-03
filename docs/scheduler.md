# Scheduler

The scheduler runs the three pipeline stages concurrently, measures what each costs, and keeps the control loop supplied with actions across the seams between successive inference results.

Source: `vla_runtime/scheduler/`

## AsyncScheduler

Source: `vla_runtime/scheduler/async_scheduler.py`

One `AsyncScheduler` serves one robot. It owns the event loop; business modules never call `asyncio.run()` themselves.

```python
scheduler = AsyncScheduler(
    vlm_backend=backend,
    action_head=head,
    budget_config=BudgetConfig(vlm_budget_ms=200, action_head_budget_ms=50),
    blend_policy="linear_overlap",
    control_rate_hz=50.0,
)
await scheduler.start()

handle = await scheduler.submit(request)
async for chunk in scheduler.stream(handle):
    send_to_robot(chunk)
```

`start()` warms up the backend before any request is accepted, so first-call compilation cost does not register as a timeout. `submit()` runs the request through perception and action generation and returns a `RequestHandle`. `stream()` yields the resulting chunks; it drains anything already queued before checking whether the request has completed, since a request submitted and awaited will frequently be finished by the time the caller reaches the loop.

Beyond the per-request stream, every produced chunk is also integrated into the scheduler's `ChunkBuffer`, which is what a control loop reads from continuously via `next_action()`.

### Constructor options

| Argument | Default | Purpose |
|---|---|---|
| `vlm_backend` | — | Any `VLMBackend` implementation |
| `action_head` | — | Any `ActionHeadRuntime` implementation |
| `budget_config` | `BudgetConfig()` | Per-stage latency budgets |
| `max_stale_frames` | 5 | Consecutive frames that may reuse a stale embedding |
| `embedding_queue_maxsize` | 2 | Bound on the perception-to-action-generation queue |
| `robot_id` | `"robot_0"` | Identifier attached to traces and metrics |
| `blend_policy` | `"linear_overlap"` | How a new chunk merges with the queued tail |
| `control_rate_hz` | 50.0 | Control loop rate, used to convert latency into chunk steps |
| `auto_latency_trim` | `True` | Feed measured latency back into the chunk trim |

### Degradation hooks

Three methods define what happens when a stage cannot meet its budget. Subclass to change any of them.

```python
async def on_vlm_timeout(self, request, last_embedding) -> Embedding
async def on_action_head_failure(self, request, error) -> ActionChunk
async def on_budget_exceeded(self, latency_ms, budget_ms) -> None
```

The default `on_vlm_timeout()` returns the last good embedding and marks the frame stale. If there is no previous embedding — a timeout on the very first frame — it raises `VLMTimeoutError`, because there is nothing to degrade to. The default `on_action_head_failure()` returns a zero-action chunk tagged `{"fallback": True}` in its metadata so downstream consumers and traces can tell a real prediction from a safe stop.

A useful override for `on_vlm_timeout()` is a smaller local model as a second-tier fallback; a useful override for `on_action_head_failure()` is holding the last commanded pose rather than commanding zeros, which is the safer behavior on hardware where zero is a valid but undesirable joint target.

## ChunkBuffer

Source: `vla_runtime/scheduler/chunk_buffer.py`

The chunk buffer decouples action generation from action execution. The action head writes whole chunks; the control loop pops one action per tick.

### The stitching problem

Because the two run at different rates, a new chunk nearly always arrives while the previous one is still executing. What happens at that moment determines whether the resulting motion is smooth.

Swapping wholesale — dropping the queued tail and restarting at step 0 of the new chunk — creates two distinct problems. The first is a discontinuity: the robot jumps from step *i* of one trajectory to step 0 of another, which is a visible jerk on hardware and is worst exactly where iterative heads are least consistent between samples. The second is a latency error: the new chunk's leading steps were predicted from an observation that is already one full pipeline latency old, so executing them from the beginning means replaying the past.

### Blend policies

Four policies address this, selected by `blend_policy`.

| Policy | Behavior |
|---|---|
| `replace` | Wholesale swap, no trim and no blending. Useful as a comparison baseline. |
| `trim` | Drop the leading `min(k, latency_trim_steps)` steps of the new chunk, where `k` is the number of actions consumed since the last integration, then replace. Fixes the latency error but not the discontinuity. |
| `linear_overlap` | `trim` plus a linear cross-fade across the overlap region — the queued tail fades from full weight to zero while the new chunk fades in. **Default.** |
| `ensemble` | Every chunk is anchored to the global step its first action targets, and each executed action is an `exp(-decay * i)` weighted average over all chunks still covering that step, weighting the oldest surviving prediction most. |

`linear_overlap` is the default because it addresses both problems at once and degrades gracefully: when the buffer has fully drained, the last executed action stands in as the old trajectory and is held for `blend_min_steps`, so the robot eases out of where it actually is rather than snapping to a fresh prediction's first step. `ensemble` averages away per-chunk prediction variance and is worth trying when a policy is noisy between samples, at the cost of holding several chunks at once.

### Latency compensation

`latency_trim_steps` is not a constant to be guessed at configuration time. `update_latency_trim(latency_ms)` derives it as `ceil(latency_ms / 1000 * control_rate_hz)` — the number of control steps that elapsed while the chunk was in flight, which is exactly the number of leading steps that were already stale on arrival. It is clamped by `max_latency_trim_steps` so one latency spike cannot swallow an entire chunk.

`AsyncScheduler` calls this after every completed inference when `auto_latency_trim` is on, so compensation tracks the pipeline's measured behavior. When a chunk arrives so late that the trim would consume all of it, the chunk is dropped and the queued tail keeps running; the trace records this as a drop rather than silently substituting stale actions.

### Seeding

`seed(pose)` anchors the buffer to the robot's measured pose and discards anything queued. Call it at episode start and after any reset, so the first blended chunk eases out of the robot's real position instead of assuming it was already following a prediction.

### Reader API

```python
buffer.next_action()   # one action [action_dim], or None if drained
buffer.current()       # remaining actions as an ActionChunk, or None
buffer.is_exhausted()  # True when nothing is queued
await buffer.wait_for_chunk(timeout=1.0)
```

`current()` is synthesized from live buffer state, so `current().actions[0]` is always exactly what the next `next_action()` will return — after trimming and blending, not the raw prediction that was pushed.

### Underrun tracking

When `next_action()` returns `None`, the control loop has nothing to execute. This is the failure that actually matters at runtime, so it is measured as an interval rather than a counter: the first `None` opens a gap, the next successful pop closes it, and an `action_buffer_gap` event records the gap's duration and how many polls it spanned. A mean underrun count says little; the tail duration is what tells you whether the loop stuttered.

### Concurrency

The buffer has one reader — the control loop, possibly on another thread — and one writer, the action head. Integrating a chunk computes the blend outside the lock and commits under an optimistic revision check, retrying if a pop landed mid-blend, so a control tick is never blocked behind a blend. A reader that polls without backoff would starve the writer that way, so after a bounded number of retries the blend runs under the lock instead; pushes that took that path are flagged `contended` in the trace.

## BudgetManager

Source: `vla_runtime/scheduler/budget_manager.py`

`BudgetManager` tracks latency per module over a rolling window.

```python
@dataclass
class BudgetConfig:
    vlm_budget_ms: float = 200.0
    action_head_budget_ms: float = 50.0
    total_budget_ms: float = 300.0
    window_size: int = 100
    degradation_threshold: float = 0.1
```

`record(module, latency_ms)` files a sample under `"vlm"`, `"action_head"`, or `"total"`. `percentile(module, p)` and `violation_rate(module)` read the window back, and `snapshot()` returns everything at once — the form the scheduler's `stats()` embeds.

`degradation_threshold` is what separates a hiccup from a regime change. A single request over budget is normal jitter; the `on_budget_exceeded` callback fires only once the fraction of recent requests exceeding budget crosses this threshold, which by default is ten percent. `check_and_raise()` is available for the stricter case where a single violation should raise `BudgetExceededError` immediately.

The `"total"` figure is the full observation-to-chunk-available latency — the sum of the perception and action generation stages, not the action head alone — because that is the number the chunk trim and the underlying real-time constraint both depend on.

## Related documents

- [Architecture](./architecture.md) — the pipeline the scheduler runs
- [VLM backends](./vlm-backends.md) — stage A
- [Action heads](./action-heads.md) — stage B
- [Fleet serving](./fleet-serving.md) — the multi-robot equivalent
- [Observability](./observability.md) — the events and metrics described above
