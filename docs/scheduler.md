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
    capabilities=PolicyCapabilities(
        state_placement="suffix",          # pi0-like: state enters the action expert
        condition_kind="per_layer_kv",
        head_structure="iterative",
        max_condition_staleness_ms=300.0,  # calibrated from sim success rate
    ),
    budget_config=BudgetConfig(vlm_budget_ms=200, action_head_budget_ms=50),
    blend_policy="linear_overlap",
    control_rate_hz=50.0,
)
await scheduler.start()

handle = await scheduler.submit(request)
async for chunk in scheduler.stream(handle):
    send_to_robot(chunk)
```

`AsyncScheduler.from_policy(policy, robot_id=...)` builds the same scheduler from a validated `Policy` (`vla_runtime/policy.py`), spawning the robot's head from the policy's prototype so it shares weights with other robots' heads.

`start()` warms up the backend before any request is accepted, so first-call compilation cost does not register as a timeout. `submit()` runs the request through perception and action generation and returns a `RequestHandle`. `stream()` yields the resulting chunks; it drains anything already queued before checking whether the request has completed, since a request submitted and awaited will frequently be finished by the time the caller reaches the loop.

Beyond the per-request stream, every produced chunk is also integrated into the scheduler's `ChunkBuffer`, which is what a control loop reads from continuously via `next_action()`.

### Constructor options

| Argument | Default | Purpose |
|---|---|---|
| `vlm_backend` | — | Any `VLMBackend` implementation |
| `action_head` | — | Any `ActionHeadRuntime` implementation |
| `capabilities` | — (required) | The policy's `PolicyCapabilities`; see below |
| `budget_config` | `BudgetConfig()` | Per-stage latency budgets |
| `embedding_queue_maxsize` | 2 | Bound on the perception-to-action-generation queue |
| `robot_id` | `"robot_0"` | Identifier attached to traces and metrics |
| `blend_policy` | `"linear_overlap"` | How a new chunk merges with the queued tail |
| `control_rate_hz` | 50.0 | Control loop rate, used to convert latency into chunk steps |
| `auto_latency_trim` | `True` | Feed measured latency back into the chunk trim |

### Policy capabilities and frequency mode

Source: `vla_runtime/capabilities.py`

A VLA policy is trained end to end, so a few facts that decide how it may be scheduled belong to the model rather than to the runtime. They are declared once per policy in a `PolicyCapabilities` (ADR-018) and are a required constructor argument — the scheduler does not guess them.

| Field | Values | Meaning |
|---|---|---|
| `state_placement` | `prefix`, `suffix`, `head_encoder`, `none` | Where robot state is consumed. `prefix` means the backbone needs it, so the scheduler passes it to `encode()` |
| `condition_kind` | see `Embedding.kind` | What the head reads; every encode result is checked against it |
| `head_structure` | `iterative`, `single_pass`, `monolithic` | Solver-loop head, one-pass parallel decode, or a fused policy whose boundary cannot be split |
| `max_condition_staleness_ms` | float or `None` | How old a reused condition may be, calibrated per model and task suite from sim success rate |
| `supports_rtc` | bool | Real-time chunking support (ADR-015); requires an iterative head |

From these the policy derives its frequency mode. It is **decoupled** only when the head consumes fresh input of its own (`suffix` or `head_encoder`) and is iterative; otherwise it is **coupled**. A decoupled policy whose staleness tolerance is not yet calibrated still runs coupled (`effective_frequency_mode`), because reusing a condition without knowing how long it stays valid is a guess.

| | Decoupled | Coupled |
|---|---|---|
| Head re-run on a cached condition | Yes — `replan(robot_state)` | Refused with `PolicyCapabilityError` |
| Backbone timeout | Reuse the last condition if younger than `max_condition_staleness_ms`, else `EmbeddingStaleError` | No new chunk; the handle is marked `degraded="vlm_timeout_hold_tail"` and the control loop keeps executing the queued tail |

The coupled timeout behavior is deliberate. For a policy whose head reads nothing but the condition — SmolVLA and π0.5 put state in the backbone prefix — re-running the head on the previous condition produces a new sample of a plan made from an observation the robot has since moved away from. Continuing the tail that is already queued is the better of the two stale options, and an empty buffer shows up as an underrun gap rather than as a silent replay.

`replan()` is what frequency decoupling buys: between two backbone refreshes, the head reacts to where the robot actually is, without touching the backbone.

### Degradation hooks

Three methods define what happens when a stage cannot meet its budget. Subclass to change any of them.

```python
async def on_vlm_timeout(self, request, last_embedding) -> Embedding
async def on_action_head_failure(self, request_id, error) -> ActionChunk
async def on_budget_exceeded(self, latency_ms, budget_ms) -> None
```

`on_vlm_timeout()` is called for decoupled policies only. The default returns the last good condition if it is within `max_condition_staleness_ms`, and raises `EmbeddingStaleError` otherwise. If there is no previous condition — a timeout on the very first frame — it raises `VLMTimeoutError`, because there is nothing to degrade to; a coupled policy raises the same error in that case. `on_action_head_failure()` takes a request ID rather than a request, since a `replan()` has no `ActionRequest`. The default `on_action_head_failure()` returns a zero-action chunk tagged `{"fallback": True}` in its metadata so downstream consumers and traces can tell a real prediction from a safe stop.

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
