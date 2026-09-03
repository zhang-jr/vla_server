# Architecture

## The rate problem

A VLA policy has to satisfy three constraints at once that pull in different directions. The vision-language backbone is a multi-billion-parameter forward pass costing hundreds of milliseconds. The action head is comparatively tiny but runs an iterative procedure — integrating an ODE or denoising a sample — that must complete before the robot needs its next action. And the control loop itself needs a fresh joint command every 10 to 50 milliseconds, forever, with no gaps.

Running these synchronously means the robot stands still during every backbone forward pass. The runtime instead splits them into three stages that run concurrently at their own natural rates:

| Stage | Rate | Work | Where it can run |
|---|---|---|---|
| Perception | 1–5 Hz | `VLMBackend.encode()` — image and instruction to embedding | Cloud or a local GPU |
| Action generation | 5–20 Hz | `ActionHeadRuntime.predict()` — embedding and state to an action chunk | Close to the robot |
| Control | 20–100 Hz | Pop the next action from the buffer and send it | On the robot |

The load-bearing invariant is that inference latency stays below action execution time. As long as the action head produces a chunk covering more wall-clock time than the pipeline takes to produce it, the control loop always has actions queued and never stalls. When that inequality is violated, the runtime degrades along an explicit path rather than blocking.

## Data flow

```
ActionRequest                     image [C,H,W] + instruction + robot_state
     |
     |  Stage A — perception
     v
VLMBackend.encode()               VLM adapter, timeout-bounded
     |
     v
Embedding                         feature tensor + created_at + frame_id
     |
     |  Stage B — action generation
     v
ActionHeadRuntime.predict()       runs in an executor, off the event loop
     |
     v
ActionChunk                       actions [chunk_size, action_dim]
     |
     |  integrated under a blend policy
     v
ChunkBuffer                       latency trim + cross-fade with the queued tail
     |
     |  Stage C — control
     v
next_action()                     one action [action_dim] per control tick
```

Two boundaries in this chain matter more than the rest.

**The embedding boundary** is what keeps the backbone and the action head independent. They exchange nothing but an `Embedding`, so they can be quantized differently, compiled with different toolchains, and run on different machines. Robot state never crosses this boundary — it is passed as a separate tensor directly to the action head rather than being serialized into the prompt, which keeps the backbone's input identical regardless of what the robot is doing.

**The chunk buffer boundary** is what decouples the two rates. Stage B writes whole chunks; stage C reads one action per tick. Because these run at different frequencies, a new chunk almost always arrives while the previous one is still executing, and how those two are merged determines whether the resulting trajectory is smooth. See [Scheduler](./scheduler.md) for the blend policies.

## Concurrency model

The runtime is `asyncio`-based, and `AsyncScheduler` owns the event loop. Stages A and B are coroutines. Because `ActionHeadRuntime.predict()` is a synchronous CPU- or GPU-bound call, the scheduler dispatches it through `loop.run_in_executor()` so it cannot block the loop while other requests are in flight.

Within a single request the two stages are sequential — there is nothing to predict from until the embedding exists. Concurrency comes from two other places. Requests overlap each other, so one request's action generation proceeds while the next request's encode is already running. And stage C is decoupled from both by the chunk buffer rather than by awaiting anything: the control loop reads whatever the buffer holds at its own rate, which is why a slow forward pass slows the *arrival* of new actions without ever stalling the *execution* of the ones already queued.

The chunk buffer is the one component touched from outside the loop. It is designed for exactly one reader — the control loop, which may be a separate thread or process — and one writer, the action head. Its internals are lock-synchronized, and the expensive part of integrating a new chunk is computed outside the lock so a control tick is never blocked behind a blend. See `vla_runtime/scheduler/chunk_buffer.py`.

The action head interface is deliberately synchronous. Frequency guarantees are the scheduler's job, not the head's; a head implementation is a pure function from conditioning inputs to an action chunk, which keeps the three paradigms in `vla_runtime/action_head/` free of scheduling logic.

## Degradation paths

Every stage has a defined behavior for the case where it cannot meet its budget. These are hooks on `AsyncScheduler`, overridable by subclassing.

| Situation | Hook | Default behavior |
|---|---|---|
| Backbone exceeds its latency budget | `on_vlm_timeout()` | Reuse the last good embedding and count the frame as stale; raise `EmbeddingStaleError` once too many consecutive frames have been reused |
| Action head raises | `on_action_head_failure()` | Return a zero-action chunk marked as a fallback in its metadata |
| Rolling latency violates the budget | `on_budget_exceeded()` | Emit a structured event carrying the latency distribution |

Reusing a stale embedding is safe for a bounded number of frames because the scene usually changes little over a few hundred milliseconds, but the bound matters: past it, the policy is acting on a world that no longer exists, so the runtime stops rather than continuing to emit actions. The stale-frame counter lives in the chunk buffer and resets on the next successful encode.

## What each stage costs

Each stage records its own latency into a `BudgetManager`, which tracks a rolling window and exposes P50, P95, and a violation rate per module. The measured end-to-end latency is not only reported — it is fed back into the chunk buffer, which converts it into the number of leading chunk steps that were already stale by the time the chunk arrived. Latency compensation therefore tracks the pipeline's real behavior instead of a constant guessed at configuration time.

## Related documents

- [VLM backends](./vlm-backends.md) — stage A implementations and the embedding contract
- [Action heads](./action-heads.md) — stage B implementations and the conditioning contract
- [Scheduler](./scheduler.md) — the pipeline itself, chunk stitching, and budgets
- [Fleet serving](./fleet-serving.md) — running this pipeline for many robots at once
- [Observability](./observability.md) — the trace and metric surface each stage emits
