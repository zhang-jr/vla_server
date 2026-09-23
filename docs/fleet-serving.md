# Fleet Serving

Serving many robots from one process is worthwhile for a specific reason: the vision-language backbone is the expensive component, and one loaded instance of it can serve every robot. What is shared is the instance and its weights, not the computation — each robot sends different images, so every request still runs its own forward pass. Everything downstream of the embedding is per-robot state that must stay strictly isolated.

Source: `vla_runtime/serving/fleet.py`

## What is shared and what is not

| Component | Shared | Reason |
|---|---|---|
| VLM backbone | One instance for the fleet | The largest weights, and the one piece with no per-robot state; forward passes run per request |
| Action head instance | One per robot | Carries per-robot intermediate state |
| Action head weights | One copy for the fleet | Read-only; each robot's instance is `spawn()`ed from a prototype and reads the same `SharedWeights`. Registering one instance for two robots is refused, since that would share state too |
| Latency budget tracking | One per robot | Robots differ in cadence and payload |
| Stale-embedding counter | One per robot | One robot's degradation must not implicate another |

This is deliberately not the batching model used in LLM serving. Requests there can be interleaved freely because they share nothing but the model; here every request belongs to a physical machine with its own real-time deadline and its own state. Merging them into one batch would couple the deadlines together, and one slow robot would drag the rest with it. Sharing stops at the backbone instance: the dispatch loop encodes one request at a time, in priority order.

## Usage

```python
policy = Policy("pi0-libero", capabilities=caps, vlm_backend=backend, head_prototype=head)
fleet = FleetScheduler.from_policy(policy, FleetConfig(max_robots=8, vlm_budget_ms=200.0))
await fleet.start()

await fleet.register_robot("arm_left", priority=1)    # head spawned from the prototype
await fleet.register_robot("arm_right", priority=1)

handle = await fleet.submit("arm_left", request)
async for chunk in fleet.stream(handle):
    send_to_robot(chunk)
```

`Policy` (`vla_runtime/policy.py`) is the model adapter layer: it holds the backbone, a head prototype and the `PolicyCapabilities`, and at construction checks that the head can read the declared condition kind and that its paradigm matches the declared head structure. `FleetScheduler(vlm_backend=..., capabilities=...)` with explicitly passed heads still works for heads that cannot share weights.

`start()` warms up the backbone and launches the dispatch task. `register_robot()` allocates a slot holding that robot's action head, priority, budget tracker, and stale-embedding state. `deregister_robot()` releases it and is a no-op for an unknown id, so teardown does not need to track what it already removed.

### Configuration

```python
@dataclass
class FleetConfig:
    max_robots: int = 8
    vlm_queue_maxsize: int = 32
    priority_levels: int = 3
    vlm_budget_ms: float = 200.0
    action_head_budget_ms: float = 50.0
```

How long a reused condition stays usable is not a fleet setting: it is the policy's calibrated `max_condition_staleness_ms`.

Registering beyond `max_robots` raises `EdgeResourceError` rather than degrading everyone already connected.

## Priority scheduling

Requests enter a priority heap; the dispatch loop drains it in priority order, with 0 the highest. `priority_levels` sets how many levels exist.

Priority defaults to the value given at `register_robot()` and can be raised for a single request with `submit(robot_id, request, priority_override=0)`. The intent is that a safety-critical action — a stop, a reflex, an obstacle response — reaches the backbone ahead of routine manipulation traffic already queued in front of it, even when the fleet is saturated.

`submit()` raises `KeyError` for an unregistered robot id, so a request for a robot that was deregistered mid-run fails at submission rather than being queued and silently discarded later.

## Back-pressure

When the queue reaches `vlm_queue_maxsize`, `submit()` raises `EdgeResourceError` immediately instead of accepting work the fleet cannot serve in time. Silently accepting it would produce actions that arrive after their deadline, which is worse than a caller that knows it was rejected and can hold position.

## Per-robot degradation

Backbone timeouts are handled per robot. Each slot keeps its own `last_embedding` and `stale_frames` counter, so one robot's fallback does not affect any other robot's state, and an error for one robot leaves the rest running. What the fallback is follows the policy's frequency mode, as in [Scheduler](./scheduler.md): a decoupled policy reuses that robot's last condition while it is within `max_condition_staleness_ms` and fails that request with `EmbeddingStaleError` past it; a coupled policy produces no chunk, marks the handle `degraded="vlm_timeout_hold_tail"`, and leaves the robot executing its queued tail. A timeout with no prior condition for that robot raises `VLMTimeoutError` in either mode.

## Failure isolation

The dispatch loop is a single background task shared by the whole fleet, which makes an escaping exception unusually costly: the task dies, no handle is ever resolved, and every stream waits forever with nothing to indicate that anything went wrong. A silent hang is the worst available failure mode, because it is indistinguishable from slow inference while calling for the opposite response.

Failures are therefore contained per request. An exception from processing one request is recorded on that request's handle, the handle is released so its stream terminates, a `fleet_request_failed` event is emitted, and the loop continues serving every other robot. The error surfaces from the owning caller's own `stream()` call once the queue drains — reaching the caller that actually issued the request, rather than a background task that belongs to no one.

```python
handle = await fleet.submit("arm_left", request)
try:
    async for chunk in fleet.stream(handle):
        send_to_robot(chunk)
except SomeBackendError:
    hold_position("arm_left")   # other robots are unaffected
```

`asyncio.CancelledError` is excluded from this containment: the handle is released and the cancellation propagates normally, since a cancelled fleet is being shut down rather than recovering.

## Introspection

`stats()` returns registered robots, queue depth, running state, and a per-robot view carrying priority, stale frame count, and latency percentiles. Since backbone contention is the thing that goes wrong at scale, queue depth alongside per-robot latency percentiles is usually the pair that identifies whether the fleet is oversubscribed or a single robot is misbehaving.

## Related documents

- [Architecture](./architecture.md) — the pipeline each robot runs through
- [Scheduler](./scheduler.md) — the single-robot equivalent and its degradation hooks
- [VLM backends](./vlm-backends.md) — the shared backbone and its KV cache
- [Observability](./observability.md) — fleet events and metrics
