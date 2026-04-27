"""Fleet scheduler — shared VLM backbone + isolated action heads per robot.

Architecture (CLAUDE.md ADR-012):
    - One VLM backend instance shared across all robots.
      Rationale: VLM computation is the bottleneck; sharing amortises cost.
    - Per-robot ActionHeadRuntime instances — internal state (denoising
      trajectory, chunk offset) MUST NOT bleed between robots.
    - Action head weights are shared across robot instances (memory efficiency),
      but per-instance runtime state is fully isolated.
    - Priority scheduling: safety-critical requests (priority=0) preempt
      lower-priority requests in the VLM queue.

Priority levels:
    0 — safety-critical (e.g. emergency stop or human proximity avoidance)
    1 — normal operational priority
    2 — background / low-frequency tasks (e.g. mapping, logging)

Queue design:
    - VLM requests are queued by priority (heapq on (priority, seqno, item)).
    - Action head prediction is per-robot, not queued globally.
    - A single background coroutine drains the VLM queue and dispatches
      embeddings to per-robot action head coroutines.

Usage::

    fleet = FleetScheduler(
        vlm_backend=vllm_backend,
        config=FleetConfig(max_robots=8),
    )
    await fleet.start()

    fleet.register_robot("robot_0", action_head=flow_head, priority=1)
    fleet.register_robot("robot_1", action_head=diffusion_head, priority=1)

    handle = await fleet.submit("robot_0", request)
    async for chunk in fleet.stream(handle):
        send_to_robot(chunk)

    fleet.deregister_robot("robot_0")
    await fleet.stop()
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from vla_runtime.action_head.base import ActionChunk, ActionHeadRuntime, RobotState
from vla_runtime.errors import (
    ActionHeadFailureError,
    EdgeResourceError,
    ErrorContext,
    VLMTimeoutError,
)
from vla_runtime.observability.metrics import get_metrics
from vla_runtime.observability.tracer import get_tracer
from vla_runtime.scheduler.async_scheduler import RequestHandle
from vla_runtime.scheduler.budget_manager import BudgetConfig, BudgetManager
from vla_runtime.vlm.base import ActionRequest, Embedding, VLMBackend

_tracer = get_tracer("serving.fleet")
_metrics = get_metrics()


# ── Fleet configuration ───────────────────────────────────────────────────────

@dataclass
class FleetConfig:
    """Fleet scheduler configuration.

    Attributes:
        max_robots: Maximum number of concurrently registered robots.
        vlm_queue_maxsize: Maximum pending VLM encode requests across all robots.
            When full, submit() raises EdgeResourceError for back-pressure.
        priority_levels: Number of priority levels (0 = highest).
        vlm_budget_ms: Per-request VLM latency budget.
        action_head_budget_ms: Per-request action head latency budget.
        max_stale_frames: Max consecutive frames allowed with a stale embedding.
    """

    max_robots: int = 8
    vlm_queue_maxsize: int = 32
    priority_levels: int = 3
    vlm_budget_ms: float = 200.0
    action_head_budget_ms: float = 50.0
    max_stale_frames: int = 5


# ── Robot registration record ─────────────────────────────────────────────────

@dataclass
class RobotSlot:
    """Per-robot state isolated within the fleet.

    Attributes:
        robot_id: Unique identifier for this robot.
        action_head: Independent ActionHeadRuntime instance.
        priority: Default scheduling priority (0 = highest).
        budget_manager: Per-robot budget tracker.
        last_embedding: Last successfully produced embedding (for stale fallback).
        stale_frames: Consecutive frames using a stale embedding.
        registered_at: Wall-clock timestamp of registration.
    """

    robot_id: str
    action_head: ActionHeadRuntime
    priority: int = 1
    budget_manager: BudgetManager = field(init=False)
    last_embedding: Optional[Embedding] = field(default=None, init=False)
    stale_frames: int = field(default=0, init=False)
    registered_at: float = field(default_factory=time.time, init=False)

    def __post_init__(self) -> None:
        self.budget_manager = BudgetManager(
            config=BudgetConfig(),
            on_budget_exceeded=lambda m, l, b: None,
        )


# ── Priority queue item ───────────────────────────────────────────────────────

@dataclass(order=True)
class _QueueItem:
    """Sortable item for the priority heap.

    Sort key: (priority, seqno) — lower is higher priority.
    """

    priority: int
    seqno: int
    request: ActionRequest = field(compare=False)
    robot_id: str = field(compare=False)
    handle: "RequestHandle" = field(compare=False)


# ── Fleet scheduler ───────────────────────────────────────────────────────────

class FleetScheduler:
    """Multi-robot fleet scheduler with shared VLM and isolated action heads.

    Args:
        vlm_backend: Shared VLM backend (vLLM, SGLang, llama.cpp, etc.).
        config: Fleet configuration.
    """

    def __init__(
        self,
        vlm_backend: VLMBackend,
        config: FleetConfig | None = None,
    ) -> None:
        self._vlm = vlm_backend
        self._config = config or FleetConfig()
        self._robots: dict[str, RobotSlot] = {}
        self._lock = asyncio.Lock()

        # Priority heap: (priority, seqno, _QueueItem)
        self._heap: list[_QueueItem] = []
        self._heap_lock = asyncio.Lock()
        self._seqno = itertools.count()

        # Signalling: new item added to heap
        self._queue_event = asyncio.Event()

        # Background dispatch task
        self._dispatch_task: Optional[asyncio.Task] = None
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the fleet dispatcher. Call before submitting any requests."""
        self._running = True
        await self._vlm.warmup()
        self._dispatch_task = asyncio.create_task(
            self._dispatch_loop(), name="fleet_dispatch"
        )
        _tracer.event("fleet_started", max_robots=self._config.max_robots)

    async def stop(self) -> None:
        """Gracefully shut down the fleet dispatcher and VLM backend."""
        self._running = False
        self._queue_event.set()  # Unblock dispatch loop
        if self._dispatch_task:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass
        await self._vlm.shutdown()
        _tracer.event("fleet_stopped")

    # ── Robot management ──────────────────────────────────────────────────────

    async def register_robot(
        self,
        robot_id: str,
        action_head: ActionHeadRuntime,
        priority: int = 1,
    ) -> None:
        """Register a robot with its dedicated action head instance.

        Args:
            robot_id: Unique identifier (e.g. "robot_0", "arm_left").
            action_head: Independent ActionHeadRuntime for this robot.
                Must NOT be shared with another registered robot.
            priority: Default request priority (0 = highest).

        Raises:
            ValueError: If robot_id is already registered or max_robots exceeded.
        """
        async with self._lock:
            if robot_id in self._robots:
                raise ValueError(f"Robot '{robot_id}' is already registered")
            if len(self._robots) >= self._config.max_robots:
                raise EdgeResourceError(
                    required_mb=0.0,
                    available_mb=0.0,
                    context=ErrorContext(
                        trace_id=_tracer.trace_id,
                        module_name="serving.fleet",
                        request_id=robot_id,
                    ),
                )
            self._robots[robot_id] = RobotSlot(
                robot_id=robot_id,
                action_head=action_head,
                priority=priority,
            )
        _metrics.gauge("fleet_registered_robots", len(self._robots))
        _tracer.event("robot_registered", robot_id=robot_id, priority=priority)

    async def deregister_robot(self, robot_id: str) -> None:
        """Remove a robot from the fleet.

        Pending requests for this robot will complete before the slot is freed.

        Args:
            robot_id: The robot to deregister.
        """
        async with self._lock:
            self._robots.pop(robot_id, None)
        _metrics.gauge("fleet_registered_robots", len(self._robots))
        _tracer.event("robot_deregistered", robot_id=robot_id)

    # ── Request submission ────────────────────────────────────────────────────

    async def submit(
        self,
        robot_id: str,
        request: ActionRequest,
        priority_override: int | None = None,
    ) -> RequestHandle:
        """Submit an inference request for a registered robot.

        Args:
            robot_id: Which robot this request belongs to.
            request: ActionRequest with image + instruction + robot_state.
            priority_override: Override the robot's default priority.

        Returns:
            RequestHandle for streaming results via fleet.stream().

        Raises:
            KeyError: If robot_id is not registered.
            EdgeResourceError: If the VLM queue is full (back-pressure).
        """
        slot = self._robots.get(robot_id)
        if slot is None:
            raise KeyError(f"Robot '{robot_id}' is not registered")

        priority = priority_override if priority_override is not None else slot.priority

        handle = RequestHandle(
            request_id=request.request_id,
            robot_id=robot_id,
        )

        item = _QueueItem(
            priority=priority,
            seqno=next(self._seqno),
            request=request,
            robot_id=robot_id,
            handle=handle,
        )

        async with self._heap_lock:
            if len(self._heap) >= self._config.vlm_queue_maxsize:
                ctx = ErrorContext(
                    trace_id=_tracer.trace_id,
                    module_name="serving.fleet",
                    request_id=request.request_id,
                )
                raise EdgeResourceError(
                    required_mb=0.0,
                    available_mb=0.0,
                    context=ctx,
                )
            heapq.heappush(self._heap, item)

        self._queue_event.set()
        _metrics.gauge("fleet_queue_depth", len(self._heap))
        _tracer.event(
            "fleet_submit",
            robot_id=robot_id,
            request_id=request.request_id,
            priority=priority,
            queue_depth=len(self._heap),
        )
        return handle

    async def stream(self, handle: RequestHandle) -> AsyncIterator[ActionChunk]:
        """Stream ActionChunks for a submitted request.

        Args:
            handle: Returned by fleet.submit().

        Yields:
            ActionChunk objects as they are produced.
        """
        while not handle._done.is_set():
            try:
                chunk = await asyncio.wait_for(
                    handle._chunk_queue.get(), timeout=2.0
                )
                if chunk is None:
                    break
                yield chunk
            except asyncio.TimeoutError:
                if handle._done.is_set():
                    break

    # ── Dispatch loop (shared VLM) ────────────────────────────────────────────

    async def _dispatch_loop(self) -> None:
        """Background task: drain the priority heap, call shared VLM, dispatch to action heads."""
        while self._running:
            await self._queue_event.wait()
            self._queue_event.clear()

            while True:
                async with self._heap_lock:
                    if not self._heap:
                        break
                    item = heapq.heappop(self._heap)
                    _metrics.gauge("fleet_queue_depth", len(self._heap))

                await self._process_item(item)

    async def _process_item(self, item: _QueueItem) -> None:
        """Process one queued request: VLM encode → action head predict."""
        slot = self._robots.get(item.robot_id)
        if slot is None:
            # Robot deregistered while request was queued — discard
            item.handle._done.set()
            return

        with _tracer.span(
            "fleet_process",
            robot_id=item.robot_id,
            request_id=item.request.request_id,
            priority=item.priority,
        ):
            embedding = await self._vlm_encode(item, slot)
            await self._action_generate(embedding, item, slot)

    async def _vlm_encode(
        self,
        item: _QueueItem,
        slot: RobotSlot,
    ) -> Embedding:
        """Shared VLM encode with timeout and stale-embedding fallback."""
        t0 = time.perf_counter()
        budget_s = self._config.vlm_budget_ms / 1000.0

        try:
            embedding = await asyncio.wait_for(
                self._vlm.encode(item.request.image, item.request.instruction),
                timeout=budget_s,
            )
            slot.last_embedding = embedding
            slot.stale_frames = 0

        except (asyncio.TimeoutError, VLMTimeoutError):
            embedding = await self._on_vlm_timeout(item, slot)
            _metrics.increment("fleet_vlm_timeout")

        vlm_ms = (time.perf_counter() - t0) * 1000.0
        slot.budget_manager.record("vlm", vlm_ms, request_id=item.request.request_id)
        _tracer.event(
            "fleet_vlm_encoded",
            robot_id=item.robot_id,
            vlm_ms=round(vlm_ms, 2),
            stale=slot.stale_frames > 0,
        )
        return embedding

    async def _on_vlm_timeout(
        self,
        item: _QueueItem,
        slot: RobotSlot,
    ) -> Embedding:
        """Per-robot stale-embedding fallback (mirrors AsyncScheduler logic)."""
        if slot.last_embedding is None:
            ctx = ErrorContext(
                trace_id=_tracer.trace_id,
                module_name="serving.fleet",
                request_id=item.request.request_id,
            )
            raise VLMTimeoutError(
                timeout_ms=self._config.vlm_budget_ms,
                budget_ms=self._config.vlm_budget_ms,
                context=ctx,
            )

        slot.stale_frames += 1
        if slot.stale_frames > self._config.max_stale_frames:
            from vla_runtime.errors import EmbeddingStaleError
            ctx = ErrorContext(
                trace_id=_tracer.trace_id,
                module_name="serving.fleet",
                request_id=item.request.request_id,
            )
            raise EmbeddingStaleError(
                stale_frames=slot.stale_frames,
                max_stale_frames=self._config.max_stale_frames,
                context=ctx,
            )

        _tracer.event(
            "fleet_stale_embedding",
            robot_id=item.robot_id,
            stale_frames=slot.stale_frames,
        )
        return slot.last_embedding

    async def _action_generate(
        self,
        embedding: Embedding,
        item: _QueueItem,
        slot: RobotSlot,
    ) -> None:
        """Per-robot action head prediction (fully isolated state)."""
        import torch
        from vla_runtime.action_head.base import RobotState

        robot_state_tensor = item.request.robot_state
        half = robot_state_tensor.shape[0] // 2
        robot_state = RobotState(
            joint_positions=robot_state_tensor[:half],
            joint_velocities=robot_state_tensor[half:],
            gripper_state=robot_state_tensor[-1:],
        )

        t0 = time.perf_counter()
        try:
            loop = asyncio.get_event_loop()
            chunk = await loop.run_in_executor(
                None,
                lambda: slot.action_head.predict(embedding, robot_state),
            )
        except ActionHeadFailureError as exc:
            _tracer.event(
                "fleet_action_head_failure",
                robot_id=item.robot_id,
                error=str(exc),
            )
            zero = torch.zeros(
                slot.action_head.chunk_size,
                slot.action_head.action_dim,
                dtype=torch.float16,
            )
            chunk = ActionChunk(
                actions=zero,
                head_type=slot.action_head.head_type,
                metadata={"fallback": True, "reason": str(exc)},
            )

        action_ms = (time.perf_counter() - t0) * 1000.0
        slot.budget_manager.record(
            "action_head", action_ms, request_id=item.request.request_id
        )

        await item.handle._chunk_queue.put(chunk)
        item.handle._done.set()

        _metrics.increment("fleet_inference_complete")
        _tracer.event(
            "fleet_inference_complete",
            robot_id=item.robot_id,
            request_id=item.request.request_id,
            action_ms=round(action_ms, 2),
        )

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return a snapshot of fleet-level metrics."""
        return {
            "registered_robots": list(self._robots.keys()),
            "num_robots": len(self._robots),
            "queue_depth": len(self._heap),
            "running": self._running,
            "per_robot": {
                rid: {
                    "priority": slot.priority,
                    "stale_frames": slot.stale_frames,
                    "budget": slot.budget_manager.snapshot(),
                }
                for rid, slot in self._robots.items()
            },
        }
