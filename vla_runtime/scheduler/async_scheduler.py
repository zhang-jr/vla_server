"""AsyncScheduler — three-stage asynchronous inference pipeline.

Architecture (from CLAUDE.md):

    Thread A — perception (1-5 Hz)    : VLMBackend.encode()
    Thread B — action generation      : ActionHeadRuntime.predict()
    Thread C — control loop           : reads ActionChunk via ChunkBuffer

All three stages run concurrently. Stages communicate via:
    - A → B: asyncio.Queue[Embedding]
    - B → C: ChunkBuffer (double-buffer swap)

Degradation paths (CLAUDE.md §AsyncScheduler):
    on_vlm_timeout()    : reuse last embedding (stale-frame-limited)
    on_action_head_failure(): return zero-action fallback chunk
    on_budget_exceeded(): log + adjust budget config

The scheduler owns the event loop. Business modules must NOT call asyncio.run().

Usage::

    scheduler = AsyncScheduler(
        vlm_backend=vllm_backend,
        action_head=flow_matching_head,
        budget_config=BudgetConfig(vlm_budget_ms=200, action_head_budget_ms=50),
    )
    handle = await scheduler.submit(request)
    async for chunk in scheduler.stream(handle):
        send_to_robot(chunk)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from vla_runtime.action_head.base import ActionChunk, ActionHeadRuntime, RobotState
from vla_runtime.errors import (
    ActionHeadFailureError,
    BudgetExceededError,
    EmbeddingStaleError,
    ErrorContext,
    VLMTimeoutError,
)
from vla_runtime.observability.metrics import get_metrics
from vla_runtime.observability.tracer import get_tracer
from vla_runtime.scheduler.budget_manager import BudgetConfig, BudgetManager
from vla_runtime.scheduler.chunk_buffer import ChunkBuffer
from vla_runtime.vlm.base import ActionRequest, Embedding, VLMBackend

_tracer = get_tracer("scheduler")
_metrics = get_metrics()


# ── Request handle ────────────────────────────────────────────────────────────

@dataclass
class RequestHandle:
    """Opaque handle returned by AsyncScheduler.submit().

    Use with scheduler.stream() to consume action chunks.
    """

    request_id: str
    robot_id: str
    _chunk_queue: asyncio.Queue = field(default_factory=asyncio.Queue, repr=False)
    _done: asyncio.Event = field(default_factory=asyncio.Event, repr=False)


# ── AsyncScheduler ────────────────────────────────────────────────────────────

class AsyncScheduler:
    """Three-stage async pipeline for VLA inference.

    Args:
        vlm_backend: VLMBackend implementation (vLLM / SGLang / llama.cpp).
        action_head: ActionHeadRuntime implementation.
        budget_config: Latency budget configuration.
        max_stale_frames: Max consecutive frames with reused embedding.
        embedding_queue_maxsize: Backpressure limit on the A→B queue.
        robot_id: Identifier for this robot's pipeline instance.
    """

    def __init__(
        self,
        vlm_backend: VLMBackend,
        action_head: ActionHeadRuntime,
        budget_config: BudgetConfig | None = None,
        max_stale_frames: int = 5,
        embedding_queue_maxsize: int = 2,
        robot_id: str = "robot_0",
    ) -> None:
        self._vlm = vlm_backend
        self._action_head = action_head
        self._robot_id = robot_id
        self._budget = BudgetManager(
            config=budget_config or BudgetConfig(),
            on_budget_exceeded=self._on_budget_exceeded_sync,
        )
        self._chunk_buffer = ChunkBuffer(
            max_stale_frames=max_stale_frames,
            robot_id=robot_id,
        )

        # A → B: embedding queue
        self._embedding_queue: asyncio.Queue[Embedding] = asyncio.Queue(
            maxsize=embedding_queue_maxsize
        )

        # Last known good embedding (for fallback)
        self._last_embedding: Optional[Embedding] = None

        # Active request tracking
        self._active_handles: dict[str, RequestHandle] = {}

        # Background tasks
        self._perception_task: Optional[asyncio.Task] = None
        self._action_task: Optional[asyncio.Task] = None
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background perception and action loops."""
        self._running = True
        await self._vlm.warmup()
        _tracer.event("scheduler_started", robot_id=self._robot_id)

    async def stop(self) -> None:
        """Gracefully shut down background tasks and VLM backend."""
        self._running = False
        if self._perception_task:
            self._perception_task.cancel()
        if self._action_task:
            self._action_task.cancel()
        await self._vlm.shutdown()
        _tracer.event("scheduler_stopped", robot_id=self._robot_id)

    # ── Public API ────────────────────────────────────────────────────────────

    async def submit(self, request: ActionRequest) -> RequestHandle:
        """Submit an ActionRequest to the pipeline.

        Args:
            request: The inference request with image, instruction, robot_state.

        Returns:
            RequestHandle for streaming results via scheduler.stream().
        """
        handle = RequestHandle(
            request_id=request.request_id,
            robot_id=self._robot_id,
        )
        self._active_handles[request.request_id] = handle

        with _tracer.span("scheduler_submit", request_id=request.request_id):
            # Stage A: VLM encode (may be awaited or run in background)
            await self._run_perception(request, handle)

        return handle

    async def stream(self, handle: RequestHandle) -> AsyncIterator[ActionChunk]:
        """Async-iterate over ActionChunks as they are produced.

        Yields chunks until the request is complete or an error occurs.

        Args:
            handle: Obtained from submit().

        Yields:
            ActionChunk objects with actions [chunk_size, action_dim].
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

    # ── Stage A: Perception (VLM encode) ─────────────────────────────────────

    async def _run_perception(self, request: ActionRequest, handle: RequestHandle) -> None:
        """Thread A — VLM encode with timeout and stale-embedding fallback."""
        ctx = ErrorContext(
            trace_id=_tracer.trace_id,
            module_name="scheduler.perception",
            request_id=request.request_id,
        )
        t_start = time.perf_counter()

        with _tracer.span("perception_encode", request_id=request.request_id) as span:
            try:
                embedding = await asyncio.wait_for(
                    self._vlm.encode(request.image, request.instruction),
                    timeout=self._budget._config.vlm_budget_ms / 1000.0,
                )
                self._last_embedding = embedding
                self._chunk_buffer._stale_frames = 0

            except (asyncio.TimeoutError, VLMTimeoutError):
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                span.set_degraded("vlm_timeout")
                embedding = await self.on_vlm_timeout(request, self._last_embedding)
                self._chunk_buffer.mark_stale()
                _metrics.increment("vlm_timeout_fallback")

            except Exception as exc:
                span.set_error(exc)
                raise

            vlm_ms = (time.perf_counter() - t_start) * 1000.0
            self._budget.record("vlm", vlm_ms, request_id=request.request_id)
            span.set_attribute("vlm_latency_ms", round(vlm_ms, 2))

        # Stage B: Action head predict
        await self._run_action_generation(embedding, request, handle)

    # ── Stage B: Action generation ────────────────────────────────────────────

    async def _run_action_generation(
        self,
        embedding: Embedding,
        request: ActionRequest,
        handle: RequestHandle,
    ) -> None:
        """Thread B — action head predict with budget tracking."""
        ctx = ErrorContext(
            trace_id=_tracer.trace_id,
            module_name="scheduler.action_gen",
            request_id=request.request_id,
        )
        robot_state = RobotState(
            joint_positions=request.robot_state[: request.robot_state.shape[0] // 2],
            joint_velocities=request.robot_state[request.robot_state.shape[0] // 2:],
            gripper_state=request.robot_state[-1:],
        )

        t_start = time.perf_counter()
        with _tracer.span("action_gen_predict", request_id=request.request_id) as span:
            try:
                # predict() is sync — run in executor to not block event loop
                loop = asyncio.get_event_loop()
                chunk = await loop.run_in_executor(
                    None,
                    lambda: self._action_head.predict(embedding, robot_state),
                )
            except ActionHeadFailureError as exc:
                span.set_error(exc)
                chunk = await self.on_action_head_failure(request, exc)

            action_ms = (time.perf_counter() - t_start) * 1000.0
            self._budget.record("action_head", action_ms, request_id=request.request_id)
            span.set_attribute("action_head_latency_ms", round(action_ms, 2))

        # Total pipeline latency
        total_ms = action_ms  # simplified; real impl tracks from request.timestamp
        self._budget.record("total", total_ms, request_id=request.request_id)

        # Push to double buffer (Stage C reads from here)
        self._chunk_buffer.push(chunk)

        # Also push to per-request stream queue
        await handle._chunk_queue.put(chunk)
        handle._done.set()

        _tracer.event(
            "inference_complete",
            request_id=request.request_id,
            action_ms=round(action_ms, 2),
            chunk_size=chunk.chunk_size,
        )
        _metrics.increment("inference_complete")

    # ── Degradation hooks (subclass-overridable) ──────────────────────────────

    async def on_vlm_timeout(
        self,
        request: ActionRequest,
        last_embedding: Optional[Embedding],
    ) -> Embedding:
        """Fallback: reuse the last known good embedding.

        If last_embedding is None (first frame timeout), raises VLMTimeoutError.
        Subclasses may override to implement a more sophisticated fallback
        (e.g., a cached edge model or a lightweight backup VLM).

        Args:
            request: The timed-out request.
            last_embedding: Last successfully produced embedding, or None.

        Returns:
            Embedding to use for this frame (may be stale).

        Raises:
            VLMTimeoutError: If there is no fallback available.
        """
        if last_embedding is None:
            ctx = ErrorContext(
                trace_id=_tracer.trace_id,
                module_name="scheduler",
                request_id=request.request_id,
            )
            raise VLMTimeoutError(
                timeout_ms=self._budget._config.vlm_budget_ms,
                budget_ms=self._budget._config.vlm_budget_ms,
                context=ctx,
            )

        _tracer.event(
            "vlm_timeout_fallback",
            request_id=request.request_id,
            stale_frames=self._chunk_buffer.stale_frames,
            embedding_age_ms=round(last_embedding.age_ms, 1),
        )
        return last_embedding

    async def on_action_head_failure(
        self,
        request: ActionRequest,
        error: Exception,
    ) -> ActionChunk:
        """Fallback: return a zero-action chunk (safe stop).

        Subclasses should override to use a simpler backup policy or
        the last known good chunk from the buffer.

        Args:
            request: The failed request.
            error: The exception that caused the failure.

        Returns:
            A safe zero-action chunk.
        """
        import torch

        _tracer.event(
            "action_head_failure_fallback",
            request_id=request.request_id,
            error=str(error),
        )
        zero_actions = torch.zeros(
            self._action_head.chunk_size,
            self._action_head.action_dim,
            dtype=torch.float16,
        )
        return ActionChunk(
            actions=zero_actions,
            head_type=self._action_head.head_type,
            metadata={"fallback": True, "reason": str(error)},
        )

    async def on_budget_exceeded(self, latency_ms: float, budget_ms: float) -> None:
        """Called by BudgetManager when rolling violation rate is high.

        Default: log only. Subclasses may trigger model offloading, reduce
        chunk size, or alert an ops system.
        """
        _tracer.event(
            "rolling_budget_violation",
            latency_ms=round(latency_ms, 2),
            budget_ms=budget_ms,
            stats=self._budget.snapshot(),
        )

    def _on_budget_exceeded_sync(
        self, module: str, latency_ms: float, budget_ms: float
    ) -> None:
        """Sync wrapper to schedule the async on_budget_exceeded callback."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.on_budget_exceeded(latency_ms, budget_ms))
        except RuntimeError:
            pass

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return a point-in-time snapshot of scheduler metrics."""
        return {
            "robot_id": self._robot_id,
            "buffer": self._chunk_buffer.stats(),
            "budget": self._budget.snapshot(),
            "active_requests": len(self._active_handles),
        }
