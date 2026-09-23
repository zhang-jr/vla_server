"""AsyncScheduler — three-stage asynchronous inference pipeline.

Architecture (from CLAUDE.md):

    Thread A — perception (1-5 Hz)    : VLMBackend.encode()
    Thread B — action generation      : ActionHeadRuntime.predict()
    Thread C — control loop           : pops actions via ChunkBuffer.next_action()

All three stages run concurrently. Stages communicate via:
    - A → B: asyncio.Queue[Embedding]
    - B → C: ChunkBuffer (latency-trimmed, blended chunk stitching — ADR-014)

Frequency mode (ADR-018) comes from the policy's PolicyCapabilities:
    decoupled : the head may re-run on a cached condition with fresh robot
                state — replan(), and VLM-timeout fallback via on_vlm_timeout()
    coupled   : every head call is paired with a fresh VLM encode; on a VLM
                timeout the head is NOT re-run (it would only resample noise
                around an outdated plan) and the control loop keeps consuming
                the queued chunk tail

Degradation paths (CLAUDE.md §AsyncScheduler):
    on_vlm_timeout()    : decoupled only — reuse the last condition while it is
                          within max_condition_staleness_ms
    on_action_head_failure(): return zero-action fallback chunk
    on_budget_exceeded(): log + adjust budget config

The scheduler owns the event loop. Business modules must NOT call asyncio.run().

Usage::

    scheduler = AsyncScheduler(
        vlm_backend=vllm_backend,
        action_head=flow_matching_head,
        capabilities=PolicyCapabilities(
            state_placement="suffix",
            condition_kind="pooled",
            head_structure="iterative",
            max_condition_staleness_ms=300.0,
        ),
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
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

from vla_runtime.action_head.base import ActionChunk, ActionHeadRuntime, RobotState
from vla_runtime.capabilities import PolicyCapabilities
from vla_runtime.errors import (
    ActionHeadFailureError,
    EmbeddingStaleError,
    ErrorContext,
    PipelineError,
    PolicyCapabilityError,
    VLMTimeoutError,
)
from vla_runtime.observability.metrics import get_metrics
from vla_runtime.observability.tracer import get_tracer
from vla_runtime.scheduler.budget_manager import BudgetConfig, BudgetManager
from vla_runtime.scheduler.chunk_buffer import ChunkBlendPolicy, ChunkBuffer
from vla_runtime.vlm.base import ActionRequest, Embedding, VLMBackend

if TYPE_CHECKING:
    from vla_runtime.policy import Policy

_tracer = get_tracer("scheduler")
_metrics = get_metrics()


# ── Request handle ────────────────────────────────────────────────────────────

@dataclass
class RequestHandle:
    """Opaque handle returned by AsyncScheduler.submit().

    Use with scheduler.stream() to consume action chunks.

    Attributes:
        error: Set when the request failed inside a dispatcher that serves more
            than one request (FleetScheduler), where raising in place would take
            down every other robot. stream() re-raises it once the queue drains.
        degraded: Why the request completed without a new chunk, if it did —
            e.g. ``"vlm_timeout_hold_tail"`` for a coupled policy whose VLM
            timed out. The stream then ends empty and the control loop keeps
            consuming what is already queued.
    """

    request_id: str
    robot_id: str
    error: Optional[Exception] = None
    degraded: Optional[str] = None
    _chunk_queue: asyncio.Queue = field(default_factory=asyncio.Queue, repr=False)
    _done: asyncio.Event = field(default_factory=asyncio.Event, repr=False)


# ── AsyncScheduler ────────────────────────────────────────────────────────────

class AsyncScheduler:
    """Three-stage async pipeline for VLA inference.

    Args:
        vlm_backend: VLMBackend implementation (vLLM / SGLang / llama.cpp).
        action_head: ActionHeadRuntime implementation.
        capabilities: What the loaded policy supports (ADR-018). Required:
            whether state goes to the backbone and whether the head may re-run
            on a cached condition are facts of the model, not scheduler
            defaults.
        budget_config: Latency budget configuration.
        embedding_queue_maxsize: Backpressure limit on the A→B queue.
        robot_id: Identifier for this robot's pipeline instance.
        blend_policy: How an incoming chunk merges with the actions still queued
            (ADR-014). Defaults to the latency-trimmed linear cross-fade.
        control_rate_hz: Control loop rate, used to convert measured latency into
            a chunk trim in steps.
        auto_latency_trim: When True, every completed inference feeds its measured
            end-to-end latency back into the buffer's trim, so compensation
            tracks real pipeline behaviour instead of a hardcoded constant.
    """

    def __init__(
        self,
        vlm_backend: VLMBackend,
        action_head: ActionHeadRuntime,
        capabilities: PolicyCapabilities,
        budget_config: BudgetConfig | None = None,
        embedding_queue_maxsize: int = 2,
        robot_id: str = "robot_0",
        blend_policy: ChunkBlendPolicy = "linear_overlap",
        control_rate_hz: float = 50.0,
        auto_latency_trim: bool = True,
    ) -> None:
        self._vlm = vlm_backend
        self._action_head = action_head
        self._caps = capabilities
        self._robot_id = robot_id
        self._budget = BudgetManager(
            config=budget_config or BudgetConfig(),
            on_budget_exceeded=self._on_budget_exceeded_sync,
        )
        self._chunk_buffer = ChunkBuffer(
            robot_id=robot_id,
            blend_policy=blend_policy,
            control_rate_hz=control_rate_hz,
        )
        self._auto_latency_trim = auto_latency_trim

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

    @classmethod
    def from_policy(
        cls,
        policy: "Policy",
        robot_id: str = "robot_0",
        **kwargs: Any,
    ) -> "AsyncScheduler":
        """Build a scheduler for one robot from a validated Policy.

        The robot gets its own head instance spawned from the policy's
        prototype (shared weights, private state).

        Args:
            policy: The loaded policy.
            robot_id: Identifier for this robot.
            **kwargs: Remaining AsyncScheduler options (budget_config, ...).
        """
        return cls(
            vlm_backend=policy.vlm_backend,
            action_head=policy.head_for_robot(),
            capabilities=policy.capabilities,
            robot_id=robot_id,
            **kwargs,
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background perception and action loops."""
        self._running = True
        await self._vlm.warmup()
        _tracer.event(
            "scheduler_started",
            robot_id=self._robot_id,
            frequency_mode=self._caps.effective_frequency_mode,
            staleness_uncalibrated=not self._caps.staleness_calibrated,
        )

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

    async def replan(
        self,
        robot_state: RobotState,
        request_id: Optional[str] = None,
    ) -> RequestHandle:
        """Re-run the action head on the cached condition with fresh robot state.

        This is what frequency decoupling buys: between two VLM refreshes the
        head can react to where the robot actually is. Only meaningful when
        the head consumes fresh input and the staleness tolerance is
        calibrated (``effective_frequency_mode == "decoupled"``).

        Args:
            robot_state: Current robot state.
            request_id: Optional ID; generated when omitted.

        Returns:
            RequestHandle for streaming the resulting chunk.

        Raises:
            PolicyCapabilityError: If the policy runs coupled.
            PipelineError: If no condition has been produced yet.
            EmbeddingStaleError: If the cached condition is older than
                max_condition_staleness_ms.
        """
        request_id = request_id or uuid.uuid4().hex[:8]
        ctx = ErrorContext(
            trace_id=_tracer.trace_id,
            module_name="scheduler.replan",
            request_id=request_id,
        )
        if self._caps.effective_frequency_mode != "decoupled":
            raise PolicyCapabilityError(
                "replan() needs a decoupled policy; this one runs coupled "
                f"(state_placement={self._caps.state_placement!r}, "
                f"head_structure={self._caps.head_structure!r}, "
                f"staleness_calibrated={self._caps.staleness_calibrated})",
                ctx,
            )
        if self._last_embedding is None:
            raise PipelineError("replan() before any condition was produced", ctx)
        self._check_condition_fresh(self._last_embedding, ctx)

        handle = RequestHandle(request_id=request_id, robot_id=self._robot_id)
        self._active_handles[request_id] = handle
        with _tracer.span(
            "scheduler_replan",
            request_id=request_id,
            condition_age_ms=round(self._last_embedding.age_ms, 1),
        ):
            await self._run_action_generation(
                self._last_embedding, request_id, robot_state, handle
            )
        return handle

    async def stream(self, handle: RequestHandle) -> AsyncIterator[ActionChunk]:
        """Async-iterate over ActionChunks as they are produced.

        Yields chunks until the request is complete or an error occurs.

        Args:
            handle: Obtained from submit().

        Chunks already queued are drained before the done flag is consulted:
        submit() awaits the whole pipeline, so by the time a caller reaches
        stream() the request is usually already complete with its chunk sitting
        in the queue. Checking done first would discard it.

        Yields:
            ActionChunk objects with actions [chunk_size, action_dim].
        """
        while True:
            if not handle._chunk_queue.empty():
                chunk = handle._chunk_queue.get_nowait()
                if chunk is None:
                    break
                yield chunk
                continue

            if handle._done.is_set():
                break

            try:
                chunk = await asyncio.wait_for(
                    handle._chunk_queue.get(), timeout=2.0
                )
            except asyncio.TimeoutError:
                if handle._done.is_set():
                    break
                continue

            if chunk is None:
                break
            yield chunk

    # ── Stage A: Perception (VLM encode) ─────────────────────────────────────

    async def _run_perception(self, request: ActionRequest, handle: RequestHandle) -> None:
        """Thread A — VLM encode with timeout and stale-embedding fallback."""
        ctx = ErrorContext(
            trace_id=_tracer.trace_id,
            module_name="scheduler.perception",
            request_id=request.request_id,
        )
        t_start = time.perf_counter()

        state_for_backbone = request.robot_state if self._caps.state_to_backbone else None

        with _tracer.span("perception_encode", request_id=request.request_id) as span:
            span.set_attribute("frequency_mode", self._caps.effective_frequency_mode)
            try:
                embedding = await asyncio.wait_for(
                    self._vlm.encode(request.images, request.instruction, state_for_backbone),
                    timeout=self._budget._config.vlm_budget_ms / 1000.0,
                )
                self._caps.check_condition(embedding, ctx)
                self._last_embedding = embedding
                self._chunk_buffer.reset_stale()
                span.set_attribute("condition_kind", embedding.kind)
                span.set_attribute("condition_bytes", embedding.nbytes)

            except (asyncio.TimeoutError, VLMTimeoutError):
                span.set_degraded("vlm_timeout")
                self._chunk_buffer.mark_stale()
                _metrics.increment("vlm_timeout_fallback")
                if self._caps.effective_frequency_mode == "coupled":
                    self._hold_tail(request, handle)
                    self._budget.record(
                        "vlm",
                        (time.perf_counter() - t_start) * 1000.0,
                        request_id=request.request_id,
                    )
                    return
                embedding = await self.on_vlm_timeout(request, self._last_embedding)

            except Exception as exc:
                span.set_error(exc)
                raise

            vlm_ms = (time.perf_counter() - t_start) * 1000.0
            self._budget.record("vlm", vlm_ms, request_id=request.request_id)
            span.set_attribute("vlm_latency_ms", round(vlm_ms, 2))

        # Stage B: Action head predict
        await self._run_action_generation(
            embedding, request.request_id, request.robot_state, handle, vlm_ms=vlm_ms
        )

    def _hold_tail(self, request: ActionRequest, handle: RequestHandle) -> None:
        """Coupled-policy VLM timeout: produce no chunk, keep the queued tail.

        Re-running a coupled head on the previous condition would replan from
        an observation the robot has already moved away from. Continuing the
        tail already in the buffer is the better of the two stale options; an
        empty buffer surfaces as an underrun gap the control loop handles.

        Raises:
            VLMTimeoutError: On the first frame, when nothing is queued at all.
        """
        if self._last_embedding is None:
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
        handle.degraded = "vlm_timeout_hold_tail"
        handle._done.set()
        _tracer.event(
            "vlm_timeout_hold_tail",
            request_id=request.request_id,
            stale_frames=self._chunk_buffer.stale_frames,
        )

    def _check_condition_fresh(self, embedding: Embedding, ctx: ErrorContext) -> None:
        """Raise EmbeddingStaleError if a reused condition exceeds the tolerance."""
        max_age_ms = self._caps.max_condition_staleness_ms
        if max_age_ms is not None and embedding.is_stale(max_age_ms):
            raise EmbeddingStaleError(embedding.age_ms, max_age_ms, ctx)

    # ── Stage B: Action generation ────────────────────────────────────────────

    async def _run_action_generation(
        self,
        embedding: Embedding,
        request_id: str,
        robot_state: RobotState,
        handle: RequestHandle,
        vlm_ms: float = 0.0,
    ) -> None:
        """Thread B — action head predict with budget tracking."""
        t_start = time.perf_counter()
        with _tracer.span("action_gen_predict", request_id=request_id) as span:
            try:
                # predict() is sync — run in executor to not block event loop
                loop = asyncio.get_event_loop()
                chunk = await loop.run_in_executor(
                    None,
                    lambda: self._action_head.predict(embedding, robot_state),
                )
            except ActionHeadFailureError as exc:
                span.set_error(exc)
                chunk = await self.on_action_head_failure(request_id, exc)

            action_ms = (time.perf_counter() - t_start) * 1000.0
            self._budget.record("action_head", action_ms, request_id=request_id)
            span.set_attribute("action_head_latency_ms", round(action_ms, 2))

        # Total pipeline latency: observation -> chunk available for execution.
        total_ms = vlm_ms + action_ms
        self._budget.record("total", total_ms, request_id=request_id)

        # Feed measured latency back into the chunk trim (ADR-014): the leading
        # steps of this chunk were predicted total_ms ago and are already stale.
        if self._auto_latency_trim:
            self._chunk_buffer.update_latency_trim(total_ms)

        # Integrate into the chunk buffer (Stage C reads from here)
        self._chunk_buffer.push(chunk)

        # Also push to per-request stream queue
        await handle._chunk_queue.put(chunk)
        handle._done.set()

        _tracer.event(
            "inference_complete",
            request_id=request_id,
            action_ms=round(action_ms, 2),
            total_ms=round(total_ms, 2),
            chunk_size=chunk.chunk_size,
            latency_trim_steps=self._chunk_buffer.latency_trim_steps,
        )
        _metrics.increment("inference_complete")

    # ── Degradation hooks (subclass-overridable) ──────────────────────────────

    async def on_vlm_timeout(
        self,
        request: ActionRequest,
        last_embedding: Optional[Embedding],
    ) -> Embedding:
        """Fallback for decoupled policies: reuse the last condition.

        The head then re-runs on that condition with this request's fresh
        robot state. Only called when the policy runs decoupled; coupled
        policies hold the queued tail instead (see _hold_tail).
        Subclasses may override to implement a more sophisticated fallback
        (e.g., a cached edge model or a lightweight backup VLM).

        Args:
            request: The timed-out request.
            last_embedding: Last successfully produced condition, or None.

        Returns:
            Condition to use for this frame (reused, within tolerance).

        Raises:
            VLMTimeoutError: If there is no fallback available.
            EmbeddingStaleError: If the last condition is older than
                max_condition_staleness_ms.
        """
        ctx = ErrorContext(
            trace_id=_tracer.trace_id,
            module_name="scheduler",
            request_id=request.request_id,
        )
        if last_embedding is None:
            raise VLMTimeoutError(
                timeout_ms=self._budget._config.vlm_budget_ms,
                budget_ms=self._budget._config.vlm_budget_ms,
                context=ctx,
            )

        self._check_condition_fresh(last_embedding, ctx)
        _tracer.event(
            "vlm_timeout_fallback",
            request_id=request.request_id,
            stale_frames=self._chunk_buffer.stale_frames,
            embedding_age_ms=round(last_embedding.age_ms, 1),
        )
        return last_embedding

    async def on_action_head_failure(
        self,
        request_id: str,
        error: Exception,
    ) -> ActionChunk:
        """Fallback: return a zero-action chunk (safe stop).

        Subclasses should override to use a simpler backup policy or
        the last known good chunk from the buffer.

        Args:
            request_id: ID of the failed request (a replan has no ActionRequest).
            error: The exception that caused the failure.

        Returns:
            A safe zero-action chunk.
        """
        import torch

        _tracer.event(
            "action_head_failure_fallback",
            request_id=request_id,
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
