"""Action chunk buffer for the AsyncScheduler — latency-compensated chunk stitching.

The buffer decouples action execution (control loop, 20-100 Hz) from action
generation (5-20 Hz): while the robot executes buffered actions, the next chunk
is computed in parallel. This is the mechanism that keeps
``inference_latency < action_execution_time`` — the hard constraint from the NXP
validation in CLAUDE.md.

The non-trivial part is what happens when a new chunk arrives while the previous
one is still executing. A plain double buffer swaps wholesale: the robot jumps
from step ``i`` of the old chunk to step 0 of the new one. That is both

    * a discontinuity — a visible jerk on real hardware, worst at the boundary
      of flow-matching / diffusion trajectories, and
    * a latency error — the new chunk's leading steps were predicted from an
      observation that is already ``inference_latency`` old.

ADR-014 replaces the wholesale swap with four pluggable blend policies:

    ``replace``         Wholesale swap. Legacy double-buffer semantics, kept as
                        an ablation baseline. No trim, no blending.
    ``trim``            Drop the leading ``min(k, latency_trim_steps)`` steps of
                        the new chunk (``k`` = steps consumed since the last
                        integrate), then replace. Fixes latency, not continuity.
    ``linear_overlap``  ``trim`` plus a linear cross-fade over the overlap region
                        (old 100%->0%, new 0%->100%). **Default.**
    ``ensemble``        ACT-style temporal ensembling: each chunk is anchored to
                        the global step it starts at, and every executed action
                        is an ``exp(-decay * i)`` weighted average over all
                        chunks still covering that step.

``linear_overlap`` and ``ensemble`` are ported from EVA-Client's
``strategy/action_buffer.py`` (Apache-2.0, arXiv:2607.02646), which validated
them on eight real robots; ``ensemble`` follows ACT (arXiv:2304.13705).

``latency_trim_steps`` is never hardcoded: :meth:`ChunkBuffer.update_latency_trim`
derives it from the measured end-to-end latency times the control rate, so the
trim tracks the pipeline's actual behaviour (CLAUDE.md forbids hardcoded timeout
constants).

Thread safety:
    Blenders are self-synchronised. ``integrate`` computes the blend OUTSIDE the
    lock and commits under an optimistic revision check, retrying if the control
    loop consumed an action meanwhile — so ``next_action()`` is not blocked by
    the blend. A reader that polls without backoff would starve the writer that
    way, so after ``_MAX_OPTIMISTIC_RETRIES`` failed attempts the blend runs
    under the lock instead, trading one bounded stall for guaranteed progress.
    Pushes that took that path are reported as ``contended`` in the trace.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from typing import Literal, Optional

import torch

from vla_runtime.action_head.base import ActionChunk, ActionHeadType
from vla_runtime.observability.metrics import get_metrics
from vla_runtime.observability.tracer import get_tracer

_tracer = get_tracer("scheduler.chunk_buffer")
_metrics = get_metrics()

ChunkBlendPolicy = Literal["replace", "trim", "linear_overlap", "ensemble"]

# Optimistic blend attempts before falling back to blending under the lock. A
# control loop that polls without backoff bumps the revision on every pop and
# can otherwise starve the writer indefinitely.
_MAX_OPTIMISTIC_RETRIES = 4


# ── Blenders ──────────────────────────────────────────────────────────────────

class _Blender:
    """Base for chunk blenders — owns the pending actions and the ``k`` counter.

    ``k`` counts actions popped since the last successful integrate; it is what
    the latency trim consumes. Every blender is self-synchronised; ChunkBuffer
    holds no lock across these calls.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._revision: int = 0
        self._k: int = 0
        self._last_action: Optional[torch.Tensor] = None

    @property
    def consumed_since_integrate(self) -> int:
        """Actions popped since the last integrate (the ``k`` of the trim rule)."""
        with self._lock:
            return self._k

    @property
    def last_action(self) -> Optional[torch.Tensor]:
        """Most recently emitted action, or the seeded pose. None before either."""
        with self._lock:
            return self._last_action

    def integrate(self, actions: torch.Tensor, max_trim: int) -> dict:
        """Merge a fresh chunk into the buffer; returns trace info."""
        raise NotImplementedError

    def pop(self) -> Optional[torch.Tensor]:
        """Emit the next action [action_dim], or None when nothing is buffered."""
        raise NotImplementedError

    def peek(self) -> Optional[torch.Tensor]:
        """Actions still to execute, [n, action_dim], or None when drained."""
        raise NotImplementedError

    def remaining(self) -> int:
        """Number of actions still executable from the buffer."""
        raise NotImplementedError

    def seed(self, action: torch.Tensor) -> None:
        """Anchor the buffer to a known pose, dropping all buffered actions."""
        raise NotImplementedError

    def reset(self) -> None:
        """Drop all buffered actions and every trace of prior state."""
        raise NotImplementedError


class _SequentialBlender(_Blender):
    """Blenders backed by one pending ``[n, action_dim]`` tensor plus a cursor.

    Subclasses differ only in ``_trims`` (whether the latency trim applies) and
    ``_combine`` (how the surviving old tail merges with the new chunk).
    """

    _trims: bool = False

    def __init__(self) -> None:
        super().__init__()
        self._pending: Optional[torch.Tensor] = None
        self._cursor: int = 0

    def _combine(
        self,
        old: Optional[torch.Tensor],
        last_action: Optional[torch.Tensor],
        new: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Merge the un-executed tail ``old`` with the trimmed chunk ``new``.

        Must be pure with respect to blender state and must never touch
        ``self._lock``: on the contended fallback path this runs while the lock
        is already held, so any re-entry would deadlock.

        Args:
            old: Actions still queued from the previous chunk, or None.
            last_action: Last emitted action / seeded pose, or None.
            new: The incoming chunk after the latency trim.

        Returns:
            ``(combined, blended_steps)`` — the new pending tensor and how many
            of its leading steps are a blend rather than raw prediction.
        """
        raise NotImplementedError

    def _old_locked(self) -> Optional[torch.Tensor]:
        if self._pending is None or self._cursor >= self._pending.shape[0]:
            return None
        # A view is safe: _pending is replaced, never mutated in place.
        return self._pending[self._cursor:]

    def _snapshot(
        self,
    ) -> tuple[int, int, Optional[torch.Tensor], Optional[torch.Tensor]]:
        with self._lock:
            return self._revision, self._k, self._old_locked(), self._last_action

    def _plan(
        self,
        consumed: int,
        old: Optional[torch.Tensor],
        last_action: Optional[torch.Tensor],
        actions: torch.Tensor,
        max_trim: int,
    ) -> tuple[Optional[torch.Tensor], dict]:
        """Compute the merge without touching shared state.

        Returns:
            ``(combined, info)`` — the tensor to commit, or None when the chunk
            arrived fully stale and the queued tail should keep running.
        """
        trim = min(consumed, max(0, max_trim)) if self._trims else 0
        if trim >= actions.shape[0]:
            return None, {"trimmed": trim, "blended": 0, "dropped": True}
        combined, blended = self._combine(old, last_action, actions[trim:])
        return combined, {"trimmed": trim, "blended": blended, "dropped": False}

    def _commit_locked(self, combined: torch.Tensor) -> None:
        self._pending = combined
        self._cursor = 0
        self._k = 0
        self._revision += 1

    def integrate(self, actions: torch.Tensor, max_trim: int) -> dict:
        """Trim, combine, and commit — retrying if the control loop raced us.

        The blend runs outside the lock so a popping control loop is never
        blocked by it. If the loop invalidates the snapshot
        ``_MAX_OPTIMISTIC_RETRIES`` times in a row, the last attempt blends while
        holding the lock: a busy-polling reader would otherwise starve the writer
        indefinitely, and one bounded blend is a far smaller stall than a chunk
        that never lands.
        """
        for _ in range(_MAX_OPTIMISTIC_RETRIES):
            revision, consumed, old, last_action = self._snapshot()
            combined, info = self._plan(consumed, old, last_action, actions, max_trim)
            with self._lock:
                if revision != self._revision:
                    continue    # an action was popped mid-blend; redo with fresh k
                if combined is not None:
                    self._commit_locked(combined)
                info["contended"] = False
                return info

        with self._lock:
            combined, info = self._plan(
                self._k, self._old_locked(), self._last_action, actions, max_trim
            )
            if combined is not None:
                self._commit_locked(combined)
            info["contended"] = True
            return info

    def pop(self) -> Optional[torch.Tensor]:
        with self._lock:
            if self._pending is None or self._cursor >= self._pending.shape[0]:
                return None
            action = self._pending[self._cursor]
            self._cursor += 1
            self._k += 1
            self._last_action = action
            self._revision += 1
            return action

    def peek(self) -> Optional[torch.Tensor]:
        with self._lock:
            if self._pending is None or self._cursor >= self._pending.shape[0]:
                return None
            return self._pending[self._cursor:]

    def remaining(self) -> int:
        with self._lock:
            if self._pending is None:
                return 0
            return max(0, self._pending.shape[0] - self._cursor)

    def seed(self, action: torch.Tensor) -> None:
        with self._lock:
            self._pending = None
            self._cursor = 0
            self._k = 0
            self._last_action = action.detach().clone()
            self._revision += 1

    def reset(self) -> None:
        with self._lock:
            self._pending = None
            self._cursor = 0
            self._k = 0
            self._last_action = None
            self._revision += 1


class ReplaceBlender(_SequentialBlender):
    """Wholesale replacement — the legacy double-buffer semantics (ablation baseline)."""

    _trims = False

    def _combine(
        self,
        old: Optional[torch.Tensor],
        last_action: Optional[torch.Tensor],
        new: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Discard the queued tail; the new chunk takes over unchanged."""
        return new, 0


class TrimBlender(_SequentialBlender):
    """Latency-aligned whole-chunk replacement: trim the stale head, no blending."""

    _trims = True

    def _combine(
        self,
        old: Optional[torch.Tensor],
        last_action: Optional[torch.Tensor],
        new: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Discard the queued tail; the already-trimmed chunk takes over."""
        return new, 0


class LinearOverlapBlender(_SequentialBlender):
    """Latency trim plus a linear cross-fade across the overlap region.

    The un-executed tail of the previous chunk fades out (weight 1 -> 0) while
    the new chunk fades in (0 -> 1), so the executed trajectory never steps
    discontinuously onto a fresh prediction. When the buffer has fully drained,
    the last emitted action (or the seeded pose) stands in as the old trajectory,
    held for ``blend_min_steps``, so the robot eases out of where it actually is
    rather than snapping to the new chunk's first step.

    Args:
        blend_min_steps: Floor on the overlap length, so a nearly drained buffer
            still produces a gradual fade instead of a one-step jump.
    """

    _trims = True

    def __init__(self, blend_min_steps: int = 8) -> None:
        super().__init__()
        self._blend_min_steps = max(1, int(blend_min_steps))

    def _combine(
        self,
        old: Optional[torch.Tensor],
        last_action: Optional[torch.Tensor],
        new: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Cross-fade the queued tail into the incoming chunk."""
        if old is None or old.shape[0] == 0:
            if last_action is None:
                return new, 0
            # Drained buffer: hold the last executed pose as the old trajectory.
            old = last_action.reshape(1, -1).expand(self._blend_min_steps, new.shape[1])
        elif old.shape[0] < self._blend_min_steps:
            pad = old[-1:].expand(self._blend_min_steps - old.shape[0], old.shape[1])
            old = torch.cat([old, pad], dim=0)

        overlap = min(old.shape[0], new.shape[0])
        weights = torch.linspace(
            1.0, 0.0, overlap, dtype=torch.float32, device=new.device
        ).unsqueeze(1)
        blended = (
            old[:overlap].to(torch.float32) * weights
            + new[:overlap].to(torch.float32) * (1.0 - weights)
        ).to(new.dtype)
        return torch.cat([blended, new[overlap:]], dim=0), overlap


class EnsembleBlender(_Blender):
    """ACT-style temporal ensembling over every chunk covering the current step.

    Each chunk is stored against the global execution step its first action
    targets. The action emitted for step ``s`` is the normalised
    ``exp(-decay * i)`` weighted average of the prediction for ``s`` taken from
    every stored chunk that still covers it, ``i`` being the age rank among the
    survivors (oldest first, so the oldest surviving prediction weighs most).

    Latency is handled implicitly by the step anchoring, so this blender ignores
    the trim.

    Args:
        decay: ``m`` in ``exp(-m * i)``. Larger -> the oldest prediction
            dominates; smaller -> a flatter average.
    """

    def __init__(self, decay: float = 0.01) -> None:
        super().__init__()
        self._decay = max(0.0, float(decay))
        self._chunks: list[tuple[int, torch.Tensor]] = []
        self._step: int = 0

    def integrate(self, actions: torch.Tensor, max_trim: int) -> dict:
        """Anchor the chunk at the current execution step for later ensembling."""
        _ = max_trim    # step anchoring already accounts for elapsed latency
        with self._lock:
            self._chunks.append((self._step, actions))
            self._prune_locked()
            self._k = 0
            self._revision += 1
            return {
                "trimmed": 0,
                "blended": len(self._chunks),
                "dropped": False,
                "contended": False,
            }

    def pop(self) -> Optional[torch.Tensor]:
        with self._lock:
            predictions = self._gather_locked(self._step)
            if not predictions:
                return None
            action = self._aggregate(predictions)
            self._step += 1
            self._k += 1
            self._last_action = action
            self._prune_locked()
            self._revision += 1
            return action

    def peek(self) -> Optional[torch.Tensor]:
        with self._lock:
            rows: list[torch.Tensor] = []
            for step in range(self._step, self._horizon_end_locked()):
                predictions = self._gather_locked(step)
                if not predictions:
                    break
                rows.append(self._aggregate(predictions))
            return torch.stack(rows, dim=0) if rows else None

    def remaining(self) -> int:
        with self._lock:
            return max(0, self._horizon_end_locked() - self._step)

    def seed(self, action: torch.Tensor) -> None:
        with self._lock:
            self._chunks.clear()
            self._step = 0
            self._k = 0
            self._last_action = action.detach().clone()
            self._revision += 1

    def reset(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._step = 0
            self._k = 0
            self._last_action = None
            self._revision += 1

    def _aggregate(self, predictions: list[torch.Tensor]) -> torch.Tensor:
        """Exponentially decayed normalised average, oldest prediction weighted most."""
        if len(predictions) == 1:
            return predictions[0]
        stacked = torch.stack(predictions, dim=0).to(torch.float32)
        weights = torch.exp(
            -self._decay
            * torch.arange(len(predictions), dtype=torch.float32, device=stacked.device)
        )
        weights = (weights / weights.sum()).reshape(-1, 1)
        return (stacked * weights).sum(dim=0).to(predictions[0].dtype)

    def _gather_locked(self, step: int) -> list[torch.Tensor]:
        """Predictions for ``step`` from every covering chunk, oldest chunk first."""
        out: list[torch.Tensor] = []
        for anchor, chunk in self._chunks:
            offset = step - anchor
            if 0 <= offset < chunk.shape[0]:
                out.append(chunk[offset])
        return out

    def _horizon_end_locked(self) -> int:
        if not self._chunks:
            return self._step
        return max(anchor + chunk.shape[0] for anchor, chunk in self._chunks)

    def _prune_locked(self) -> None:
        """Drop chunks whose horizon no longer reaches the current step."""
        self._chunks = [
            (anchor, chunk)
            for anchor, chunk in self._chunks
            if anchor + chunk.shape[0] > self._step
        ]


_BLENDERS: dict[str, type] = {
    "replace": ReplaceBlender,
    "trim": TrimBlender,
    "linear_overlap": LinearOverlapBlender,
    "ensemble": EnsembleBlender,
}


# ── ChunkBuffer ───────────────────────────────────────────────────────────────

class ChunkBuffer:
    """Latency-compensated action chunk store shared by action head and control loop.

    The action head (Thread B) calls :meth:`push`; the control loop (Thread C)
    calls :meth:`next_action`. How an incoming chunk merges with the actions
    still queued is decided by ``blend_policy`` (ADR-014).

    Args:
        robot_id: Identifier for the robot owning this buffer (tracing).
        blend_policy: How a new chunk merges with the queued tail.
        latency_trim_steps: Initial cap on the leading steps dropped from a new
            chunk. Superseded by :meth:`update_latency_trim` once real latency
            has been measured.
        max_latency_trim_steps: Ceiling for the derived trim, so a single latency
            spike cannot swallow an entire chunk.
        blend_min_steps: Overlap floor for the ``linear_overlap`` policy.
        ensemble_decay: ``m`` in ``exp(-m * i)`` for the ``ensemble`` policy.
        control_rate_hz: Control loop rate, used to convert a measured latency
            into a step count in :meth:`update_latency_trim`.

    Raises:
        ValueError: If ``blend_policy`` is not one of the four known policies.
    """

    def __init__(
        self,
        robot_id: str = "robot_0",
        blend_policy: ChunkBlendPolicy = "linear_overlap",
        latency_trim_steps: int = 0,
        max_latency_trim_steps: int = 8,
        blend_min_steps: int = 8,
        ensemble_decay: float = 0.01,
        control_rate_hz: float = 50.0,
    ) -> None:
        if blend_policy not in _BLENDERS:
            raise ValueError(
                f"unknown blend_policy {blend_policy!r}; "
                f"expected one of {sorted(_BLENDERS)}"
            )
        self._robot_id = robot_id
        self._blend_policy: ChunkBlendPolicy = blend_policy
        self._latency_trim_steps = max(0, int(latency_trim_steps))
        self._max_latency_trim_steps = max(0, int(max_latency_trim_steps))
        self._control_rate_hz = max(1.0, float(control_rate_hz))

        if blend_policy == "linear_overlap":
            self._blender: _Blender = LinearOverlapBlender(
                blend_min_steps=blend_min_steps
            )
        elif blend_policy == "ensemble":
            self._blender = EnsembleBlender(decay=ensemble_decay)
        else:
            self._blender = _BLENDERS[blend_policy]()

        self._lock = threading.Lock()
        self._stale_frames: int = 0
        self._total_frames: int = 0
        self._head_type: Optional[ActionHeadType] = None
        self._last_push_at: float = 0.0

        # Buffer-underrun tracking: gaps are recorded as intervals, not counters,
        # because the tail is what breaks a control loop, not the mean.
        self._gap_started_at: Optional[float] = None
        self._gap_polls: int = 0

        # Asyncio event: signals that a new chunk is ready
        self._chunk_ready: asyncio.Event = asyncio.Event()

    # ── Writer API (action head / scheduler) ──────────────────────────────────

    def push(self, chunk: ActionChunk) -> None:
        """Integrate a freshly computed chunk under the configured blend policy.

        Called by the action head runner after predict() returns. The blend runs
        outside the blender's lock, so the control loop is not stalled by it.

        Args:
            chunk: The action chunk just produced for this robot.
        """
        actions = chunk.actions.detach().clone()
        info = self._blender.integrate(actions, max_trim=self._latency_trim_steps)

        with self._lock:
            self._head_type = chunk.head_type
            self._last_push_at = time.time()
            self._stale_frames = 0

        _tracer.event(
            "chunk_pushed",
            robot_id=self._robot_id,
            chunk_size=chunk.chunk_size,
            head_type=chunk.head_type,
            blend_policy=self._blend_policy,
            trimmed_steps=info["trimmed"],
            blended_steps=info["blended"],
            dropped=info["dropped"],
            contended=info["contended"],
            remaining=self._blender.remaining(),
        )
        _metrics.increment("chunk_pushed")
        if info["contended"]:
            _metrics.increment("chunk_blend_contended")
        if info["trimmed"]:
            _metrics.histogram("chunk_trimmed_steps", float(info["trimmed"]))
        if info["dropped"]:
            _metrics.increment("chunk_dropped_fully_stale")

        self._chunk_ready.set()

    def seed(self, action: torch.Tensor) -> None:
        """Anchor the buffer to a known pose, dropping everything queued.

        Call this on episode start or after a reset with the robot's measured
        pose, so the first blended chunk eases out of where the robot actually is
        instead of snapping to the prediction's first step.

        Args:
            action: The robot's current pose, [action_dim].
        """
        self._blender.seed(action)
        _tracer.event("chunk_buffer_seeded", robot_id=self._robot_id)

    def reset(self) -> None:
        """Drop every queued action and all stale / gap state."""
        self._blender.reset()
        with self._lock:
            self._stale_frames = 0
            self._gap_started_at = None
            self._gap_polls = 0
        self._chunk_ready.clear()
        _tracer.event("chunk_buffer_reset", robot_id=self._robot_id)

    def update_latency_trim(self, latency_ms: float) -> int:
        """Derive the latency trim from a measured end-to-end latency.

        ``ceil(latency_ms / 1000 * control_rate_hz)`` is how many control steps
        elapse while the chunk is in flight — exactly the number of leading steps
        that are already stale on arrival. Clamped to ``max_latency_trim_steps``
        so one spike cannot swallow a whole chunk.

        Args:
            latency_ms: Measured latency from observation to chunk availability.

        Returns:
            The trim now in effect, in control steps.
        """
        steps = int(math.ceil(max(0.0, latency_ms) / 1000.0 * self._control_rate_hz))
        steps = min(steps, self._max_latency_trim_steps)
        with self._lock:
            changed = steps != self._latency_trim_steps
            self._latency_trim_steps = steps
        if changed:
            _tracer.event(
                "latency_trim_updated",
                robot_id=self._robot_id,
                latency_ms=round(latency_ms, 2),
                trim_steps=steps,
            )
        return steps

    def mark_stale(self) -> None:
        """Count one frame that fell back to a reused condition.

        Observability only: whether a reused condition is still usable is
        decided by the scheduler from its age against the policy's calibrated
        tolerance (ADR-018), not from how many frames reused it.
        """
        with self._lock:
            self._stale_frames += 1

    def reset_stale(self) -> None:
        """Clear the stale-frame counter after a fresh embedding arrives."""
        with self._lock:
            self._stale_frames = 0

    # ── Reader API (control loop) ─────────────────────────────────────────────

    def current(self) -> Optional[ActionChunk]:
        """Actions still to execute, as an ActionChunk. None once drained.

        Synthesized from the blender's live state, so ``current().actions[0]`` is
        always the action next_action() would emit next — after any trim and
        blending, not the raw prediction that was pushed.
        """
        remaining = self._blender.peek()
        if remaining is None or remaining.shape[0] == 0:
            return None
        with self._lock:
            head_type = self._head_type
        if head_type is None:
            return None
        return ActionChunk(
            actions=remaining,
            head_type=head_type,
            metadata={
                "synthesized": True,
                "blend_policy": self._blend_policy,
                "robot_id": self._robot_id,
            },
        )

    def next_action(self) -> Optional[torch.Tensor]:
        """Emit the next action [action_dim]; None when the buffer has drained.

        A None opens a buffer-underrun gap; the next successful pop closes it and
        emits an ``action_buffer_gap`` event carrying the gap's duration and poll
        count. Underruns are what a control loop actually feels, so they are
        measured as intervals rather than counted.
        """
        action = self._blender.pop()
        now = time.monotonic()

        if action is None:
            with self._lock:
                self._gap_polls += 1
                started = self._gap_started_at is None
                if started:
                    self._gap_started_at = now
            if started:
                _tracer.event(
                    "action_buffer_gap",
                    robot_id=self._robot_id,
                    state="start",
                )
                _metrics.increment("action_buffer_gap")
            return None

        with self._lock:
            self._total_frames += 1
            gap_started_at = self._gap_started_at
            gap_polls = self._gap_polls
            self._gap_started_at = None
            self._gap_polls = 0

        if gap_started_at is not None:
            duration_ms = (now - gap_started_at) * 1000.0
            _tracer.event(
                "action_buffer_gap",
                robot_id=self._robot_id,
                state="end",
                duration_ms=round(duration_ms, 2),
                empty_polls=gap_polls,
            )
            _metrics.histogram("action_buffer_gap_ms", duration_ms)

        return action

    def is_exhausted(self) -> bool:
        """True when no action remains to execute."""
        return self._blender.remaining() == 0

    async def wait_for_chunk(self, timeout: float = 1.0) -> bool:
        """Async-wait until a new chunk is pushed.

        Args:
            timeout: Maximum wait in seconds.

        Returns:
            True if a chunk arrived, False if the wait timed out.
        """
        self._chunk_ready.clear()
        try:
            await asyncio.wait_for(self._chunk_ready.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def blend_policy(self) -> ChunkBlendPolicy:
        return self._blend_policy

    @property
    def latency_trim_steps(self) -> int:
        with self._lock:
            return self._latency_trim_steps

    @property
    def consumed_since_push(self) -> int:
        """Actions emitted since the last push — the ``k`` the trim consumes."""
        return self._blender.consumed_since_integrate

    @property
    def last_action(self) -> Optional[torch.Tensor]:
        """Most recently emitted action, or the seeded pose. None before either."""
        return self._blender.last_action

    @property
    def stale_frames(self) -> int:
        with self._lock:
            return self._stale_frames

    @property
    def total_frames(self) -> int:
        with self._lock:
            return self._total_frames

    def stats(self) -> dict:
        """Point-in-time snapshot of buffer occupancy, trim, and degradation state."""
        remaining = self._blender.remaining()
        consumed = self._blender.consumed_since_integrate
        with self._lock:
            return {
                "robot_id": self._robot_id,
                "blend_policy": self._blend_policy,
                "has_chunk": remaining > 0,
                "remaining": remaining,
                "latency_trim_steps": self._latency_trim_steps,
                "consumed_since_push": consumed,
                "stale_frames": self._stale_frames,
                "total_frames": self._total_frames,
                "in_buffer_gap": self._gap_started_at is not None,
            }
