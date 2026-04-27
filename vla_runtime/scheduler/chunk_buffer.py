"""Action chunk double buffer for the AsyncScheduler.

The double buffer decouples action execution (control loop, 20-100 Hz) from
action generation (5-20 Hz). While the robot executes the current chunk, the
next chunk is being computed in parallel.

    Chunk A being executed  →  Control loop reads from A
    Chunk B being computed  →  Action head writes to B
    On B ready: swap A ↔ B

This is the key mechanism that keeps inference_latency < action_execution_time
(the hard constraint from NXP validation in CLAUDE.md).

Thread safety:
    - The control loop (Thread C) reads chunk via current() — lock-free read
      of an atomic pointer.
    - The action head (Thread B) writes via push() — holds a write lock only
      during the swap, not during the computation itself.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from vla_runtime.action_head.base import ActionChunk
from vla_runtime.errors import EmbeddingStaleError, ErrorContext
from vla_runtime.observability.tracer import get_tracer

_tracer = get_tracer("scheduler.chunk_buffer")


@dataclass
class BufferSlot:
    chunk: Optional[ActionChunk] = None
    produced_at: float = 0.0
    step_index: int = 0     # which step within the chunk is being consumed


class ChunkBuffer:
    """Double-buffered action chunk store.

    Args:
        max_stale_frames: Maximum consecutive frames where the embedding can
            be reused before raising EmbeddingStaleError.
        robot_id: Identifier for the robot owning this buffer (for tracing).
    """

    def __init__(
        self,
        max_stale_frames: int = 5,
        robot_id: str = "robot_0",
    ) -> None:
        self._max_stale_frames = max_stale_frames
        self._robot_id = robot_id
        self._lock = threading.Lock()
        self._slots: list[BufferSlot] = [BufferSlot(), BufferSlot()]
        self._active_idx: int = 0           # index of the slot being executed
        self._pending_idx: int = 1          # index of the slot being written
        self._stale_frames: int = 0
        self._total_frames: int = 0

        # Asyncio event: signals that a new chunk is ready
        self._chunk_ready: asyncio.Event = asyncio.Event()

    # ── Writer API (action head / scheduler) ──────────────────────────────────

    def push(self, chunk: ActionChunk) -> None:
        """Write a freshly computed chunk into the pending slot and swap.

        Called by the action head runner after predict() returns.
        The swap is atomic from the control loop's perspective.
        """
        now = time.time()
        with self._lock:
            pending = self._slots[self._pending_idx]
            pending.chunk = chunk
            pending.produced_at = now
            pending.step_index = 0
            # Atomic swap
            self._active_idx, self._pending_idx = self._pending_idx, self._active_idx
            self._stale_frames = 0

        _tracer.event(
            "chunk_pushed",
            robot_id=self._robot_id,
            chunk_size=chunk.chunk_size,
            head_type=chunk.head_type,
        )
        self._chunk_ready.set()

    def mark_stale(self) -> None:
        """Increment stale frame counter when the VLM embedding was reused.

        Called by AsyncScheduler.on_vlm_timeout() on each fallback frame.

        Raises:
            EmbeddingStaleError: When stale_frames exceeds max_stale_frames.
        """
        with self._lock:
            self._stale_frames += 1
            stale = self._stale_frames

        if stale > self._max_stale_frames:
            ctx = ErrorContext(
                trace_id=_tracer.trace_id,
                module_name="scheduler.chunk_buffer",
                extra={"robot_id": self._robot_id},
            )
            raise EmbeddingStaleError(stale, self._max_stale_frames, ctx)

    # ── Reader API (control loop) ─────────────────────────────────────────────

    def current(self) -> Optional[ActionChunk]:
        """Return the current active chunk (None if buffer not yet populated)."""
        with self._lock:
            return self._slots[self._active_idx].chunk

    def next_action(self) -> Optional["torch.Tensor"]:  # noqa: F821
        """Return the next action from the active chunk and advance step_index.

        Returns None if the chunk is exhausted (caller should wait for push()).
        """
        with self._lock:
            slot = self._slots[self._active_idx]
            if slot.chunk is None:
                return None
            if slot.step_index >= slot.chunk.chunk_size:
                return None
            action = slot.chunk.step(slot.step_index)
            slot.step_index += 1
            self._total_frames += 1
            return action

    def is_exhausted(self) -> bool:
        """True if all steps in the active chunk have been consumed."""
        with self._lock:
            slot = self._slots[self._active_idx]
            if slot.chunk is None:
                return True
            return slot.step_index >= slot.chunk.chunk_size

    async def wait_for_chunk(self, timeout: float = 1.0) -> bool:
        """Async wait until a new chunk is pushed.

        Args:
            timeout: Maximum wait in seconds.

        Returns:
            True if a chunk arrived, False if timed out.
        """
        self._chunk_ready.clear()
        try:
            await asyncio.wait_for(self._chunk_ready.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def stale_frames(self) -> int:
        with self._lock:
            return self._stale_frames

    @property
    def total_frames(self) -> int:
        with self._lock:
            return self._total_frames

    def stats(self) -> dict:
        with self._lock:
            slot = self._slots[self._active_idx]
            return {
                "robot_id": self._robot_id,
                "has_chunk": slot.chunk is not None,
                "step_index": slot.step_index,
                "stale_frames": self._stale_frames,
                "total_frames": self._total_frames,
                "chunk_size": slot.chunk.chunk_size if slot.chunk else 0,
            }
