"""Unit tests for ChunkBuffer and its four blend policies (ADR-014).

CPU only, no GPU required. Coverage:
- replace / trim / linear_overlap / ensemble blend semantics
- latency trim derived from measured latency, and its clamp
- fully-stale chunk drop (trim >= chunk length)
- seeding from the robot's measured pose, and easing out of a drained buffer
- buffer-underrun gaps recorded as intervals
- stale-embedding accounting and EmbeddingStaleError
- concurrent push / next_action, and the locked-blend fallback under starvation
"""

from __future__ import annotations

import threading
import time

import pytest
import torch

from vla_runtime.action_head.base import ActionChunk
from vla_runtime.errors import EmbeddingStaleError
from vla_runtime.scheduler.chunk_buffer import (
    _MAX_OPTIMISTIC_RETRIES,
    ChunkBuffer,
    LinearOverlapBlender,
)

CHUNK_SIZE = 8
ACTION_DIM = 4


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_chunk(
    value: float = 1.0,
    size: int = CHUNK_SIZE,
    dim: int = ACTION_DIM,
    dtype: torch.dtype = torch.float32,
) -> ActionChunk:
    """A constant-valued chunk — makes blend weights readable in assertions."""
    return ActionChunk(
        actions=torch.full((size, dim), value, dtype=dtype),
        head_type="flow_matching",
    )


def make_ramp_chunk(size: int = CHUNK_SIZE, dim: int = ACTION_DIM) -> ActionChunk:
    """Chunk whose step i holds the constant i — makes trims readable."""
    steps = torch.arange(size, dtype=torch.float32).reshape(size, 1)
    return ActionChunk(actions=steps.expand(size, dim).clone(), head_type="flow_matching")


def drain(buf: ChunkBuffer) -> list[torch.Tensor]:
    """Pop every buffered action."""
    out = []
    while True:
        action = buf.next_action()
        if action is None:
            return out
        out.append(action)


# ── Construction ──────────────────────────────────────────────────────────────

class TestConstruction:
    def test_rejects_unknown_policy(self):
        with pytest.raises(ValueError, match="unknown blend_policy"):
            ChunkBuffer(blend_policy="bilinear")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "policy", ["replace", "trim", "linear_overlap", "ensemble"]
    )
    def test_all_policies_construct_and_report(self, policy):
        buf = ChunkBuffer(blend_policy=policy)
        assert buf.blend_policy == policy
        assert buf.is_exhausted()
        assert buf.current() is None

    def test_default_policy_is_linear_overlap(self):
        assert ChunkBuffer().blend_policy == "linear_overlap"


# ── replace: legacy double-buffer semantics ───────────────────────────────────

class TestReplacePolicy:
    def test_new_chunk_replaces_wholesale(self):
        buf = ChunkBuffer(blend_policy="replace")
        buf.push(make_chunk(1.0))
        buf.push(make_chunk(2.0))
        assert buf.current().actions[0, 0].item() == 2.0
        assert buf.stats()["remaining"] == CHUNK_SIZE

    def test_ignores_latency_trim(self):
        buf = ChunkBuffer(blend_policy="replace", latency_trim_steps=4)
        buf.push(make_chunk(1.0))
        for _ in range(4):
            buf.next_action()
        buf.push(make_ramp_chunk())
        # No trim: execution restarts at the new chunk's step 0.
        assert buf.next_action()[0].item() == 0.0

    def test_consumes_all_steps_then_reports_exhausted(self):
        buf = ChunkBuffer(blend_policy="replace")
        buf.push(make_chunk())
        assert len(drain(buf)) == CHUNK_SIZE
        assert buf.is_exhausted()
        assert buf.next_action() is None


# ── trim: latency-aligned replacement ─────────────────────────────────────────

class TestTrimPolicy:
    def test_trims_consumed_steps_from_new_chunk(self):
        buf = ChunkBuffer(blend_policy="trim", latency_trim_steps=3)
        buf.push(make_chunk(1.0))
        for _ in range(3):
            buf.next_action()
        buf.push(make_ramp_chunk())
        # 3 steps elapsed in flight, so the new chunk resumes at its step 3.
        assert buf.next_action()[0].item() == 3.0

    def test_trim_is_capped_by_steps_actually_consumed(self):
        buf = ChunkBuffer(blend_policy="trim", latency_trim_steps=5)
        buf.push(make_chunk(1.0))
        buf.next_action()   # only one step consumed
        buf.push(make_ramp_chunk())
        assert buf.next_action()[0].item() == 1.0

    def test_no_trim_when_nothing_consumed(self):
        buf = ChunkBuffer(blend_policy="trim", latency_trim_steps=5)
        buf.push(make_chunk(1.0))
        buf.push(make_ramp_chunk())
        assert buf.next_action()[0].item() == 0.0

    def test_fully_stale_chunk_is_dropped_and_old_survives(self):
        buf = ChunkBuffer(blend_policy="trim", latency_trim_steps=99,
                          max_latency_trim_steps=99)
        buf.push(make_chunk(1.0, size=4))
        for _ in range(4):
            buf.next_action()        # k = 4
        buf.push(make_chunk(9.0, size=4))
        # trim (4) >= chunk length (4): the whole chunk is stale, nothing queued.
        assert buf.is_exhausted()
        assert buf.next_action() is None

    def test_consumed_since_push_tracks_k(self):
        buf = ChunkBuffer(blend_policy="trim")
        buf.push(make_chunk())
        assert buf.consumed_since_push == 0
        buf.next_action()
        buf.next_action()
        assert buf.consumed_since_push == 2
        buf.push(make_chunk())
        assert buf.consumed_since_push == 0


# ── linear_overlap: the default ───────────────────────────────────────────────

class TestLinearOverlapPolicy:
    def test_first_chunk_passes_through_unblended(self):
        buf = ChunkBuffer(blend_policy="linear_overlap")
        buf.push(make_chunk(2.0))
        assert buf.next_action()[0].item() == pytest.approx(2.0)

    def test_cross_fade_runs_from_old_to_new(self):
        buf = ChunkBuffer(blend_policy="linear_overlap")
        buf.push(make_chunk(1.0))
        buf.push(make_chunk(2.0))
        actions = torch.stack(drain(buf))[:, 0]
        # Weight 1 on the old trajectory at the seam, weight 1 on the new at the end.
        assert actions[0].item() == pytest.approx(1.0)
        assert actions[-1].item() == pytest.approx(2.0)
        # And strictly monotone in between — that is what removes the jerk.
        assert torch.all(actions[1:] > actions[:-1])

    def test_no_discontinuity_at_the_seam(self):
        """The blended action must continue from the last executed one."""
        buf = ChunkBuffer(blend_policy="linear_overlap")
        buf.push(make_chunk(1.0))
        for _ in range(4):
            buf.next_action()
        last_executed = buf.last_action[0].item()
        buf.push(make_chunk(5.0))
        first_after = buf.next_action()[0].item()
        # replace/trim would jump straight to 5.0 here.
        assert first_after == pytest.approx(last_executed)

    def test_drained_buffer_eases_out_of_last_executed_action(self):
        buf = ChunkBuffer(blend_policy="linear_overlap")
        buf.push(make_chunk(1.0))
        drain(buf)                      # buffer empty, last executed = 1.0
        buf.push(make_chunk(2.0))
        actions = torch.stack(drain(buf))[:, 0]
        assert actions[0].item() == pytest.approx(1.0)
        assert actions[-1].item() == pytest.approx(2.0)

    def test_seed_anchors_the_first_blend_to_the_measured_pose(self):
        buf = ChunkBuffer(blend_policy="linear_overlap")
        buf.seed(torch.zeros(ACTION_DIM))
        buf.push(make_chunk(2.0))
        # Eases out of the robot's actual pose instead of snapping to 2.0.
        assert buf.next_action()[0].item() == pytest.approx(0.0)

    def test_short_tail_is_padded_to_blend_min_steps(self):
        buf = ChunkBuffer(blend_policy="linear_overlap", blend_min_steps=8)
        buf.push(make_chunk(1.0))
        for _ in range(CHUNK_SIZE - 1):
            buf.next_action()           # one step left in the tail
        buf.push(make_chunk(2.0))
        actions = torch.stack(drain(buf))[:, 0]
        # Without padding the fade would collapse to a single step.
        assert actions[0].item() == pytest.approx(1.0)
        assert (actions[1] - actions[0]).item() < 1.0

    def test_trim_applies_before_blending(self):
        buf = ChunkBuffer(blend_policy="linear_overlap", latency_trim_steps=2)
        buf.push(make_chunk(0.0, size=4))
        for _ in range(4):
            buf.next_action()           # k = 4, drained; last executed = 0.0
        buf.push(make_ramp_chunk(size=8))
        actions = torch.stack(drain(buf))[:, 0]
        # Trimmed to steps 2..7, then faded in from the last executed 0.0,
        # so the tail must end on the ramp's last value.
        assert actions[-1].item() == pytest.approx(7.0)
        assert len(actions) == 6

    def test_preserves_dtype(self):
        buf = ChunkBuffer(blend_policy="linear_overlap")
        buf.push(make_chunk(1.0, dtype=torch.float16))
        buf.push(make_chunk(2.0, dtype=torch.float16))
        assert buf.next_action().dtype == torch.float16


# ── ensemble: ACT temporal ensembling ─────────────────────────────────────────

class TestEnsemblePolicy:
    def test_single_chunk_is_returned_unchanged(self):
        buf = ChunkBuffer(blend_policy="ensemble")
        buf.push(make_chunk(3.0))
        assert buf.next_action()[0].item() == pytest.approx(3.0)

    def test_overlapping_chunks_are_averaged(self):
        buf = ChunkBuffer(blend_policy="ensemble", ensemble_decay=0.0)
        buf.push(make_chunk(1.0))
        buf.push(make_chunk(3.0))       # same anchor step, decay 0 -> plain mean
        assert buf.next_action()[0].item() == pytest.approx(2.0)

    def test_decay_weights_the_older_prediction_more(self):
        buf = ChunkBuffer(blend_policy="ensemble", ensemble_decay=1.0)
        buf.push(make_chunk(0.0))
        buf.push(make_chunk(1.0))
        # Weights exp(0), exp(-1) normalised -> the newer chunk contributes < 0.5.
        assert buf.next_action()[0].item() < 0.5

    def test_chunks_anchor_at_the_current_execution_step(self):
        buf = ChunkBuffer(blend_policy="ensemble", ensemble_decay=0.0)
        buf.push(make_chunk(1.0, size=4))
        buf.next_action()
        buf.next_action()               # execution step is now 2
        buf.push(make_chunk(3.0, size=4))
        # Steps 2-3 are covered by both chunks, steps 4-5 only by the new one.
        assert buf.next_action()[0].item() == pytest.approx(2.0)
        assert buf.next_action()[0].item() == pytest.approx(2.0)
        assert buf.next_action()[0].item() == pytest.approx(3.0)

    def test_exhausted_chunks_are_pruned(self):
        buf = ChunkBuffer(blend_policy="ensemble")
        buf.push(make_chunk(1.0, size=4))
        assert len(drain(buf)) == 4
        assert buf.is_exhausted()
        assert buf.next_action() is None

    def test_remaining_spans_the_furthest_horizon(self):
        buf = ChunkBuffer(blend_policy="ensemble")
        buf.push(make_chunk(1.0, size=4))
        buf.next_action()
        buf.push(make_chunk(1.0, size=8))
        # anchored at step 1, horizon reaches step 9
        assert buf.stats()["remaining"] == 8


# ── Latency trim derivation ───────────────────────────────────────────────────

class TestLatencyTrim:
    def test_derived_from_measured_latency_and_control_rate(self):
        buf = ChunkBuffer(control_rate_hz=50.0, max_latency_trim_steps=99)
        # 100 ms in flight at 50 Hz = 5 control steps already stale.
        assert buf.update_latency_trim(100.0) == 5
        assert buf.latency_trim_steps == 5

    def test_rounds_up_partial_steps(self):
        buf = ChunkBuffer(control_rate_hz=50.0, max_latency_trim_steps=99)
        assert buf.update_latency_trim(101.0) == 6

    def test_clamped_by_max_latency_trim_steps(self):
        buf = ChunkBuffer(control_rate_hz=50.0, max_latency_trim_steps=4)
        assert buf.update_latency_trim(1000.0) == 4

    def test_zero_latency_means_no_trim(self):
        buf = ChunkBuffer(control_rate_hz=50.0)
        assert buf.update_latency_trim(0.0) == 0

    def test_negative_latency_is_floored_at_zero(self):
        buf = ChunkBuffer(control_rate_hz=50.0)
        assert buf.update_latency_trim(-10.0) == 0

    def test_derived_trim_takes_effect_on_the_next_push(self):
        buf = ChunkBuffer(blend_policy="trim", control_rate_hz=50.0,
                          max_latency_trim_steps=99)
        buf.push(make_chunk(1.0))
        for _ in range(4):
            buf.next_action()
        buf.update_latency_trim(60.0)   # 3 steps
        buf.push(make_ramp_chunk())
        assert buf.next_action()[0].item() == 3.0


# ── Buffer underrun gaps ──────────────────────────────────────────────────────

class TestBufferGap:
    def test_gap_opens_when_the_buffer_underruns(self):
        buf = ChunkBuffer()
        assert buf.next_action() is None
        assert buf.stats()["in_buffer_gap"]

    def test_gap_closes_on_the_next_successful_pop(self):
        buf = ChunkBuffer()
        buf.next_action()
        buf.next_action()
        assert buf.stats()["in_buffer_gap"]
        buf.push(make_chunk())
        assert buf.next_action() is not None
        assert not buf.stats()["in_buffer_gap"]

    def test_total_frames_counts_only_emitted_actions(self):
        buf = ChunkBuffer()
        buf.next_action()               # underrun, not an emitted action
        buf.push(make_chunk(size=3))
        drain(buf)
        assert buf.total_frames == 3


# ── Stale embedding accounting ────────────────────────────────────────────────

class TestStaleFrames:
    def test_mark_stale_increments(self):
        buf = ChunkBuffer(max_stale_frames=10)
        buf.mark_stale()
        assert buf.stale_frames == 1

    def test_raises_once_the_tolerance_is_exceeded(self):
        buf = ChunkBuffer(max_stale_frames=2)
        buf.mark_stale()
        buf.mark_stale()
        with pytest.raises(EmbeddingStaleError):
            buf.mark_stale()

    def test_push_resets_the_counter(self):
        buf = ChunkBuffer(max_stale_frames=10)
        buf.mark_stale()
        buf.push(make_chunk())
        assert buf.stale_frames == 0

    def test_reset_stale_clears_the_counter(self):
        buf = ChunkBuffer(max_stale_frames=10)
        buf.mark_stale()
        buf.reset_stale()
        assert buf.stale_frames == 0


# ── Reader-side views ─────────────────────────────────────────────────────────

class TestCurrentAndReset:
    def test_current_reflects_what_is_left_to_execute(self):
        buf = ChunkBuffer(blend_policy="replace")
        buf.push(make_ramp_chunk())
        buf.next_action()
        buf.next_action()
        current = buf.current()
        assert current.chunk_size == CHUNK_SIZE - 2
        assert current.actions[0, 0].item() == 2.0

    def test_current_preserves_head_type(self):
        buf = ChunkBuffer()
        buf.push(make_chunk())
        assert buf.current().head_type == "flow_matching"

    def test_current_is_none_once_drained(self):
        buf = ChunkBuffer()
        buf.push(make_chunk())
        drain(buf)
        assert buf.current() is None

    def test_reset_clears_everything(self):
        buf = ChunkBuffer()
        buf.push(make_chunk())
        buf.next_action()
        buf.mark_stale()
        buf.reset()
        assert buf.is_exhausted()
        assert buf.current() is None
        assert buf.stale_frames == 0
        assert buf.last_action is None

    def test_stats_structure(self):
        buf = ChunkBuffer(robot_id="r1", blend_policy="trim")
        stats = buf.stats()
        assert stats["robot_id"] == "r1"
        assert stats["blend_policy"] == "trim"
        assert not stats["has_chunk"]
        assert stats["remaining"] == 0

    @pytest.mark.asyncio
    async def test_wait_for_chunk_resolves_on_push(self):
        import asyncio

        buf = ChunkBuffer()

        async def push_after_delay():
            await asyncio.sleep(0.05)
            buf.push(make_chunk())

        task = asyncio.create_task(push_after_delay())
        assert await buf.wait_for_chunk(timeout=1.0)
        await task

    @pytest.mark.asyncio
    async def test_wait_for_chunk_times_out(self):
        buf = ChunkBuffer()
        assert not await buf.wait_for_chunk(timeout=0.05)


# ── Concurrency ───────────────────────────────────────────────────────────────

class TestConcurrency:
    def test_falls_back_to_locked_blend_when_the_reader_never_yields(self):
        """A reader that invalidates every snapshot must not starve the writer."""

        class RacingBlender(LinearOverlapBlender):
            """Invalidates the snapshot on every optimistic attempt.

            Bumps the revision the way a concurrent pop would, but only while the
            lock is free — on the locked fallback path the acquire fails, which is
            exactly the condition being tested.
            """

            def __init__(self) -> None:
                super().__init__()
                self.attempts = 0

            def _combine(self, old, last_action, new):
                self.attempts += 1
                if self._lock.acquire(blocking=False):
                    self._revision += 1     # as a racing control loop would
                    self._lock.release()
                return super()._combine(old, last_action, new)

        blender = RacingBlender()
        blender.integrate(torch.full((8, ACTION_DIM), 1.0), max_trim=0)
        blender.attempts = 0
        info = blender.integrate(torch.full((8, ACTION_DIM), 2.0), max_trim=0)

        assert info["contended"] is True
        assert blender.remaining() > 0, "the chunk must still land"
        # Bounded: the optimistic attempts plus exactly one under the lock.
        assert blender.attempts == _MAX_OPTIMISTIC_RETRIES + 1

    @pytest.mark.parametrize(
        "policy", ["replace", "trim", "linear_overlap", "ensemble"]
    )
    def test_uncontended_push_is_not_flagged(self, policy):
        buf = ChunkBuffer(blend_policy=policy)
        buf.push(make_chunk())
        assert not buf.stats()["in_buffer_gap"]
        assert buf.stats()["remaining"] > 0

    @pytest.mark.parametrize(
        "policy", ["replace", "trim", "linear_overlap", "ensemble"]
    )
    def test_control_loop_pops_while_action_head_pushes(self, policy):
        """The optimistic-commit retry must never deadlock, drop, or corrupt state.

        The reader ticks at ~2 kHz — far faster than any real control loop, so
        every push contends — and the whole run is held to a wall-clock bound so
        a starved writer fails the test instead of merely running slowly.
        """
        buf = ChunkBuffer(blend_policy=policy, latency_trim_steps=2)
        buf.push(make_chunk(1.0))
        stop = threading.Event()
        popped: list[torch.Tensor] = []
        errors: list[BaseException] = []

        def control_loop():
            try:
                while not stop.is_set():
                    action = buf.next_action()
                    if action is not None:
                        popped.append(action)
                    time.sleep(0.0005)
            except BaseException as exc:  # noqa: BLE001 - surfaced to the test
                errors.append(exc)

        thread = threading.Thread(target=control_loop, name="test-control", daemon=True)
        thread.start()
        started = time.monotonic()
        for i in range(100):
            buf.push(make_chunk(float(i)))
        elapsed = time.monotonic() - started
        stop.set()
        thread.join(timeout=5.0)

        assert not thread.is_alive()
        assert not errors
        assert popped, "control loop starved for the whole run"
        assert all(a.shape == (ACTION_DIM,) for a in popped)
        assert buf.total_frames == len(popped)
        assert elapsed < 5.0, f"writer starved by the control loop: {elapsed:.1f}s"
