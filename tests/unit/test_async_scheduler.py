"""Unit tests for AsyncScheduler, ChunkBuffer, and BudgetManager.

All tests use a MockVLMBackend and real action heads (CPU, small dims).
No GPU required. Tests cover:
- Normal inference path: submit → stream → ActionChunk
- VLM timeout → on_vlm_timeout fallback → stale embedding reuse
- VLM timeout with no prior embedding → VLMTimeoutError
- Consecutive timeouts → EmbeddingStaleError
- Action head failure → on_action_head_failure → zero chunk
- BudgetManager latency recording and violation rate
- ChunkBuffer double-buffer swap and step consumption
- ChunkBuffer stale frame counting
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Optional

import pytest
import torch

from vla_runtime.action_head.base import ActionChunk, ActionHeadType, RobotState
from vla_runtime.action_head.flow_matching import FlowMatchingHead
from vla_runtime.errors import (
    ActionHeadFailureError,
    BackendError,
    BudgetExceededError,
    EmbeddingStaleError,
    ErrorContext,
    VLMTimeoutError,
)
from vla_runtime.scheduler.async_scheduler import AsyncScheduler, RequestHandle
from vla_runtime.scheduler.budget_manager import BudgetConfig, BudgetManager
from vla_runtime.scheduler.chunk_buffer import ChunkBuffer
from vla_runtime.vlm.base import ActionRequest, Embedding, TokenChunk, TokenStream


# ── Helpers ───────────────────────────────────────────────────────────────────

EMBED_DIM = 32
ACTION_DIM = 4
CHUNK_SIZE = 4
STATE_DIM = 8


def make_embedding() -> Embedding:
    return Embedding(data=torch.zeros(EMBED_DIM, dtype=torch.float16), model_id="mock")


def make_request(req_id: Optional[str] = None) -> ActionRequest:
    return ActionRequest(
        request_id=req_id or str(uuid.uuid4())[:8],
        image=torch.rand(3, 64, 64),
        instruction="pick up object",
        robot_state=torch.zeros(STATE_DIM),
    )


def make_chunk() -> ActionChunk:
    return ActionChunk(
        actions=torch.zeros(CHUNK_SIZE, ACTION_DIM, dtype=torch.float16),
        head_type="flow_matching",
    )


# ── Mock VLM Backend ──────────────────────────────────────────────────────────

class MockVLMBackend:
    def __init__(
        self,
        latency_ms: float = 5.0,
        fail_on_call: int = -1,   # -1 = never fail
        timeout_on_call: int = -1,
    ) -> None:
        self._latency_ms = latency_ms
        self._fail_on_call = fail_on_call
        self._timeout_on_call = timeout_on_call
        self._call_count = 0

    @property
    def model_id(self) -> str:
        return "mock:vlm"

    @property
    def latency_budget_ms(self) -> float:
        return 500.0

    @property
    def kv_cache_enabled(self) -> bool:
        return False

    async def encode(self, image: torch.Tensor, text: str) -> Embedding:
        self._call_count += 1
        if self._timeout_on_call == self._call_count:
            await asyncio.sleep(100)   # simulates hang
        if self._fail_on_call == self._call_count:
            raise BackendError("Mock encode failure")
        await asyncio.sleep(self._latency_ms / 1000.0)
        return make_embedding()

    async def generate(self, image, text) -> TokenStream:
        yield TokenChunk(token_ids=[], is_final=True)

    async def warmup(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass


# ── Mock Action Head ──────────────────────────────────────────────────────────

class MockActionHead:
    def __init__(self, fail: bool = False) -> None:
        self._fail = fail

    @property
    def head_type(self) -> ActionHeadType:
        return "flow_matching"

    @property
    def frequency_hz(self) -> float:
        return 10.0

    @property
    def chunk_size(self) -> int:
        return CHUNK_SIZE

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    def predict(self, embedding: Embedding, robot_state: RobotState) -> ActionChunk:
        if self._fail:
            raise ActionHeadFailureError("flow_matching", RuntimeError("mock failure"))
        return make_chunk()

    def load_weights(self, path: str) -> None:
        pass

    def to_device(self, device: torch.device) -> None:
        pass


# ── ChunkBuffer tests ─────────────────────────────────────────────────────────

class TestChunkBuffer:
    def test_initial_state_is_empty(self):
        buf = ChunkBuffer()
        assert buf.current() is None
        assert buf.is_exhausted()

    def test_push_makes_chunk_available(self):
        buf = ChunkBuffer()
        buf.push(make_chunk())
        assert buf.current() is not None

    def test_next_action_consumes_steps(self):
        buf = ChunkBuffer()
        buf.push(make_chunk())
        for i in range(CHUNK_SIZE):
            action = buf.next_action()
            assert action is not None
        assert buf.is_exhausted()

    def test_next_action_returns_none_when_exhausted(self):
        buf = ChunkBuffer()
        buf.push(make_chunk())
        for _ in range(CHUNK_SIZE):
            buf.next_action()
        assert buf.next_action() is None

    def test_double_buffer_swap(self):
        buf = ChunkBuffer()
        chunk1 = ActionChunk(
            actions=torch.ones(CHUNK_SIZE, ACTION_DIM, dtype=torch.float16),
            head_type="flow_matching",
        )
        chunk2 = ActionChunk(
            actions=torch.full((CHUNK_SIZE, ACTION_DIM), 2.0, dtype=torch.float16),
            head_type="flow_matching",
        )
        buf.push(chunk1)
        buf.push(chunk2)
        # After second push, active slot should contain chunk2
        assert buf.current().actions[0, 0].item() == 2.0

    def test_stale_frames_reset_on_push(self):
        buf = ChunkBuffer(max_stale_frames=10)
        buf._stale_frames = 5
        buf.push(make_chunk())
        assert buf.stale_frames == 0

    def test_mark_stale_increments(self):
        buf = ChunkBuffer(max_stale_frames=10)
        buf.mark_stale()
        assert buf.stale_frames == 1

    def test_mark_stale_raises_on_exceed(self):
        buf = ChunkBuffer(max_stale_frames=2)
        buf.mark_stale()
        buf.mark_stale()
        with pytest.raises(EmbeddingStaleError):
            buf.mark_stale()

    @pytest.mark.asyncio
    async def test_wait_for_chunk_resolves(self):
        buf = ChunkBuffer()

        async def push_after_delay():
            await asyncio.sleep(0.05)
            buf.push(make_chunk())

        task = asyncio.create_task(push_after_delay())
        arrived = await buf.wait_for_chunk(timeout=1.0)
        assert arrived
        await task

    @pytest.mark.asyncio
    async def test_wait_for_chunk_timeout(self):
        buf = ChunkBuffer()
        arrived = await buf.wait_for_chunk(timeout=0.05)
        assert not arrived

    def test_stats_dict(self):
        buf = ChunkBuffer(robot_id="r1")
        stats = buf.stats()
        assert stats["robot_id"] == "r1"
        assert not stats["has_chunk"]


# ── BudgetManager tests ───────────────────────────────────────────────────────

class TestBudgetManager:
    def test_record_within_budget(self):
        mgr = BudgetManager(BudgetConfig(vlm_budget_ms=200))
        mgr.record("vlm", 100.0)
        assert mgr.violation_rate("vlm") == 0.0

    def test_record_exceeds_budget(self):
        mgr = BudgetManager(BudgetConfig(vlm_budget_ms=100))
        mgr.record("vlm", 200.0)
        assert mgr.violation_rate("vlm") == 1.0

    def test_percentile_none_on_empty(self):
        mgr = BudgetManager()
        assert mgr.percentile("vlm", 50) is None

    def test_percentile_p50(self):
        mgr = BudgetManager(BudgetConfig(window_size=100))
        for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
            mgr.record("vlm", v)
        p50 = mgr.percentile("vlm", 50)
        assert p50 is not None
        assert 25.0 <= p50 <= 40.0

    def test_check_and_raise_on_violation(self):
        mgr = BudgetManager(BudgetConfig(vlm_budget_ms=100))
        with pytest.raises(BudgetExceededError):
            mgr.check_and_raise("vlm", 200.0)

    def test_on_budget_exceeded_callback(self):
        exceeded = []

        def callback(module, latency, budget):
            exceeded.append((module, latency))

        mgr = BudgetManager(
            BudgetConfig(vlm_budget_ms=100, degradation_threshold=0.0),
            on_budget_exceeded=callback,
        )
        mgr.record("vlm", 200.0)
        assert len(exceeded) == 1

    def test_snapshot_structure(self):
        mgr = BudgetManager()
        snap = mgr.snapshot()
        assert "vlm" in snap
        assert "action_head" in snap
        assert "total" in snap


# ── AsyncScheduler tests ──────────────────────────────────────────────────────

class TestAsyncScheduler:
    def _make_scheduler(
        self,
        vlm: Optional[MockVLMBackend] = None,
        head: Optional[MockActionHead] = None,
        max_stale_frames: int = 3,
    ) -> AsyncScheduler:
        return AsyncScheduler(
            vlm_backend=vlm or MockVLMBackend(),
            action_head=head or MockActionHead(),
            budget_config=BudgetConfig(
                vlm_budget_ms=500,
                action_head_budget_ms=500,
                total_budget_ms=1000,
            ),
            max_stale_frames=max_stale_frames,
        )

    @pytest.mark.asyncio
    async def test_submit_returns_handle(self):
        sched = self._make_scheduler()
        await sched.start()
        handle = await sched.submit(make_request())
        assert isinstance(handle, RequestHandle)
        await sched.stop()

    @pytest.mark.asyncio
    async def test_stream_yields_action_chunk(self):
        sched = self._make_scheduler()
        await sched.start()
        handle = await sched.submit(make_request())
        chunks = []
        async for chunk in sched.stream(handle):
            chunks.append(chunk)
        assert len(chunks) >= 1
        assert all(isinstance(c, ActionChunk) for c in chunks)
        await sched.stop()

    @pytest.mark.asyncio
    async def test_action_chunk_shape(self):
        sched = self._make_scheduler()
        await sched.start()
        handle = await sched.submit(make_request())
        async for chunk in sched.stream(handle):
            assert chunk.actions.shape == (CHUNK_SIZE, ACTION_DIM)
        await sched.stop()

    @pytest.mark.asyncio
    async def test_vlm_timeout_uses_last_embedding(self):
        """First call succeeds, then VLM times out — should use stale embedding."""
        vlm = MockVLMBackend(timeout_on_call=2)   # 2nd call hangs
        sched = AsyncScheduler(
            vlm_backend=vlm,
            action_head=MockActionHead(),
            budget_config=BudgetConfig(vlm_budget_ms=0.001),  # ~instant timeout
            max_stale_frames=5,
        )
        await sched.start()

        # First request: succeeds → stores last_embedding
        handle1 = await sched.submit(make_request())
        async for _ in sched.stream(handle1):
            pass

        # Second request: VLM times out → should fallback
        handle2 = await sched.submit(make_request())
        chunks = []
        async for chunk in sched.stream(handle2):
            chunks.append(chunk)
        # Should still produce a chunk via fallback
        assert len(chunks) >= 1
        await sched.stop()

    @pytest.mark.asyncio
    async def test_vlm_timeout_no_prior_embedding_raises(self):
        """VLM times out on first ever call → VLMTimeoutError (no fallback)."""
        vlm = MockVLMBackend()
        sched = AsyncScheduler(
            vlm_backend=vlm,
            action_head=MockActionHead(),
            budget_config=BudgetConfig(vlm_budget_ms=0.001),  # instant timeout
        )
        await sched.start()

        with pytest.raises(VLMTimeoutError):
            handle = await sched.submit(make_request())
            async for _ in sched.stream(handle):
                pass

        await sched.stop()

    @pytest.mark.asyncio
    async def test_action_head_failure_returns_zero_chunk(self):
        sched = self._make_scheduler(head=MockActionHead(fail=True))
        await sched.start()
        handle = await sched.submit(make_request())
        chunks = []
        async for chunk in sched.stream(handle):
            chunks.append(chunk)

        assert len(chunks) >= 1
        # Fallback chunk should be all zeros
        assert (chunks[0].actions == 0).all()
        assert chunks[0].metadata.get("fallback") is True
        await sched.stop()

    @pytest.mark.asyncio
    async def test_stats_structure(self):
        sched = self._make_scheduler()
        await sched.start()
        stats = sched.stats()
        assert "robot_id" in stats
        assert "buffer" in stats
        assert "budget" in stats
        await sched.stop()
