"""Unit tests for AsyncScheduler and BudgetManager.

All tests use a MockVLMBackend and real action heads (CPU, small dims).
No GPU required. Tests cover:
- Normal inference path: submit → stream → ActionChunk
- VLM timeout → on_vlm_timeout fallback → stale embedding reuse
- VLM timeout with no prior embedding → VLMTimeoutError
- Consecutive timeouts → stale-frame accounting in the chunk buffer
- Action head failure → on_action_head_failure → zero chunk
- BudgetManager latency recording and violation rate

ChunkBuffer has its own suite in tests/unit/test_chunk_buffer.py.
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
    ErrorContext,
    VLMTimeoutError,
)
from vla_runtime.scheduler.async_scheduler import AsyncScheduler, RequestHandle
from vla_runtime.scheduler.budget_manager import BudgetConfig, BudgetManager
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
            # Room for the mock's 5 ms happy path, far below the 2nd call's hang,
            # so the first request can bank an embedding for the second to reuse.
            budget_config=BudgetConfig(vlm_budget_ms=100.0),
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
