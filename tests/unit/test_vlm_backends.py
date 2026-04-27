"""Unit tests for VLM backend implementations.

Tests use mock VLM backends to avoid requiring actual GPU / model weights.
Kernel tests (kernels/) are intentionally excluded here — they require real GPU.

Coverage:
- VLMBackend Protocol compliance
- Embedding type correctness
- Timeout → VLMTimeoutError
- Uninitialized backend → BackendError
- QuantizationConfig integration
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest
import torch

from vla_runtime.errors import BackendError, VLMTimeoutError
from vla_runtime.quantization.config import QuantizationConfig
from vla_runtime.vlm.base import ActionRequest, Embedding, VLMBackend


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_image(h: int = 224, w: int = 224, c: int = 3) -> torch.Tensor:
    return torch.rand(c, h, w)


def make_request(instruction: str = "pick up the cup") -> ActionRequest:
    return ActionRequest(
        request_id="test-001",
        image=make_image(),
        instruction=instruction,
        robot_state=torch.zeros(14),
    )


# ── Mock backend ──────────────────────────────────────────────────────────────

class MockVLMBackend:
    """Minimal VLMBackend implementation for testing."""

    def __init__(
        self,
        hidden_dim: int = 64,
        latency_ms: float = 10.0,
        fail_encode: bool = False,
        simulate_timeout: bool = False,
    ) -> None:
        self._hidden_dim = hidden_dim
        self._latency_ms = latency_ms
        self._fail_encode = fail_encode
        self._simulate_timeout = simulate_timeout
        self._warmed_up = False

    @property
    def model_id(self) -> str:
        return "mock:test-vlm"

    @property
    def latency_budget_ms(self) -> float:
        return 500.0

    @property
    def kv_cache_enabled(self) -> bool:
        return False

    async def encode(self, image: torch.Tensor, text: str) -> Embedding:
        if self._fail_encode:
            raise BackendError("Intentional test failure")
        if self._simulate_timeout:
            await asyncio.sleep(10)  # will be cancelled by test's wait_for
        await asyncio.sleep(self._latency_ms / 1000.0)
        return Embedding(
            data=torch.zeros(self._hidden_dim, dtype=torch.float16),
            model_id=self.model_id,
        )

    async def generate(self, image: torch.Tensor, text: str):
        yield  # empty async generator

    async def warmup(self) -> None:
        self._warmed_up = True

    async def shutdown(self) -> None:
        self._warmed_up = False


# ── Protocol compliance ───────────────────────────────────────────────────────

class TestVLMBackendProtocol:
    """Verify MockVLMBackend satisfies the VLMBackend Protocol."""

    def test_protocol_check(self):
        backend = MockVLMBackend()
        assert isinstance(backend, VLMBackend)

    def test_model_id(self):
        assert MockVLMBackend().model_id == "mock:test-vlm"

    def test_latency_budget(self):
        assert MockVLMBackend().latency_budget_ms == 500.0

    def test_kv_cache_flag(self):
        assert MockVLMBackend().kv_cache_enabled is False


# ── Embedding correctness ─────────────────────────────────────────────────────

class TestEmbedding:
    def test_dim_property(self):
        emb = Embedding(data=torch.zeros(128, dtype=torch.float16))
        assert emb.dim == 128

    def test_age_ms(self):
        emb = Embedding(data=torch.zeros(16, dtype=torch.float16))
        time.sleep(0.05)
        assert emb.age_ms >= 50.0

    def test_is_stale(self):
        emb = Embedding(data=torch.zeros(16))
        assert not emb.is_stale(max_age_ms=1000.0)
        time.sleep(0.1)
        assert emb.is_stale(max_age_ms=50.0)

    def test_model_id_default(self):
        emb = Embedding(data=torch.zeros(8))
        assert emb.model_id == "unknown"


# ── Encode ────────────────────────────────────────────────────────────────────

class TestEncode:
    @pytest.mark.asyncio
    async def test_encode_returns_embedding(self):
        backend = MockVLMBackend(hidden_dim=64)
        emb = await backend.encode(make_image(), "test")
        assert isinstance(emb, Embedding)
        assert emb.dim == 64

    @pytest.mark.asyncio
    async def test_encode_dtype_float16(self):
        backend = MockVLMBackend()
        emb = await backend.encode(make_image(), "test")
        assert emb.data.dtype == torch.float16

    @pytest.mark.asyncio
    async def test_encode_failure_raises_backend_error(self):
        backend = MockVLMBackend(fail_encode=True)
        with pytest.raises(BackendError):
            await backend.encode(make_image(), "test")

    @pytest.mark.asyncio
    async def test_encode_timeout(self):
        backend = MockVLMBackend(simulate_timeout=True)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                backend.encode(make_image(), "test"),
                timeout=0.05,
            )


# ── Warmup / shutdown ─────────────────────────────────────────────────────────

class TestLifecycle:
    @pytest.mark.asyncio
    async def test_warmup_sets_flag(self):
        backend = MockVLMBackend()
        assert not backend._warmed_up
        await backend.warmup()
        assert backend._warmed_up

    @pytest.mark.asyncio
    async def test_shutdown_clears_flag(self):
        backend = MockVLMBackend()
        await backend.warmup()
        await backend.shutdown()
        assert not backend._warmed_up


# ── QuantizationConfig integration ────────────────────────────────────────────

class TestQuantizationConfig:
    def test_full_precision_factory(self):
        cfg = QuantizationConfig.full_precision()
        assert cfg.vision_encoder == "fp16"
        assert cfg.llm_backbone == "fp16"
        assert cfg.action_head == "fp16"

    def test_cloud_aggressive_factory(self):
        cfg = QuantizationConfig.cloud_aggressive()
        assert cfg.llm_backbone == "w4a16"
        assert cfg.action_head == "fp16"

    def test_edge_balanced_factory(self):
        cfg = QuantizationConfig.edge_balanced()
        assert cfg.llm_backbone == "gguf_q4"

    def test_validate_flow_matching_fp16_ok(self):
        cfg = QuantizationConfig.full_precision()
        cfg.validate_for_head_type("flow_matching")  # should not raise

    def test_validate_flow_matching_w8a16_raises(self):
        from vla_runtime.errors import PrecisionDegradationError
        cfg = QuantizationConfig(action_head="w8a16")
        with pytest.raises(PrecisionDegradationError):
            cfg.validate_for_head_type("flow_matching")

    def test_validate_diffusion_fp16_ok(self):
        cfg = QuantizationConfig.full_precision()
        cfg.validate_for_head_type("diffusion")  # should not raise

    def test_validate_discrete_token_w8a16_ok(self):
        cfg = QuantizationConfig(action_head="w8a16")
        cfg.validate_for_head_type("discrete_token")  # allowed

    def test_summary(self):
        cfg = QuantizationConfig.cloud_aggressive()
        summary = cfg.summary()
        assert "w4a16" in summary
        assert "w8a8" in summary
