"""Unit tests for VLM backend implementations.

Tests use mock VLM backends to avoid requiring actual GPU / model weights.
Kernel tests (kernels/) are intentionally excluded here — they require real GPU.

Coverage:
- VLMBackend Protocol compliance
- Embedding type correctness, condition kinds (ADR-019)
- ActionRequest multi-view validation
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
from vla_runtime.vlm.base import ActionRequest, Embedding, LayerKV, VLMBackend
from tests.unit.factories import make_robot_state


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_image(h: int = 224, w: int = 224, c: int = 3) -> torch.Tensor:
    return torch.rand(c, h, w)


def make_views() -> dict[str, torch.Tensor]:
    return {"base": make_image()}


def make_request(instruction: str = "pick up the cup") -> ActionRequest:
    return ActionRequest(
        request_id="test-001",
        images=make_views(),
        instruction=instruction,
        robot_state=make_robot_state(),
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

    async def encode(self, images, text: str, state=None) -> Embedding:
        if self._fail_encode:
            raise BackendError("Intentional test failure")
        if self._simulate_timeout:
            await asyncio.sleep(10)  # will be cancelled by test's wait_for
        await asyncio.sleep(self._latency_ms / 1000.0)
        return Embedding(
            data=torch.zeros(self._hidden_dim, dtype=torch.float16),
            model_id=self.model_id,
        )

    async def generate(self, images, text: str, state=None):
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
        emb = await backend.encode(make_views(), "test")
        assert isinstance(emb, Embedding)
        assert emb.dim == 64

    @pytest.mark.asyncio
    async def test_encode_dtype_float16(self):
        backend = MockVLMBackend()
        emb = await backend.encode(make_views(), "test")
        assert emb.data.dtype == torch.float16

    @pytest.mark.asyncio
    async def test_encode_failure_raises_backend_error(self):
        backend = MockVLMBackend(fail_encode=True)
        with pytest.raises(BackendError):
            await backend.encode(make_views(), "test")

    @pytest.mark.asyncio
    async def test_encode_timeout(self):
        backend = MockVLMBackend(simulate_timeout=True)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                backend.encode(make_views(), "test"),
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


# ── Condition kinds (ADR-019) ─────────────────────────────────────────────────

def _kv(seq_len: int, heads: int = 1, head_dim: int = 8) -> LayerKV:
    t = torch.zeros(heads, seq_len, head_dim, dtype=torch.float16)
    return LayerKV(key=t, value=t.clone())


class TestConditionKinds:
    def test_kind_inferred_from_data_rank(self):
        assert Embedding(data=torch.zeros(8)).kind == "pooled"
        assert Embedding(data=torch.zeros(5, 8)).kind == "prefix_tokens"

    def test_seq_len(self):
        assert Embedding(data=torch.zeros(8)).seq_len is None
        assert Embedding(data=torch.zeros(5, 8)).seq_len == 5
        emb = Embedding(kind="per_layer_kv", layer_kv=[_kv(6), _kv(6)])
        assert emb.seq_len == 6
        assert emb.num_layers == 2

    def test_per_layer_kv_needs_layers(self):
        with pytest.raises(ValueError, match="layer_kv"):
            Embedding(kind="per_layer_kv")

    def test_per_layer_kv_layers_must_agree_on_length(self):
        with pytest.raises(ValueError, match="prefix length"):
            Embedding(kind="per_layer_kv", layer_kv=[_kv(6), _kv(7)])

    def test_layer_kv_shape_checks(self):
        with pytest.raises(ValueError, match="mismatch"):
            LayerKV(key=torch.zeros(1, 4, 8), value=torch.zeros(1, 5, 8))
        with pytest.raises(ValueError, match="num_kv_heads"):
            LayerKV(key=torch.zeros(4, 8), value=torch.zeros(4, 8))

    def test_projected_features_needs_features(self):
        with pytest.raises(ValueError, match="features"):
            Embedding(kind="projected_features")
        emb = Embedding(kind="projected_features", features={"vl": torch.zeros(3, 8)})
        assert emb.seq_len is None

    def test_pooled_rejects_wrong_rank(self):
        with pytest.raises(ValueError, match="pooled"):
            Embedding(data=torch.zeros(2, 8), kind="pooled")

    def test_dim_undefined_without_data(self):
        emb = Embedding(kind="per_layer_kv", layer_kv=[_kv(4)])
        with pytest.raises(ValueError, match="no `data`"):
            _ = emb.dim

    def test_attention_mask_validated(self):
        mask = torch.tensor([True, True, False])
        assert Embedding(data=torch.zeros(3, 8), attention_mask=mask).attention_mask is mask
        with pytest.raises(ValueError, match="length"):
            Embedding(data=torch.zeros(4, 8), attention_mask=mask)
        with pytest.raises(ValueError, match="bool"):
            Embedding(data=torch.zeros(3, 8), attention_mask=torch.ones(3))

    def test_nbytes_counts_every_payload(self):
        emb = Embedding(
            kind="per_layer_kv",
            layer_kv=[_kv(10, heads=2, head_dim=4)] * 3,
            attention_mask=torch.ones(10, dtype=torch.bool),
        )
        # 3 layers x (K + V) x 2 heads x 10 tokens x 4 dims x 2 bytes + 10-byte mask
        assert emb.nbytes == 3 * 2 * (2 * 10 * 4 * 2) + 10

    def test_pi0_scale_prefix_kv_is_megabytes(self):
        """Why the edge-cloud split cannot sit at the VLM -> head boundary for pi0-like models.

        Shape assumption (~800 prefix tokens, 18 layers, 1 KV head, head_dim 256)
        mirrors a Gemma-2B backbone; see ADR-018.
        """
        emb = Embedding(kind="per_layer_kv", layer_kv=[_kv(800, heads=1, head_dim=256)] * 18)
        assert 14e6 < emb.nbytes < 16e6


class TestActionRequest:
    def test_requires_a_camera_view(self):
        with pytest.raises(ValueError, match="camera"):
            ActionRequest(
                request_id="r",
                images={},
                instruction="x",
                robot_state=make_robot_state(),
            )

    def test_keeps_view_order(self):
        req = ActionRequest(
            request_id="r",
            images={"wrist": make_image(), "base": make_image()},
            instruction="x",
            robot_state=make_robot_state(),
        )
        assert list(req.images) == ["wrist", "base"]
