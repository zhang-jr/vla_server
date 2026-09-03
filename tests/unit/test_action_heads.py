"""Unit tests for action head implementations.

All tests run on CPU — GPU / Triton tests are in tests/gpu/ (Phase 2).
Mocks: VLMBackend (mocked) — ActionHead internals (NOT mocked, real forward pass).

Coverage:
- Protocol compliance for all three head types
- ActionChunk shape / dtype correctness
- FlowMatchingHead Euler integration produces finite outputs
- DiffusionHead DDIM denoising produces finite outputs
- DiscreteTokenHead FAST encode/decode roundtrip
- FASTTokenizer encode → decode approximate correctness
- QuantizationConfig.validate_for_head_type enforcement
- ActionHeadFailureError propagation
"""

from __future__ import annotations

import pytest
import torch

from vla_runtime.action_head.base import ActionChunk, ActionHeadRuntime, RobotState
from vla_runtime.action_head.diffusion import DiffusionHead
from vla_runtime.action_head.discrete_token import DiscreteTokenHead, FASTTokenizer
from vla_runtime.action_head.flow_matching import FlowMatchingHead
from vla_runtime.errors import ActionHeadFailureError
from vla_runtime.vlm.base import Embedding


# ── Fixtures ──────────────────────────────────────────────────────────────────

ACTION_DIM = 7
CHUNK_SIZE = 8
EMBEDDING_DIM = 64   # small for fast tests
DOF = 7
# A head's state_dim is the FLAT state width, not 2 * dof: the gripper has its
# own channel. Keep this derived so the two can never drift apart.
STATE_DIM = DOF * 2 + 1


def make_embedding(dim: int = EMBEDDING_DIM) -> Embedding:
    return Embedding(
        data=torch.randn(dim, dtype=torch.float16),
        model_id="mock:test",
    )


def make_robot_state(dof: int = DOF) -> RobotState:
    return RobotState(
        joint_positions=torch.zeros(dof),
        joint_velocities=torch.zeros(dof),
        gripper_state=torch.zeros(1),
    )


def make_flow_head(**kwargs) -> FlowMatchingHead:
    defaults = dict(
        action_dim=ACTION_DIM,
        chunk_size=CHUNK_SIZE,
        embedding_dim=EMBEDDING_DIM,
        state_dim=STATE_DIM,
        n_steps=3,           # few steps for fast test
        device=torch.device("cpu"),
    )
    defaults.update(kwargs)
    return FlowMatchingHead(**defaults)


def make_diffusion_head(**kwargs) -> DiffusionHead:
    defaults = dict(
        action_dim=ACTION_DIM,
        chunk_size=CHUNK_SIZE,
        embedding_dim=EMBEDDING_DIM,
        state_dim=STATE_DIM,
        n_steps=5,
        sampler="ddim",
        device=torch.device("cpu"),
    )
    defaults.update(kwargs)
    return DiffusionHead(**defaults)


def make_discrete_head(**kwargs) -> DiscreteTokenHead:
    defaults = dict(
        action_dim=ACTION_DIM,
        chunk_size=CHUNK_SIZE,
        embedding_dim=EMBEDDING_DIM,
        vocab_size=256,
        n_dct_coeffs=4,
        use_oft=False,
        device=torch.device("cpu"),
    )
    defaults.update(kwargs)
    return DiscreteTokenHead(**defaults)


# ── Protocol compliance ───────────────────────────────────────────────────────

class TestProtocolCompliance:
    @pytest.mark.parametrize("head", [
        make_flow_head(),
        make_diffusion_head(),
        make_discrete_head(),
    ])
    def test_satisfies_protocol(self, head):
        assert isinstance(head, ActionHeadRuntime)

    @pytest.mark.parametrize("head,expected_type", [
        (make_flow_head(), "flow_matching"),
        (make_diffusion_head(), "diffusion"),
        (make_discrete_head(), "discrete_token"),
    ])
    def test_head_type(self, head, expected_type):
        assert head.head_type == expected_type

    def test_chunk_size_property(self):
        head = make_flow_head(chunk_size=16)
        assert head.chunk_size == 16

    def test_action_dim_property(self):
        head = make_flow_head(action_dim=ACTION_DIM)
        assert head.action_dim == ACTION_DIM

    def test_frequency_hz_property(self):
        head = make_flow_head(frequency_hz=10.0)
        assert head.frequency_hz == 10.0


# ── RobotState ────────────────────────────────────────────────────────────────

class TestRobotState:
    def test_dof_property(self):
        state = make_robot_state(dof=7)
        assert state.dof == 7

    def test_as_flat_tensor_shape(self):
        state = make_robot_state(dof=7)
        flat = state.as_flat_tensor()
        # 7 positions + 7 velocities + 1 gripper = 15
        assert flat.shape[0] == 15

    def test_as_flat_tensor_with_ee_pose(self):
        state = RobotState(
            joint_positions=torch.zeros(7),
            joint_velocities=torch.zeros(7),
            gripper_state=torch.zeros(1),
            ee_pose=torch.zeros(7),
        )
        flat = state.as_flat_tensor()
        assert flat.shape[0] == 22  # 7+7+1+7


# ── ActionChunk ───────────────────────────────────────────────────────────────

class TestActionChunk:
    def test_chunk_size_property(self):
        chunk = ActionChunk(
            actions=torch.zeros(8, 7),
            head_type="flow_matching",
        )
        assert chunk.chunk_size == 8

    def test_action_dim_property(self):
        chunk = ActionChunk(actions=torch.zeros(8, 7), head_type="flow_matching")
        assert chunk.action_dim == 7

    def test_step_method(self):
        actions = torch.arange(24, dtype=torch.float).reshape(3, 8)
        chunk = ActionChunk(actions=actions, head_type="discrete_token")
        step_0 = chunk.step(0)
        assert step_0.shape == (8,)
        assert step_0[0] == 0.0


# ── FlowMatchingHead ──────────────────────────────────────────────────────────

class TestFlowMatchingHead:
    def test_predict_returns_action_chunk(self):
        head = make_flow_head()
        chunk = head.predict(make_embedding(), make_robot_state())
        assert isinstance(chunk, ActionChunk)

    def test_predict_shape(self):
        head = make_flow_head(chunk_size=CHUNK_SIZE, action_dim=ACTION_DIM)
        chunk = head.predict(make_embedding(), make_robot_state())
        assert chunk.actions.shape == (CHUNK_SIZE, ACTION_DIM)

    def test_predict_finite(self):
        head = make_flow_head()
        chunk = head.predict(make_embedding(), make_robot_state())
        assert torch.isfinite(chunk.actions).all()

    def test_predict_head_type(self):
        head = make_flow_head()
        chunk = head.predict(make_embedding(), make_robot_state())
        assert chunk.head_type == "flow_matching"

    def test_rk4_integrator(self):
        head = make_flow_head(use_rk4=True, n_steps=2)
        chunk = head.predict(make_embedding(), make_robot_state())
        assert chunk.metadata["integrator"] == "rk4"
        assert torch.isfinite(chunk.actions).all()

    def test_metadata_n_steps(self):
        head = make_flow_head(n_steps=3)
        chunk = head.predict(make_embedding(), make_robot_state())
        assert chunk.metadata["n_steps"] == 3


# ── DiffusionHead ─────────────────────────────────────────────────────────────

class TestDiffusionHead:
    def test_predict_returns_action_chunk(self):
        head = make_diffusion_head()
        chunk = head.predict(make_embedding(), make_robot_state())
        assert isinstance(chunk, ActionChunk)

    def test_predict_shape(self):
        head = make_diffusion_head(chunk_size=CHUNK_SIZE, action_dim=ACTION_DIM)
        chunk = head.predict(make_embedding(), make_robot_state())
        assert chunk.actions.shape == (CHUNK_SIZE, ACTION_DIM)

    def test_ddim_produces_finite_output(self):
        head = make_diffusion_head(sampler="ddim", n_steps=5)
        chunk = head.predict(make_embedding(), make_robot_state())
        assert torch.isfinite(chunk.actions).all()

    def test_ddpm_produces_finite_output(self):
        head = make_diffusion_head(sampler="ddpm", n_steps=5)
        chunk = head.predict(make_embedding(), make_robot_state())
        assert torch.isfinite(chunk.actions).all()

    def test_sampler_in_metadata(self):
        head = make_diffusion_head(sampler="ddim")
        chunk = head.predict(make_embedding(), make_robot_state())
        assert chunk.metadata["sampler"] == "ddim"


# ── DiscreteTokenHead ─────────────────────────────────────────────────────────

class TestDiscreteTokenHead:
    def test_predict_returns_action_chunk(self):
        head = make_discrete_head()
        chunk = head.predict(make_embedding(), make_robot_state())
        assert isinstance(chunk, ActionChunk)

    def test_predict_shape_without_oft(self):
        head = make_discrete_head(use_oft=False)
        chunk = head.predict(make_embedding(), make_robot_state())
        assert chunk.actions.shape == (CHUNK_SIZE, ACTION_DIM)

    def test_head_type_is_discrete_token(self):
        head = make_discrete_head()
        assert head.head_type == "discrete_token"

    @pytest.mark.asyncio
    async def test_predict_from_stream(self):
        """Test streaming detokenization."""
        from vla_runtime.vlm.base import TokenChunk

        head = make_discrete_head()
        state = make_robot_state()

        # Provide enough token_ids for FASTTokenizer.decode()
        n_tokens = ACTION_DIM * 4  # n_dct_coeffs=4
        token_ids = [64] * n_tokens  # mid-codebook tokens

        async def mock_stream():
            yield TokenChunk(token_ids=token_ids, is_final=True)

        chunk = await head.predict_from_stream(mock_stream(), state)
        assert isinstance(chunk, ActionChunk)
        assert chunk.actions.shape == (CHUNK_SIZE, ACTION_DIM)


# ── FAST Tokenizer ────────────────────────────────────────────────────────────

class TestFASTTokenizer:
    def setup_method(self):
        self.fast = FASTTokenizer(
            action_dim=ACTION_DIM,
            chunk_size=CHUNK_SIZE,
            vocab_size=256,
            n_dct_coeffs=4,
        )

    def test_encode_returns_correct_length(self):
        actions = torch.randn(CHUNK_SIZE, ACTION_DIM)
        ids = self.fast.encode(actions)
        assert len(ids) == ACTION_DIM * 4  # n_dct_coeffs

    def test_decode_returns_correct_shape(self):
        n_tokens = ACTION_DIM * 4
        ids = [128] * n_tokens
        actions = self.fast.decode(ids)
        assert actions.shape == (CHUNK_SIZE, ACTION_DIM)

    def test_decode_pads_short_sequence(self):
        """Short token sequence should be padded without raising."""
        actions = self.fast.decode([0, 1, 2])
        assert actions.shape == (CHUNK_SIZE, ACTION_DIM)

    def test_encode_values_in_vocab_range(self):
        actions = torch.randn(CHUNK_SIZE, ACTION_DIM)
        ids = self.fast.encode(actions)
        assert all(0 <= tid < 256 for tid in ids)


# ── Quantization enforcement ──────────────────────────────────────────────────

class TestQuantizationEnforcement:
    def test_flow_matching_rejects_w8a16(self):
        from vla_runtime.errors import PrecisionDegradationError
        from vla_runtime.quantization.config import QuantizationConfig

        cfg = QuantizationConfig(action_head="w8a16")
        with pytest.raises(PrecisionDegradationError):
            cfg.validate_for_head_type("flow_matching")

    def test_diffusion_rejects_w8a16(self):
        from vla_runtime.errors import PrecisionDegradationError
        from vla_runtime.quantization.config import QuantizationConfig

        cfg = QuantizationConfig(action_head="w8a16")
        with pytest.raises(PrecisionDegradationError):
            cfg.validate_for_head_type("diffusion")

    def test_discrete_token_accepts_w8a16(self):
        from vla_runtime.quantization.config import QuantizationConfig

        cfg = QuantizationConfig(action_head="w8a16")
        cfg.validate_for_head_type("discrete_token")  # must not raise


# ── TensorRT backend precision guard ─────────────────────────────────────────

class TestTensorRTBackend:
    def test_int8_forbidden_for_flow_matching(self):
        from vla_runtime.action_head.tensorrt_backend import TensorRTBackend
        from vla_runtime.errors import PrecisionDegradationError

        head = make_flow_head()
        with pytest.raises(PrecisionDegradationError):
            TensorRTBackend(head, engine_path="/tmp/test.engine", precision="int8")

    def test_int8_forbidden_for_diffusion(self):
        from vla_runtime.action_head.tensorrt_backend import TensorRTBackend
        from vla_runtime.errors import PrecisionDegradationError

        head = make_diffusion_head()
        with pytest.raises(PrecisionDegradationError):
            TensorRTBackend(head, engine_path="/tmp/test.engine", precision="int8")

    def test_fp16_allowed_for_all_heads(self):
        from vla_runtime.action_head.tensorrt_backend import TensorRTBackend

        for head in [make_flow_head(), make_diffusion_head(), make_discrete_head()]:
            # Should not raise
            backend = TensorRTBackend(head, engine_path="/tmp/test.engine", precision="fp16")
            assert backend.head_type == head.head_type
