"""Unit tests for the reference action head and the shared head types.

All tests run on CPU. Real model heads (π0.5 / SmolVLA experts) are covered by
the GPU tests in ``benchmarks/benchmark_sim.py``.

Coverage:
- Protocol compliance of FlowMatchingHead
- RobotState / ActionChunk shape correctness
- FlowMatchingHead Euler / RK4 integration produces finite outputs
- QuantizationConfig.validate_for_head_type enforcement (ADR-003)
- Condition-kind acceptance (ADR-019)
"""

from __future__ import annotations

import pytest
import torch

from vla_runtime.action_head.base import ActionChunk, ActionHeadRuntime, RobotState
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






# ── Protocol compliance ───────────────────────────────────────────────────────

class TestProtocolCompliance:
    def test_satisfies_protocol(self):
        assert isinstance(make_flow_head(), ActionHeadRuntime)

    def test_head_type(self):
        assert make_flow_head().head_type == "flow_matching"

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


# ── Condition kind acceptance (ADR-019) ───────────────────────────────────────

class TestConditionKindAcceptance:
    """The built-in MLP heads read a pooled vector only."""

    def test_flow_head_rejects_prefix_tokens(self):
        head = make_flow_head()
        emb = Embedding(data=torch.randn(5, EMBEDDING_DIM, dtype=torch.float16))
        with pytest.raises(ActionHeadFailureError, match="prefix_tokens"):
            head.predict(emb, make_robot_state())

    def test_flow_head_rejects_per_layer_kv(self):
        from vla_runtime.vlm.base import LayerKV

        t = torch.zeros(1, 4, 8, dtype=torch.float16)
        emb = Embedding(kind="per_layer_kv", layer_kv=[LayerKV(key=t, value=t)])
        with pytest.raises(ActionHeadFailureError, match="per_layer_kv"):
            make_flow_head().predict(emb, make_robot_state())
