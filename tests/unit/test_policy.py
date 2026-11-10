"""Unit tests for head weight sharing (ADR-012 ③) and the Policy adapter layer (ADR-018).

Coverage:
- spawn(): new instance, same weights; group-wide load_weights / to_device
- Shared weights are frozen for inference (eval mode, no grad)
- TensorRTBackend.spawn(): engine shared, execution context per instance
- Policy construction-time validation (condition kind, head structure)
- AsyncScheduler.from_policy / FleetScheduler.from_policy
- Fleet refuses one head instance registered for two robots
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from vla_runtime.action_head.base import WeightSharingHead
from vla_runtime.action_head.diffusion import DiffusionHead
from vla_runtime.action_head.discrete_token import DiscreteTokenHead
from vla_runtime.action_head.flow_matching import FlowMatchingHead
from vla_runtime.action_head.tensorrt_backend import TensorRTBackend
from vla_runtime.capabilities import PolicyCapabilities
from vla_runtime.policy import Policy, PolicyConfigurationError
from vla_runtime.scheduler.async_scheduler import AsyncScheduler
from vla_runtime.serving.fleet import FleetConfig, FleetScheduler
from vla_runtime.vlm.base import Embedding
from tests.unit.factories import coupled_caps, decoupled_caps, make_request, make_robot_state

EMBED_DIM = 16
DOF = 7
STATE_DIM = 2 * DOF + 1


def _flow_head() -> FlowMatchingHead:
    return FlowMatchingHead(
        action_dim=DOF, chunk_size=4, embedding_dim=EMBED_DIM, state_dim=STATE_DIM, n_steps=2
    )


def _embedding() -> Embedding:
    return Embedding(data=torch.randn(EMBED_DIM, dtype=torch.float16))


class _PooledVLM:
    model_id = "mock"
    latency_budget_ms = 500.0
    kv_cache_enabled = False

    async def encode(self, images, text, state=None) -> Embedding:
        return _embedding()

    async def generate(self, images, text, state=None):
        raise NotImplementedError

    async def warmup(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass


class _NoSpawnHead:
    """A third-party head that satisfies the core Protocol but cannot share weights."""

    head_type = "flow_matching"
    frequency_hz = 10.0
    chunk_size = 4
    action_dim = DOF

    def predict(self, embedding, robot_state):
        raise NotImplementedError

    def load_weights(self, path):
        pass

    def to_device(self, device):
        pass


# ── Weight sharing ────────────────────────────────────────────────────────────

class TestSpawn:
    @pytest.mark.parametrize(
        "make",
        [
            _flow_head,
            lambda: DiffusionHead(
                action_dim=DOF, chunk_size=4, embedding_dim=EMBED_DIM,
                state_dim=STATE_DIM, n_steps=2,
            ),
            lambda: DiscreteTokenHead(
                action_dim=DOF, chunk_size=4, embedding_dim=EMBED_DIM, use_oft=True
            ),
            lambda: DiscreteTokenHead(action_dim=DOF, chunk_size=4, embedding_dim=EMBED_DIM),
        ],
        ids=["flow", "diffusion", "discrete_oft", "discrete_no_weights"],
    )
    def test_spawn_is_a_new_instance_on_the_same_weights(self, make):
        head = make()
        clone = head.spawn()
        assert isinstance(head, WeightSharingHead)
        assert clone is not head
        assert clone.shares_weights_with(head)
        assert head.shares_weights_with(clone)

    def test_independent_heads_do_not_share(self):
        assert not _flow_head().shares_weights_with(_flow_head())

    def test_spawned_heads_both_predict(self):
        head = _flow_head()
        clone = head.spawn()
        for h in (head, clone):
            chunk = h.predict(_embedding(), make_robot_state(DOF))
            assert chunk.actions.shape == (4, DOF)
            assert torch.isfinite(chunk.actions).all()

    def test_load_weights_applies_to_every_sharer(self, tmp_path):
        head = _flow_head()
        clone = head.spawn()
        source = _flow_head()
        path = tmp_path / "velocity.pt"
        torch.save(source._velocity_net.state_dict(), path)

        clone.load_weights(str(path))

        for name, tensor in source._velocity_net.state_dict().items():
            assert torch.equal(head._velocity_net.state_dict()[name], tensor)

    def test_to_device_moves_the_group(self):
        head = _flow_head()
        clone = head.spawn()
        clone.to_device(torch.device("cpu"))
        assert head._device == clone._device == torch.device("cpu")

    def test_shared_weights_are_frozen_for_inference(self):
        net = _flow_head()._velocity_net
        assert not net.training
        assert not any(p.requires_grad for p in net.parameters())


class TestTensorRTSpawn:
    def test_spawn_before_engine_build_shares_wrapped_weights(self, tmp_path):
        trt = TensorRTBackend(_flow_head(), engine_path=tmp_path / "head.engine")
        clone = trt.spawn()
        assert clone is not trt
        assert clone._wrapped is not trt._wrapped
        assert clone.shares_weights_with(trt)
        assert clone._context is None

    def test_spawn_shares_engine_but_not_context(self, tmp_path):
        class _Engine:
            def create_execution_context(self):
                return object()

        trt = TensorRTBackend(_flow_head(), engine_path=tmp_path / "head.engine")
        trt._engine = _Engine()
        trt._context = trt._engine.create_execution_context()

        clone = trt.spawn()
        assert clone._engine is trt._engine
        assert clone._context is not None and clone._context is not trt._context
        assert clone.shares_weights_with(trt)

    def test_spawn_requires_a_sharing_wrapped_head(self, tmp_path):
        trt = TensorRTBackend(_NoSpawnHead(), engine_path=tmp_path / "head.engine")
        with pytest.raises(TypeError, match="weight sharing"):
            trt.spawn()

    def test_delegates_accepted_condition_kinds(self, tmp_path):
        trt = TensorRTBackend(_flow_head(), engine_path=tmp_path / "head.engine")
        assert trt.accepted_condition_kinds == ("pooled",)


# ── Policy validation ─────────────────────────────────────────────────────────

class TestPolicy:
    def test_valid_policy(self):
        policy = Policy("toy", decoupled_caps(), _PooledVLM(), _flow_head())
        head = policy.head_for_robot()
        assert head is not policy.head_prototype
        assert head.shares_weights_with(policy.head_prototype)

    def test_rejects_condition_kind_the_head_cannot_read(self):
        with pytest.raises(PolicyConfigurationError, match="per_layer_kv"):
            Policy("pi0", decoupled_caps("per_layer_kv"), _PooledVLM(), _flow_head())

    def test_rejects_head_structure_contradicting_the_head(self):
        caps = PolicyCapabilities(
            state_placement="none", condition_kind="pooled", head_structure="single_pass"
        )
        with pytest.raises(PolicyConfigurationError, match="iterative"):
            Policy("toy", caps, _PooledVLM(), _flow_head())

    def test_discrete_head_must_not_be_iterative(self):
        head = DiscreteTokenHead(action_dim=DOF, chunk_size=4, embedding_dim=EMBED_DIM)
        with pytest.raises(PolicyConfigurationError, match="discrete_token"):
            Policy("openvla", coupled_caps(), _PooledVLM(), head)

    def test_head_without_spawn_cannot_be_multiplied(self):
        policy = Policy("third_party", coupled_caps(), _PooledVLM(), _NoSpawnHead())
        with pytest.raises(PolicyConfigurationError, match="spawn"):
            policy.head_for_robot()


# ── Schedulers built from a policy ────────────────────────────────────────────

class TestFromPolicy:
    @pytest.mark.asyncio
    async def test_async_scheduler_from_policy(self):
        policy = Policy("toy", decoupled_caps(), _PooledVLM(), _flow_head())
        sched = AsyncScheduler.from_policy(policy, robot_id="arm_left")
        assert sched._action_head is not policy.head_prototype
        assert sched._action_head.shares_weights_with(policy.head_prototype)
        assert sched._caps is policy.capabilities

        await sched.start()
        chunks = [c async for c in sched.stream(await sched.submit(make_request(dof=DOF)))]
        assert len(chunks) == 1
        await sched.stop()

    @pytest.mark.asyncio
    async def test_fleet_spawns_one_head_per_robot_on_shared_weights(self):
        policy = Policy("toy", decoupled_caps(), _PooledVLM(), _flow_head())
        fleet = FleetScheduler.from_policy(policy, FleetConfig(max_robots=3))
        await fleet.start()
        for rid in ("a", "b", "c"):
            await fleet.register_robot(rid)

        heads = [fleet._robots[rid].action_head for rid in ("a", "b", "c")]
        assert len({id(h) for h in heads}) == 3, "each robot needs its own instance"
        assert all(h.shares_weights_with(policy.head_prototype) for h in heads)

        handle = await fleet.submit("b", make_request(dof=DOF))
        chunks = [c async for c in fleet.stream(handle)]
        assert len(chunks) == 1
        await fleet.stop()

    @pytest.mark.asyncio
    async def test_fleet_without_policy_needs_explicit_head(self):
        fleet = FleetScheduler(vlm_backend=_PooledVLM(), capabilities=decoupled_caps())
        with pytest.raises(ValueError, match="from_policy"):
            await fleet.register_robot("a")

    @pytest.mark.asyncio
    async def test_fleet_rejects_one_instance_for_two_robots(self):
        fleet = FleetScheduler(vlm_backend=_PooledVLM(), capabilities=decoupled_caps())
        head = _flow_head()
        await fleet.register_robot("a", head)
        with pytest.raises(ValueError, match="already registered for 'a'"):
            await fleet.register_robot("b", head)
        await fleet.register_robot("b", head.spawn())
