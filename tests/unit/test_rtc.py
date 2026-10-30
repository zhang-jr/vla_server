"""Unit tests for Real-Time Chunking (ADR-015).

CPU only. Covers:
- Prefix weight schedules (frozen / soft / zero regions, LeRobot parity)
- Guidance weight clipping across tau
- Guided velocity against a linear field with an analytic Jacobian (vjp, identity)
- Context → guidance preparation: padding, truncation, delay clamping, errors
- FlowMatchingHead: no context = unchanged sampling; guidance pulls the frozen prefix
- ChunkBuffer.rtc_context: queued snapshot, delay estimate without the trim ceiling
- Capability wiring: Policy / AsyncScheduler / FleetScheduler reject mismatches
  and deliver the context to the head
"""

from __future__ import annotations

import asyncio
import math
from typing import Optional

import pytest
import torch

from vla_runtime.action_head.base import (
    ActionChunk,
    ActionHeadContext,
    RobotState,
    head_supports_rtc,
)
from vla_runtime.action_head.diffusion import DiffusionHead
from vla_runtime.action_head.flow_matching import FlowMatchingHead
from vla_runtime.action_head.rtc import (
    RTCConfig,
    guidance_weight,
    guided_velocity,
    prefix_weights,
    prepare_guidance,
)
from vla_runtime.capabilities import PolicyCapabilities
from vla_runtime.errors import ActionHeadFailureError, PolicyCapabilityError
from vla_runtime.policy import Policy, PolicyConfigurationError
from vla_runtime.scheduler.async_scheduler import AsyncScheduler
from vla_runtime.scheduler.chunk_buffer import ChunkBuffer
from vla_runtime.serving.fleet import FleetConfig, FleetScheduler
from vla_runtime.vlm.base import Embedding
from tests.unit.factories import make_request, make_robot_state

EMBED_DIM = 16
ACTION_DIM = 3
CHUNK = 8
DOF = 2
STATE_DIM = 2 * DOF + 1


def rtc_caps(supports_rtc: bool = True) -> PolicyCapabilities:
    return PolicyCapabilities(
        state_placement="suffix",
        condition_kind="pooled",
        head_structure="iterative",
        supports_rtc=supports_rtc,
    )


def make_head(**kwargs) -> FlowMatchingHead:
    return FlowMatchingHead(
        action_dim=ACTION_DIM,
        chunk_size=CHUNK,
        embedding_dim=EMBED_DIM,
        state_dim=STATE_DIM,
        **kwargs,
    )


def make_embedding() -> Embedding:
    return Embedding(data=torch.zeros(EMBED_DIM, dtype=torch.float16), model_id="mock")


class _MockVLM:
    latency_budget_ms: float = 500.0
    kv_cache_enabled: bool = False

    async def encode(self, images, text, state=None) -> Embedding:
        return make_embedding()

    async def generate(self, images, text, state=None):
        raise NotImplementedError

    async def warmup(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass


class _SpyRTCHead:
    """RTC-capable head that records the context it was given."""

    supports_rtc = True
    head_type = "flow_matching"
    frequency_hz = 10.0
    chunk_size = CHUNK
    action_dim = ACTION_DIM

    def __init__(self) -> None:
        self.contexts: list[Optional[ActionHeadContext]] = []

    def predict(
        self,
        embedding: Embedding,
        robot_state: RobotState,
        context: Optional[ActionHeadContext] = None,
    ) -> ActionChunk:
        self.contexts.append(context)
        return ActionChunk(actions=torch.ones(CHUNK, ACTION_DIM), head_type="flow_matching")

    def load_weights(self, checkpoint_path: str) -> None:
        pass

    def to_device(self, device: torch.device) -> None:
        pass


class _PlainHead(_SpyRTCHead):
    """Same head without the RTC declaration."""

    supports_rtc = False


# ── Prefix weights ────────────────────────────────────────────────────────────

def test_prefix_weights_regions():
    w = prefix_weights(delay_steps=2, overlap_steps=6, horizon=10, schedule="linear")
    assert w.shape == (10,)
    assert torch.equal(w[:2], torch.ones(2))
    assert torch.equal(w[6:], torch.zeros(4))
    soft = w[2:6]
    assert torch.all(soft > 0) and torch.all(soft < 1)
    assert torch.all(soft[1:] < soft[:-1])   # strictly decaying


def test_linear_schedule_matches_lerobot_formula():
    # LeRobot: linspace(1, 0, L + 2)[1:-1] over [delay, overlap), L = overlap - delay.
    delay, overlap = 3, 9
    w = prefix_weights(delay, overlap, 12, "linear")
    expected = torch.linspace(1, 0, overlap - delay + 2)[1:-1]
    torch.testing.assert_close(w[delay:overlap], expected)


def test_exp_schedule_is_below_linear_and_matches_formula():
    lin = prefix_weights(1, 7, 8, "linear")
    exp = prefix_weights(1, 7, 8, "exp")
    c = lin[1:7]
    torch.testing.assert_close(exp[1:7], c * torch.expm1(c) / (math.e - 1))
    assert torch.all(exp[1:7] <= lin[1:7])


def test_ones_and_zeros_schedules():
    assert torch.equal(prefix_weights(2, 5, 8, "ones"), torch.tensor([1.0] * 5 + [0.0] * 3))
    assert torch.equal(prefix_weights(2, 5, 8, "zeros"), torch.tensor([1.0] * 2 + [0.0] * 6))


def test_delay_beyond_overlap_freezes_only_the_overlap():
    w = prefix_weights(delay_steps=10, overlap_steps=4, horizon=8)
    assert torch.equal(w, torch.tensor([1.0] * 4 + [0.0] * 4))


# ── Guidance weight ───────────────────────────────────────────────────────────

def test_guidance_weight_clips_and_vanishes():
    assert guidance_weight(0.0, 5.0) == 5.0
    assert guidance_weight(1.0, 5.0) == 0.0
    # tau = 0.5: (0.5 / 0.5) * (0.25 + 0.25) / 0.25 = 2
    assert guidance_weight(0.5, 5.0) == pytest.approx(2.0)
    assert guidance_weight(0.05, 5.0) == 5.0


# ── Guided velocity ───────────────────────────────────────────────────────────

def _linear_field(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(ACTION_DIM, ACTION_DIM, generator=g) * 0.3
    b = torch.randn(ACTION_DIM, generator=g)
    return a, b, (lambda x: x @ a.T + b)


def test_vjp_guidance_matches_analytic_jacobian():
    a, b, field = _linear_field()
    x = torch.randn(CHUNK, ACTION_DIM, generator=torch.Generator().manual_seed(1))
    target = torch.randn(CHUNK, ACTION_DIM, generator=torch.Generator().manual_seed(2))
    weights = prefix_weights(2, 6, CHUNK, "linear")
    tau = 0.4

    out = guided_velocity(field, x, tau, target, weights, max_guidance_weight=5.0)

    v = field(x)
    x1 = x + (1 - tau) * v
    err = (target - x1) * weights.unsqueeze(-1)
    jacobian = torch.eye(ACTION_DIM) + (1 - tau) * a     # dx1/dx per row
    correction = err @ jacobian                           # row-wise J^T err
    expected = v + guidance_weight(tau, 5.0) * correction
    torch.testing.assert_close(out, expected)


def test_identity_guidance_skips_the_jacobian():
    _, _, field = _linear_field()
    x = torch.randn(CHUNK, ACTION_DIM, generator=torch.Generator().manual_seed(1))
    target = torch.randn(CHUNK, ACTION_DIM, generator=torch.Generator().manual_seed(2))
    weights = prefix_weights(2, 6, CHUNK, "linear")
    tau = 0.4

    out = guided_velocity(field, x, tau, target, weights, 5.0, jacobian="identity")

    v = field(x)
    err = (target - (x + (1 - tau) * v)) * weights.unsqueeze(-1)
    torch.testing.assert_close(out, v + guidance_weight(tau, 5.0) * err)


def test_zero_weights_leave_velocity_unchanged():
    _, _, field = _linear_field()
    x = torch.randn(CHUNK, ACTION_DIM)
    out = guided_velocity(field, x, 0.3, torch.randn(CHUNK, ACTION_DIM), torch.zeros(CHUNK), 5.0)
    torch.testing.assert_close(out, field(x))


# ── Guidance preparation ──────────────────────────────────────────────────────

def test_prepare_guidance_without_previous_actions_is_unguided():
    config = RTCConfig()
    assert prepare_guidance(None, CHUNK, ACTION_DIM, config) is None
    assert prepare_guidance(ActionHeadContext(), CHUNK, ACTION_DIM, config) is None
    empty = ActionHeadContext(prev_actions=torch.zeros(0, ACTION_DIM))
    assert prepare_guidance(empty, CHUNK, ACTION_DIM, config) is None


def test_prepare_guidance_pads_truncates_and_clamps():
    prev = torch.arange(5 * ACTION_DIM, dtype=torch.float32).reshape(5, ACTION_DIM)
    guidance = prepare_guidance(
        ActionHeadContext(prev_actions=prev, inference_delay_steps=9),
        CHUNK,
        ACTION_DIM,
        RTCConfig(prefix_schedule="linear"),
    )
    assert guidance.target.shape == (CHUNK, ACTION_DIM)
    torch.testing.assert_close(guidance.target[:5], prev)
    assert torch.equal(guidance.target[5:], torch.zeros(3, ACTION_DIM))
    assert guidance.overlap_steps == 5 and guidance.delay_steps == 5

    long_prev = torch.ones(CHUNK + 4, ACTION_DIM)
    guidance = prepare_guidance(
        ActionHeadContext(prev_actions=long_prev, inference_delay_steps=2),
        CHUNK,
        ACTION_DIM,
        RTCConfig(),
    )
    assert guidance.overlap_steps == CHUNK


def test_prepare_guidance_rejects_bad_shapes():
    with pytest.raises(ValueError, match="prev_actions"):
        prepare_guidance(
            ActionHeadContext(prev_actions=torch.zeros(4, ACTION_DIM + 1)),
            CHUNK,
            ACTION_DIM,
            RTCConfig(),
        )
    with pytest.raises(ValueError, match="inference_delay_steps"):
        prepare_guidance(
            ActionHeadContext(prev_actions=torch.zeros(4, ACTION_DIM), inference_delay_steps=-1),
            CHUNK,
            ACTION_DIM,
            RTCConfig(),
        )


# ── FlowMatchingHead ──────────────────────────────────────────────────────────

def test_flow_matching_head_declares_rtc():
    assert head_supports_rtc(make_head())
    assert not head_supports_rtc(DiffusionHead(action_dim=ACTION_DIM, chunk_size=CHUNK))


def test_context_without_previous_actions_samples_like_no_context():
    head = make_head()
    embedding, state = make_embedding(), make_robot_state(DOF)
    torch.manual_seed(7)
    plain = head.predict(embedding, state)
    torch.manual_seed(7)
    empty = head.predict(embedding, state, ActionHeadContext())
    torch.testing.assert_close(plain.actions, empty.actions)
    assert empty.metadata["rtc"] is False


@pytest.mark.parametrize("jacobian", ["vjp", "identity"])
def test_guidance_pulls_the_frozen_prefix_toward_previous_actions(jacobian):
    head = make_head(
        n_steps=20,
        rtc_config=RTCConfig(max_guidance_weight=10.0, prefix_schedule="ones", jacobian=jacobian),
    )
    embedding, state = make_embedding(), make_robot_state(DOF)
    prev = torch.full((CHUNK, ACTION_DIM), 0.5)
    context = ActionHeadContext(prev_actions=prev, inference_delay_steps=3)

    torch.manual_seed(3)
    plain = head.predict(embedding, state).actions.float()
    torch.manual_seed(3)
    guided = head.predict(embedding, state, context)

    plain_err = (plain - prev).abs().mean()
    guided_err = (guided.actions.float() - prev).abs().mean()
    assert guided_err < 0.5 * plain_err
    assert guided.metadata["rtc"] is True
    assert guided.metadata["rtc_delay_steps"] == 3
    assert guided.metadata["rtc_overlap_steps"] == CHUNK


def test_guidance_works_with_rk4():
    head = make_head(n_steps=4, use_rk4=True, rtc_config=RTCConfig(prefix_schedule="ones"))
    prev = torch.zeros(CHUNK, ACTION_DIM)
    chunk = head.predict(
        make_embedding(), make_robot_state(DOF), ActionHeadContext(prev_actions=prev)
    )
    assert chunk.actions.shape == (CHUNK, ACTION_DIM)
    assert torch.isfinite(chunk.actions.float()).all()


def test_malformed_context_surfaces_as_head_failure():
    head = make_head()
    bad = ActionHeadContext(prev_actions=torch.zeros(4, ACTION_DIM + 2))
    with pytest.raises(ActionHeadFailureError):
        head.predict(make_embedding(), make_robot_state(DOF), bad)


# ── ChunkBuffer.rtc_context ───────────────────────────────────────────────────

def test_rtc_context_snapshots_the_queued_actions():
    buf = ChunkBuffer(blend_policy="trim", control_rate_hz=50.0)
    assert buf.rtc_context().prev_actions is None

    actions = torch.arange(CHUNK * ACTION_DIM, dtype=torch.float32).reshape(CHUNK, ACTION_DIM)
    buf.push(ActionChunk(actions=actions, head_type="flow_matching"))
    buf.next_action()
    buf.next_action()

    context = buf.rtc_context(delay_steps=4)
    torch.testing.assert_close(context.prev_actions, actions[2:])
    assert context.inference_delay_steps == 4
    # A snapshot, not a view: later pops do not change it.
    buf.next_action()
    assert context.prev_actions.shape[0] == CHUNK - 2


def test_rtc_delay_estimate_ignores_the_trim_ceiling():
    buf = ChunkBuffer(blend_policy="trim", control_rate_hz=50.0, max_latency_trim_steps=8)
    assert buf.update_latency_trim(400.0) == 8          # trim is clamped
    assert buf.rtc_context().inference_delay_steps == 20  # 400 ms * 50 Hz
    assert buf.stats()["estimated_delay_steps"] == 20


# ── Capability wiring ─────────────────────────────────────────────────────────

def test_policy_rejects_rtc_declaration_for_a_head_without_it():
    with pytest.raises(PolicyConfigurationError, match="supports_rtc"):
        Policy(
            name="bad",
            capabilities=rtc_caps(),
            vlm_backend=_MockVLM(),
            head_prototype=_PlainHead(),
        )


def test_async_scheduler_rejects_rtc_caps_with_plain_head():
    with pytest.raises(PolicyCapabilityError):
        AsyncScheduler(vlm_backend=_MockVLM(), action_head=_PlainHead(), capabilities=rtc_caps())


async def test_async_scheduler_rejects_context_for_non_rtc_policy():
    scheduler = AsyncScheduler(
        vlm_backend=_MockVLM(), action_head=_PlainHead(), capabilities=rtc_caps(False)
    )
    with pytest.raises(PolicyCapabilityError):
        await scheduler.submit(
            make_request(dof=DOF), head_context=ActionHeadContext(prev_actions=torch.zeros(2, 3))
        )


async def test_async_scheduler_snapshots_its_buffer_at_submit():
    head = _SpyRTCHead()
    scheduler = AsyncScheduler(vlm_backend=_MockVLM(), action_head=head, capabilities=rtc_caps())

    await scheduler.submit(make_request(dof=DOF))      # empty buffer → no previous actions
    scheduler._chunk_buffer.next_action()
    await scheduler.submit(make_request(dof=DOF))      # buffer now holds chunk 1 minus one pop

    assert head.contexts[0] is not None and head.contexts[0].prev_actions is None
    assert head.contexts[1].prev_actions.shape == (CHUNK - 1, ACTION_DIM)
    assert scheduler._head_contexts == {}


async def test_async_scheduler_prefers_the_caller_context():
    head = _SpyRTCHead()
    scheduler = AsyncScheduler(vlm_backend=_MockVLM(), action_head=head, capabilities=rtc_caps())
    given = ActionHeadContext(prev_actions=torch.zeros(5, ACTION_DIM), inference_delay_steps=2)
    await scheduler.submit(make_request(dof=DOF), head_context=given)
    assert head.contexts == [given]


async def test_plain_policy_head_is_called_without_context():
    head = _PlainHead()
    scheduler = AsyncScheduler(vlm_backend=_MockVLM(), action_head=head, capabilities=rtc_caps(False))
    await scheduler.submit(make_request(dof=DOF))
    assert head.contexts == [None]


async def test_fleet_forwards_the_context_and_guards_registration():
    fleet = FleetScheduler(
        vlm_backend=_MockVLM(),
        capabilities=rtc_caps(),
        config=FleetConfig(max_robots=2, vlm_budget_ms=200.0),
    )
    await fleet.start()
    try:
        with pytest.raises(PolicyCapabilityError):
            await fleet.register_robot("plain", _PlainHead())
        head = _SpyRTCHead()
        await fleet.register_robot("robot_0", head)
        given = ActionHeadContext(prev_actions=torch.zeros(3, ACTION_DIM), inference_delay_steps=1)
        handle = await fleet.submit("robot_0", make_request(dof=DOF), head_context=given)
        chunks = [chunk async for chunk in fleet.stream(handle)]
        assert len(chunks) == 1
        assert head.contexts == [given]
    finally:
        await fleet.stop()


async def test_fleet_rejects_context_for_non_rtc_policy():
    fleet = FleetScheduler(
        vlm_backend=_MockVLM(),
        capabilities=rtc_caps(False),
        config=FleetConfig(max_robots=1, vlm_budget_ms=200.0),
    )
    await fleet.start()
    try:
        await fleet.register_robot("robot_0", _PlainHead())
        with pytest.raises(PolicyCapabilityError):
            await fleet.submit(
                "robot_0",
                make_request(dof=DOF),
                head_context=ActionHeadContext(prev_actions=torch.zeros(2, ACTION_DIM)),
            )
    finally:
        await fleet.stop()
