"""Unit tests for fleet admission control (ADR-023).

Tests verify:
  - Round-latency prediction from VLM and per-batch-size head costs
  - Level switching on the prediction, with hysteresis on the way back
  - Admission refusal past admit_latency_ms, and admission without data
  - FleetScheduler wiring: chunks carry the advised threshold, refusal raises
    BudgetExceededError, and a fleet without admission is unchanged

All tests are CPU-only (mock VLM and heads).
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
import torch

from tests.unit.factories import decoupled_caps
from tests.unit.factories import make_request as _factory_request
from vla_runtime.action_head.base import ActionChunk, RobotState
from vla_runtime.errors import BudgetExceededError
from vla_runtime.serving.admission import AdmissionConfig, AdmissionController
from vla_runtime.serving.fleet import FleetConfig, FleetScheduler
from vla_runtime.vlm.base import Embedding

CAPS = decoupled_caps(condition_kind="prefix_tokens")


def _controller(**kwargs) -> AdmissionController:
    return AdmissionController(
        AdmissionConfig(**{"latency_target_ms": 300.0, "skip_first": 0, **kwargs})
    )


# ── Controller ────────────────────────────────────────────────────────────────

def test_no_prediction_before_both_stages_are_measured():
    c = _controller()
    assert c.predict_round_ms(4) is None
    c.record_vlm(50.0)
    assert c.predict_round_ms(4) is None
    assert c.update(4) is None and c.level == "fresh"


def test_round_is_vlm_per_robot_plus_one_head_call():
    c = _controller()
    c.record_vlm(51.0)
    c.record_head(4, 52.0)
    assert c.predict_round_ms(4) == pytest.approx(4 * 51.0 + 52.0)


def test_head_cost_is_interpolated_across_batch_sizes():
    c = _controller()
    c.record_head(1, 30.0)
    c.record_head(3, 50.0)
    assert c.head_estimate(2) == pytest.approx(40.0)
    assert c.head_estimate(5) == pytest.approx(70.0)


def test_single_measured_batch_size_is_taken_as_flat():
    c = _controller()
    c.record_head(2, 44.0)
    assert c.head_estimate(6) == 44.0


def test_stage_costs_are_smoothed():
    c = _controller(ewma_alpha=0.5)
    c.record_vlm(40.0)
    c.record_vlm(60.0)
    assert c.vlm_ms == pytest.approx(50.0)


def test_first_sample_per_stage_and_batch_size_is_dropped():
    c = _controller(skip_first=1)
    c.record_vlm(700.0)  # cold start
    c.record_vlm(50.0)
    c.record_head(2, 650.0)  # CUDA Graph capture at a new batch size
    c.record_head(2, 52.0)
    c.record_head(4, 600.0)
    assert c.vlm_ms == 50.0 and c.head_ms == {2: 52.0}


def test_level_relaxes_over_target_and_recovers_below_hysteresis():
    c = _controller(hysteresis=0.1)
    c.record_vlm(50.0)
    c.record_head(1, 50.0)  # flat head: round = 50 N + 50
    assert c.update(5) is None and c.threshold == 0.9  # 300: at target, not over
    change = c.update(6)  # 350
    assert change is not None and change.level == "relaxed" and change.num_robots == 6
    assert c.threshold == 0.5
    assert c.update(5) is None and c.level == "relaxed"  # 300 > 270: stays relaxed
    change = c.update(4)  # 250 <= 270
    assert change is not None and change.level == "fresh" and c.threshold == 0.9


def test_admission_refuses_past_limit_and_admits_without_data():
    c = _controller(admit_latency_ms=400.0)
    assert c.admits(8) == (True, None)
    c.record_vlm(50.0)
    c.record_head(1, 50.0)
    assert c.admits(7) == (True, 400.0)
    admitted, predicted = c.admits(8)
    assert not admitted and predicted == 450.0


def test_without_limit_every_robot_is_admitted():
    c = _controller()
    c.record_vlm(500.0)
    c.record_head(1, 500.0)
    assert c.admits(8)[0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fresh_threshold": 0.4, "relaxed_threshold": 0.5},
        {"hysteresis": 1.0},
        {"ewma_alpha": 0.0},
        {"skip_first": -1},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        AdmissionConfig(latency_target_ms=300.0, **kwargs)


# ── Fleet wiring ──────────────────────────────────────────────────────────────

class _SlowVLM:
    latency_budget_ms: float = 1000.0
    kv_cache_enabled: bool = False

    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s

    async def encode(self, images, text, state=None) -> Embedding:
        await asyncio.sleep(self.delay_s)
        return Embedding(data=torch.zeros(1, 32), model_id="mock")

    async def generate(self, images, text, state=None):
        raise NotImplementedError

    async def warmup(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass


class _Head:
    head_type = "flow_matching"
    frequency_hz = 10.0
    chunk_size = 4
    action_dim = 7

    def predict(self, embedding: Embedding, robot_state: RobotState) -> ActionChunk:
        return ActionChunk(
            actions=torch.zeros(self.chunk_size, self.action_dim), head_type=self.head_type
        )


async def _one_chunk(fleet: FleetScheduler, robot_id: str) -> ActionChunk:
    handle = await fleet.submit(robot_id, _factory_request(str(uuid.uuid4())))
    (chunk,) = [c async for c in fleet.stream(handle)]
    return chunk


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_fleet_chunks_carry_the_advised_threshold():
    async def run():
        # A 20 ms VLM against a 1 ms target: over target from the first chunk.
        admission = AdmissionConfig(latency_target_ms=1.0, fresh_threshold=0.9, skip_first=0)
        fleet = FleetScheduler(
            _SlowVLM(0.02), CAPS, FleetConfig(max_robots=2, admission=admission)
        )
        await fleet.start()
        try:
            assert fleet.replan_threshold == 0.9
            await fleet.register_robot("r0", _Head())
            chunk = await _one_chunk(fleet, "r0")
            stats = fleet.stats()
        finally:
            await fleet.stop()
        return chunk, stats, fleet.replan_threshold

    chunk, stats, threshold = _run(run())
    assert chunk.metadata["replan_threshold"] == 0.5 == threshold
    assert stats["admission"]["level"] == "relaxed"
    assert stats["admission"]["vlm_ms"] >= 15.0 and 1 in stats["admission"]["head_ms"]


def test_fleet_refuses_a_robot_past_the_admission_limit():
    async def run():
        admission = AdmissionConfig(
            latency_target_ms=1000.0, admit_latency_ms=30.0, skip_first=0
        )
        fleet = FleetScheduler(
            _SlowVLM(0.02), CAPS, FleetConfig(max_robots=4, admission=admission)
        )
        await fleet.start()
        try:
            await fleet.register_robot("r0", _Head())  # no measurements yet: admitted
            await _one_chunk(fleet, "r0")  # one robot: ~20 ms round
            with pytest.raises(BudgetExceededError) as info:
                await fleet.register_robot("r1", _Head())  # two: ~40 ms > 30 ms
            return info.value, fleet.stats()["registered_robots"]
        finally:
            await fleet.stop()

    error, registered = _run(run())
    assert error.budget_ms == 30.0 and error.latency_ms > 30.0
    assert registered == ["r0"]


def test_fleet_without_admission_is_unchanged():
    async def run():
        fleet = FleetScheduler(_SlowVLM(0.0), CAPS, FleetConfig(max_robots=2))
        await fleet.start()
        try:
            await fleet.register_robot("r0", _Head())
            chunk = await _one_chunk(fleet, "r0")
            return chunk, fleet.stats(), fleet.replan_threshold
        finally:
            await fleet.stop()

    chunk, stats, threshold = _run(run())
    assert "replan_threshold" not in chunk.metadata
    assert stats["admission"] is None and threshold is None
