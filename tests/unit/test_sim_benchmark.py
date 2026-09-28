"""CPU tests for the closed-loop sim benchmark harness (benchmarks/sim).

Mock env + mock policy only: these check the runners, the simulated clock and
the metric definitions. Real simulators and models are exercised by the
gpu-marked tests in benchmarks/benchmark_sim.py on the remote host.
"""

from __future__ import annotations

import asyncio
import math
import random

import numpy as np
import pytest
import torch

from benchmarks.sim.envs import EnvSpec, MockBenchEnv, env_factory, quat2axisangle
from benchmarks.sim.lerobot_smolvla import _batch_first, _heads_first, _kv_layout, parse_camera_map
from benchmarks.sim.metrics import (
    boundary_discontinuity,
    jerk_rms,
    markdown_table,
    percentile,
    summarize,
    wilson_interval,
)
from benchmarks.sim.policies import make_policy
from benchmarks.sim.run import build_configs, parse_args
from benchmarks.sim.runners import RunConfig, episode_plan, run_benchmark


def _mock_env(**kwargs) -> MockBenchEnv:
    defaults = dict(num_tasks=2, steps_to_done=12, success_probability=0.5, control_hz=100.0)
    defaults.update(kwargs)
    return MockBenchEnv(**defaults)


def _run(mode: str, policy_opts: dict[str, str] | None = None, **config_kwargs):
    env = _mock_env()
    bench = make_policy("mock", env.spec, {"chunk_size": "16", **(policy_opts or {})})
    config = RunConfig(mode=mode, episodes_per_task=2, warmup=1, **config_kwargs)
    return asyncio.run(run_benchmark(bench, _mock_env, config))


# ── Metrics ───────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_jerk_zero_for_constant_velocity(self):
        # Constant delta command → linear trajectory → zero third difference.
        actions = torch.ones(20, 7)
        assert jerk_rms(actions, action_is_delta=True) == pytest.approx(0.0)

    def test_jerk_ignores_gripper(self):
        actions = torch.zeros(20, 7)
        actions[::2, -1] = 1.0  # gripper chatter only
        assert jerk_rms(actions, action_is_delta=False) == pytest.approx(0.0)

    def test_jerk_needs_four_steps(self):
        assert jerk_rms(torch.zeros(3, 7), action_is_delta=True) is None

    def test_boundary_discontinuity_detects_jump(self):
        actions = torch.zeros(10, 3)
        actions[:, 0] = torch.arange(10, dtype=torch.float32) * 0.1
        actions[5:, 0] += 1.0  # a chunk switch at step 5
        jump, ratio = boundary_discontinuity(actions, [0, 5], gripper_index=-1)
        assert jump == pytest.approx(1.1, abs=1e-5)
        assert ratio == pytest.approx(11.0, rel=1e-4)

    def test_boundary_smooth_ratio_is_one(self):
        actions = torch.arange(10, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
        _, ratio = boundary_discontinuity(actions, [4], gripper_index=-1)
        assert ratio == pytest.approx(1.0)

    def test_percentile_interpolates(self):
        assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)
        assert percentile([], 50) is None

    def test_wilson_interval_brackets_rate(self):
        lo, hi = wilson_interval(8, 10)
        assert lo < 0.8 < hi
        assert wilson_interval(0, 0) == (0.0, 0.0)

    def test_markdown_table_formats_rows(self):
        table = markdown_table([{"run": "sync", "success_rate": 0.5, "success_ci95": [0.2, 0.8]}])
        assert "| sync | 50.0% | [0.20, 0.80] |" in table


# ── Envs ──────────────────────────────────────────────────────────────────────

class TestEnvs:
    def test_quat2axisangle_matches_rotation(self):
        # 90° about z: q = (0, 0, sin45, cos45).
        s = math.sqrt(0.5)
        np.testing.assert_allclose(quat2axisangle(np.array([0.0, 0.0, s, s])), [0, 0, math.pi / 2], atol=1e-9)
        np.testing.assert_allclose(quat2axisangle(np.array([0.0, 0.0, 0.0, 1.0])), [0, 0, 0])

    def test_mock_obs_matches_spec(self):
        env = _mock_env()
        obs = env.reset(env.task_names()[0], 0, 1)
        assert tuple(obs["images"]) == env.spec.camera_names
        assert obs["robot_state"].flat_dim == env.spec.state_dim
        assert obs["robot_state"].extra["policy_state"].numel() == env.spec.policy_state_dim

    def test_hold_action_delta_keeps_gripper(self):
        env = _mock_env()
        last = torch.arange(7, dtype=torch.float32)
        hold = env.hold_action(last)
        assert torch.all(hold[:-1] == 0)
        assert hold[-1] == last[-1]

    def test_hold_action_absolute_repeats_last(self):
        env = _mock_env()
        env.spec = EnvSpec(**{**env.spec.__dict__, "action_is_delta": False})
        last = torch.arange(7, dtype=torch.float32)
        assert torch.equal(env.hold_action(last), last)

    def test_env_factory_rejects_unknown(self):
        with pytest.raises(ValueError):
            env_factory("gazebo", {})


# ── Runners ───────────────────────────────────────────────────────────────────

class TestRunners:
    def test_episode_plan_matches_validator_enumeration(self):
        env = _mock_env()
        config = RunConfig(mode="sync", episodes_per_task=3, seed=42)
        rng = random.Random(42)
        expected = [
            (task, ep, rng.randint(0, 2**31)) for task in env.task_names() for ep in range(3)
        ]
        assert episode_plan(env, config) == expected

    @pytest.mark.parametrize("mode", ["native", "sync", "async_sim"])
    def test_modes_agree_on_mock_success(self, mode):
        # Mock success depends only on the reset seed, so every mode must match.
        records, _ = _run(mode)
        reference, _ = _run("native")
        assert [r.success for r in records] == [r.success for r in reference]
        assert all(r.inference_ms for r in records)

    def test_sync_separates_vlm_and_head(self):
        records, _ = _run("sync", {"latency_ms": "20"})
        vlm = [v for r in records for v in r.vlm_ms]
        assert vlm and min(vlm) >= 19.0
        assert all(r.head_ms for r in records)

    def test_async_sim_no_underrun_when_latency_fits(self):
        # 16-step chunks, refill at 8 queued; 50 ms at 100 Hz lands in 5 steps.
        records, _ = _run("async_sim", latency_ms=50.0)
        assert sum(r.underrun_steps for r in records) == 0

    def test_async_sim_underruns_when_latency_exceeds_queue(self):
        # 200 ms at 100 Hz = 20 steps in flight, only 8 queued → underruns.
        records, _ = _run("async_sim", latency_ms=200.0)
        assert sum(r.underrun_steps for r in records) > 0
        assert all(lat == 200.0 for r in records for lat in r.effective_latency_ms[1:])

    def test_realtime_fleet_runs_every_episode_once(self):
        records, _ = _run("realtime", num_robots=2, max_steps=20)
        assert len(records) == 4
        assert {r.robot_id for r in records} == {"robot_0", "robot_1"}
        assert sorted((r.task, r.episode) for r in records) == sorted(
            (t, e) for t in _mock_env().task_names() for e in range(2)
        )

    def test_summary_fields(self):
        records, wall_s = _run("sync")
        summary = summarize(records, wall_s, extra={"run": "sync"})
        assert summary["episodes"] == 4
        assert summary["latency_p50_ms"] is not None
        assert 0.0 <= summary["success_rate"] <= 1.0


# ── CLI ───────────────────────────────────────────────────────────────────────

class TestCli:
    def test_build_configs_expands_sweeps(self):
        args = parse_args([
            "--env", "mock", "--policy", "mock",
            "--modes", "native,async_sim,realtime",
            "--blend-policies", "replace,ensemble",
            "--inject-latency-ms", "0,500",
            "--robots", "1,4",
        ])
        labels = [c.label for c in build_configs(args)]
        assert labels[0] == "native"
        assert "async_sim/ensemble/lat500ms" in labels
        assert "realtime/replace/4robot" in labels
        assert len(labels) == 1 + 2 * 2 + 2 * 2

    def test_unknown_mode_rejected(self):
        args = parse_args(["--env", "mock", "--policy", "mock", "--modes", "turbo"])
        with pytest.raises(SystemExit):
            build_configs(args)


# ── SmolVLA adapter helpers (no LeRobot needed) ───────────────────────────────

class TestSmolVLAHelpers:
    @pytest.mark.parametrize("shape,layout", [((1, 5, 2, 4), "BLHD"), ((1, 2, 5, 4), "BHLD")])
    def test_kv_layout_roundtrip(self, shape, layout):
        t = torch.randn(*shape)
        assert _kv_layout(t, prefix_len=5) == layout
        heads_first = _heads_first(t, layout)
        assert heads_first.shape == (2, 5, 4)
        assert torch.equal(_batch_first(heads_first, layout), t)

    def test_kv_layout_rejects_mismatch(self):
        with pytest.raises(ValueError):
            _kv_layout(torch.randn(1, 3, 2, 4), prefix_len=5)

    def test_parse_camera_map(self):
        mapping = parse_camera_map("agentview=observation.images.image,wrist=observation.images.image2")
        assert mapping == {
            "agentview": "observation.images.image",
            "wrist": "observation.images.image2",
        }
        with pytest.raises(ValueError):
            parse_camera_map("agentview")
