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

from benchmarks.sim.env_process import EnvProcessError, ProcessEnv
from benchmarks.sim.envs import EnvSpec, MockBenchEnv, env_factory, quat2axisangle
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
from vla_runtime.models._lerobot import parse_camera_map
from vla_runtime.models.prefix_kv import batch_guidance, euler_denoise, stack_prefix_kv
from vla_runtime.models.smolvla import _batch_first, _heads_first, _kv_layout
from vla_runtime.action_head.base import ActionHeadContext
from vla_runtime.action_head.rtc import RTCConfig, prepare_guidance
from vla_runtime.vlm.base import Embedding, LayerKV


def _mock_env(**kwargs) -> MockBenchEnv:
    defaults = dict(num_tasks=2, steps_to_done=12, success_probability=0.5, control_hz=100.0)
    defaults.update(kwargs)
    return MockBenchEnv(**defaults)


# Same env as _mock_env, through the picklable factory a child process can build.
_MOCK_OPTS = {"num_tasks": "2", "steps_to_done": "12", "success_probability": "0.5",
              "control_hz": "100.0"}


def _run(mode: str, policy_opts: dict[str, str] | None = None, **config_kwargs):
    env = _mock_env()
    bench = make_policy("mock", env.spec, {"chunk_size": "16", **(policy_opts or {})})
    config = RunConfig(mode=mode, episodes_per_task=2, warmup=1, **config_kwargs)
    return asyncio.run(run_benchmark(bench, env_factory("mock", _MOCK_OPTS), config))


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

    def test_realtime_thread_envs_match_process_envs(self):
        threaded, _ = _run("realtime", num_robots=2, max_steps=20, env_process=False)
        isolated, _ = _run("realtime", num_robots=2, max_steps=20)

        def by_episode(records):
            return [r.success for r in sorted(records, key=lambda r: (r.task, r.episode))]

        assert by_episode(threaded) == by_episode(isolated)

    def test_summary_fields(self):
        records, wall_s = _run("sync")
        summary = summarize(records, wall_s, extra={"run": "sync"})
        assert summary["episodes"] == 4
        assert summary["latency_p50_ms"] is not None
        assert 0.0 <= summary["success_rate"] <= 1.0


# ── Env process ───────────────────────────────────────────────────────────────

class TestProcessEnv:
    def test_round_trip_matches_in_process_env(self):
        local = _mock_env()
        remote = ProcessEnv(env_factory("mock", _MOCK_OPTS))
        try:
            assert remote.spec == local.spec
            assert remote.task_names() == local.task_names()
            task = local.task_names()[0]
            assert remote.max_steps(task) == local.max_steps(task)
            pairs = [(remote.reset(task, 0, 7), local.reset(task, 0, 7))]
            action = torch.full((7,), 0.5)
            for _ in range(3):
                r_obs, r_rew, r_done, _ = remote.step(action)
                l_obs, l_rew, l_done, _ = local.step(action)
                assert (r_rew, r_done) == (l_rew, l_done)
                pairs.append((r_obs, l_obs))
            for r_obs, l_obs in pairs:
                for name, image in l_obs["images"].items():
                    assert torch.equal(r_obs["images"][name], image)  # uint8 transport is exact
                assert torch.equal(
                    r_obs["robot_state"].extra["policy_state"],
                    l_obs["robot_state"].extra["policy_state"],
                )
        finally:
            remote.close()

    def test_call_errors_surface_in_parent(self):
        remote = ProcessEnv(env_factory("mock", _MOCK_OPTS))
        try:
            remote.reset(remote.task_names()[0], 0, 7)
            with pytest.raises(EnvProcessError, match="step"):
                remote.step(None)  # type: ignore[arg-type]
            assert remote.max_steps(remote.task_names()[0]) > 0  # the child survives a failed call
        finally:
            remote.close()

    def test_build_failure_surfaces_in_parent(self):
        with pytest.raises(EnvProcessError, match="start"):
            ProcessEnv(env_factory("mock", {"no_such_option": "1"}))


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

    def test_realtime_env_flag(self):
        args = parse_args(["--env", "mock", "--policy", "mock", "--modes", "realtime",
                           "--realtime-env", "thread"])
        (config,) = build_configs(args)
        assert not config.env_process and config.label.endswith("/thread-env")

    def test_head_batch_flag(self):
        args = parse_args(["--env", "mock", "--policy", "mock", "--modes", "realtime",
                           "--head-batch", "1"])
        (config,) = build_configs(args)
        assert config.head_batch == 1 and config.label.endswith("/hb1")
        (default,) = build_configs(parse_args(["--env", "mock", "--policy", "mock",
                                               "--modes", "realtime"]))
        assert "hb" not in default.label

    def test_realtime_records_head_batch(self):
        records, wall_s = _run("realtime", num_robots=2, max_steps=20)
        assert all(r.head_batch and set(r.head_batch) == {1} for r in records)  # mock head: no batching
        assert summarize(records, wall_s)["head_batch_mean"] == 1.0

    def test_adaptive_threshold_flags(self):
        args = parse_args(["--env", "mock", "--policy", "mock", "--modes", "realtime",
                           "--adaptive-target-ms", "400", "--relaxed-threshold", "0.4"])
        (config,) = build_configs(args)
        assert config.adaptive_target_ms == 400.0 and config.relaxed_threshold == 0.4
        assert config.label.endswith("/adaptive400ms")

    def test_realtime_follows_the_fleet_advised_threshold(self):
        # Any measured round exceeds a 1 us target: once past the dropped first samples
        # the fleet advises 0.5 and keeps it.
        records, wall_s = _run("realtime", num_robots=2, max_steps=20, chunk_threshold=0.9,
                               adaptive_target_ms=0.001)
        for r in records:
            assert r.replan_threshold and r.replan_threshold[-1] == 0.5
            assert set(r.replan_threshold) <= {0.9, 0.5}
        assert summarize(records, wall_s)["replan_threshold_mean"] < 0.9
        fixed, _ = _run("realtime", num_robots=1, max_steps=20)
        assert all(not r.replan_threshold for r in fixed)

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


# ── Head batching helpers (ADR-022, no LeRobot needed) ────────────────────────

def _kv_condition(length: int, seed: int, layers: int = 2, heads: int = 2, dim: int = 4) -> Embedding:
    g = torch.Generator().manual_seed(seed)
    return Embedding(
        kind="per_layer_kv",
        layer_kv=tuple(
            LayerKV(key=torch.randn(heads, length, dim, generator=g),
                    value=torch.randn(heads, length, dim, generator=g))
            for _ in range(layers)
        ),
        attention_mask=torch.ones(length, dtype=torch.bool),
        model_id="mock",
    )


class TestHeadBatchingHelpers:
    def test_stack_prefix_kv_right_pads_and_masks(self):
        short, long = _kv_condition(3, 0), _kv_condition(5, 1)
        layers, masks = stack_prefix_kv([short, long])
        assert masks.tolist() == [[True] * 3 + [False] * 2, [True] * 5]
        key, value = layers[1]
        assert key.shape == value.shape == (2, 2, 5, 4)
        assert torch.equal(key[0, :, :3], short.layer_kv[1].key)
        assert torch.equal(key[1], long.layer_kv[1].key)
        assert torch.all(key[0, :, 3:] == 0)

    def test_stack_prefix_kv_single_row_is_a_view(self):
        cond = _kv_condition(3, 0)
        layers, masks = stack_prefix_kv([cond])
        assert layers[0][0].data_ptr() == cond.layer_kv[0].key.data_ptr()
        assert stack_prefix_kv([cond], clone=True)[0][0][0].data_ptr() != cond.layer_kv[0].key.data_ptr()
        assert masks.shape == (1, 3)

    def test_stack_prefix_kv_rejects_mismatched_rows(self):
        with pytest.raises(ValueError, match="disagree"):
            stack_prefix_kv([_kv_condition(3, 0), _kv_condition(3, 1, heads=3)])

    def test_batch_guidance_leaves_unguided_rows_at_zero_weight(self):
        context = ActionHeadContext(prev_actions=torch.ones(6, 3), inference_delay_steps=2)
        guided = prepare_guidance(context, 8, 3, RTCConfig())
        target, guidance = batch_guidance(
            [None, guided], lambda a: a * 2, (2, 8, 5), 3, torch.device("cpu")
        )
        assert target.shape == (2, 8, 5) and guidance.weights.shape == (2, 8)
        assert torch.all(target[0] == 0) and torch.all(guidance.weights[0] == 0)
        assert torch.equal(target[1, :, :3], guided.target * 2)
        assert torch.equal(guidance.weights[1], guided.weights)
        assert batch_guidance([None, None], lambda a: a, (2, 8, 5), 3, torch.device("cpu")) == (None, None)

    def test_batch_guidance_rejects_mixed_settings(self):
        context = ActionHeadContext(prev_actions=torch.ones(6, 3), inference_delay_steps=2)
        rows = [prepare_guidance(context, 8, 3, RTCConfig(max_guidance_weight=b)) for b in (5.0, 1.0)]
        with pytest.raises(ValueError, match="disagree"):
            batch_guidance(rows, lambda a: a, (2, 8, 3), 3, torch.device("cpu"))

    @pytest.mark.parametrize("jacobian", ["identity", "vjp"])
    def test_batched_euler_matches_rows(self, jacobian):
        # Row-independent toy field: any batching error shows up as a row mismatch.
        weight = torch.randn(5, 5, generator=torch.Generator().manual_seed(0))

        def velocity(x, t):
            return torch.tanh(x @ weight) * t.view(-1, 1, 1)

        noise = torch.randn(2, 8, 5, generator=torch.Generator().manual_seed(1))
        context = ActionHeadContext(prev_actions=torch.ones(6, 3), inference_delay_steps=2)
        rows = [None, prepare_guidance(context, 8, 3, RTCConfig(jacobian=jacobian))]
        target, guidance = batch_guidance(rows, lambda a: a, (2, 8, 5), 3, torch.device("cpu"))
        batched = euler_denoise(velocity, noise, 10, target, guidance)
        for row, g in enumerate(rows):
            t1, g1 = batch_guidance([g], lambda a: a, (1, 8, 5), 3, torch.device("cpu"))
            alone = euler_denoise(velocity, noise[row : row + 1], 10, t1, g1)
            torch.testing.assert_close(batched[row : row + 1], alone)
