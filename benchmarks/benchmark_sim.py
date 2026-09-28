"""Closed-loop sim checks on real simulators and models — remote GPU host only.

What must hold before any LIBERO / ManiSkill number goes into BENCHMARKS.md:

  1. The SmolVLA split (encode → per_layer_kv → expert denoise) reproduces
     LeRobot's own predict_action_chunk with shared noise. Otherwise a
     native-vs-runtime success gap would be an adapter bug, not a runtime effect.
  2. Each simulator wrapper produces the observation format the policies read.
  3. One short episode per mode runs end to end on LIBERO.

Every test skips when its simulator / LeRobot is not installed, and the
conftest skips ``gpu`` tests where no CUDA device is visible. Run::

    python scripts/remote.py test benchmarks/benchmark_sim.py -m gpu -v

Full success-rate runs are not tests; use ``python -m benchmarks.sim.run``.
"""

from __future__ import annotations

import asyncio

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.fixture(scope="module")
def smolvla_model():
    pytest.importorskip("lerobot")
    from benchmarks.sim.lerobot_smolvla import DEFAULT_REPO, LIBERO_CAMERAS, SmolVLAModel

    return SmolVLAModel(DEFAULT_REPO, torch.device("cuda"), LIBERO_CAMERAS)


def _synthetic_obs(resolution: int = 256) -> tuple[dict[str, torch.Tensor], str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    images = {
        "agentview": torch.rand(3, resolution, resolution, generator=generator),
        "wrist": torch.rand(3, resolution, resolution, generator=generator),
    }
    state = torch.randn(8, generator=generator) * 0.1
    return images, "put the bowl on the plate", state


def test_smolvla_split_matches_native(smolvla_model):
    model = smolvla_model
    images, text, state = _synthetic_obs()
    config = model.config
    noise = torch.randn(
        1, config.chunk_size, config.max_action_dim,
        generator=torch.Generator().manual_seed(1),
    ).to(model.device)

    native = model.native_chunk(images, text, state, noise=noise.clone())
    embedding = model.encode_prefix(images, text, state)
    split = model.denoise(embedding, noise=noise.clone())

    assert embedding.kind == "per_layer_kv"
    assert embedding.seq_len == embedding.attention_mask.shape[0]
    assert split.shape == native.shape == (config.chunk_size, model.action_dim)
    torch.testing.assert_close(split, native, atol=1e-3, rtol=1e-3)


def test_smolvla_condition_size(smolvla_model, record_property):
    """Bytes crossing the VLM → head boundary per frame (ADR-005 split-point data)."""
    images, text, state = _synthetic_obs()
    embedding = smolvla_model.encode_prefix(images, text, state)
    assert embedding.nbytes > 0
    # Recorded (junit property), not asserted against a target.
    record_property("condition_nbytes", embedding.nbytes)
    record_property("condition_layers", embedding.num_layers)
    record_property("prefix_len", embedding.seq_len)


def test_libero_env_observation_format():
    pytest.importorskip("libero")
    from benchmarks.sim.envs import LiberoEnv

    env = LiberoEnv(suite="libero_object", task_ids=[0])
    try:
        task = env.task_names()[0]
        obs = env.reset(task, 0, 7)
        assert set(obs["images"]) == set(env.spec.camera_names)
        assert obs["images"]["agentview"].shape == (3, 256, 256)
        assert obs["robot_state"].flat_dim == env.spec.state_dim
        assert obs["robot_state"].extra["policy_state"].shape == (8,)
        assert obs["instruction"]
        obs, _, done, info = env.step(env.hold_action(None))
        assert "success" in info
    finally:
        env.close()


def test_maniskill_env_observation_format():
    pytest.importorskip("mani_skill")
    from benchmarks.sim.envs import ManiSkillEnv

    env = ManiSkillEnv(env_ids=("PickCube-v1",), resolution=128)
    try:
        assert env.spec.action_dim == 7
        assert env.spec.control_hz > 0
        obs = env.reset("PickCube-v1", 0, 3)
        assert obs["robot_state"].flat_dim == env.spec.state_dim
        _, _, _, info = env.step(env.hold_action(None))
        assert "success" in info
        assert env.max_steps("PickCube-v1") > 0
    finally:
        env.close()


@pytest.fixture(scope="module")
def libero_smolvla():
    pytest.importorskip("libero")
    pytest.importorskip("lerobot")
    from benchmarks.sim.envs import LiberoEnv, env_factory
    from benchmarks.sim.policies import make_policy

    factory = env_factory("libero", {"suite": "libero_object", "task_ids": "0"})
    # Constructing LiberoEnv only reads the suite; scenes open on reset().
    spec = LiberoEnv(suite="libero_object", task_ids=[0]).spec
    bench = make_policy("smolvla", spec, {})
    yield factory, bench
    bench.close()


@pytest.mark.parametrize("mode", ["native", "sync", "async_sim", "realtime"])
def test_libero_smolvla_episode_runs(libero_smolvla, mode):
    from benchmarks.sim.runners import RunConfig, run_benchmark

    factory, bench = libero_smolvla
    config = RunConfig(mode=mode, episodes_per_task=1, num_tasks=1, max_steps=60, warmup=1)
    records, _ = asyncio.run(run_benchmark(bench, factory, config))
    assert len(records) == 1
    record = records[0]
    assert record.steps > 0
    assert record.inference_ms
