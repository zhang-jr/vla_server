"""Closed-loop sim checks on real simulators and models — remote GPU host only.

What must hold before any LIBERO / ManiSkill number goes into BENCHMARKS.md:

  1. The SmolVLA split (encode → per_layer_kv → expert denoise) reproduces
     LeRobot's own predict_action_chunk with shared noise. Otherwise a
     native-vs-runtime success gap would be an adapter bug, not a runtime effect.
     Same for the π0.5 split (PaliGemma prefix → Gemma expert).
  2. The runtime's RTC guidance on SmolVLA and π0.5 (``identity`` Jacobian) reproduces
     LeRobot's built-in RTCProcessor with shared noise; the ``vjp`` mode runs
     and its head latency is recorded.
  3. Each simulator wrapper produces the observation format the policies read.
  4. One short episode per mode runs end to end on LIBERO.

Every test skips when its simulator / LeRobot is not installed, and the
conftest skips ``gpu`` tests where no CUDA device is visible. Run::

    python scripts/remote.py test benchmarks/benchmark_sim.py -m gpu -v

The SmolVLA checkpoint comes from the environment, so offline hosts can
point at local copies::

    VLA_SMOLVLA_REPO=/models/smolvla_libero
    VLA_SMOLVLA_VLM=/models/SmolVLM2-500M-Video-Instruct
    VLA_SMOLVLA_CAMERAS=agentview=observation.images.camera1,wrist=observation.images.camera2
    VLA_PI05_REPO=/models/pi05_libero_finetuned_v044
    VLA_PI05_TOKENIZER=/models/paligemma-3b-pt-224

Full success-rate runs are not tests; use ``python -m benchmarks.sim.run``.
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.fixture(scope="module")
def smolvla_model():
    pytest.importorskip("lerobot")
    from benchmarks.sim.lerobot_smolvla import SmolVLAModel

    repo, cameras, vlm = _smolvla_source()
    return SmolVLAModel(repo, torch.device("cuda"), cameras, vlm=vlm)


def _smolvla_source() -> tuple[str, dict[str, str], str | None]:
    from benchmarks.sim.lerobot_smolvla import DEFAULT_REPO, LIBERO_CAMERAS, parse_camera_map

    cameras = os.environ.get("VLA_SMOLVLA_CAMERAS")
    return (
        os.environ.get("VLA_SMOLVLA_REPO", DEFAULT_REPO),
        parse_camera_map(cameras) if cameras else dict(LIBERO_CAMERAS),
        os.environ.get("VLA_SMOLVLA_VLM") or None,
    )


@pytest.fixture(scope="module")
def pi05_model():
    pytest.importorskip("lerobot")
    from benchmarks.sim.lerobot_pi05 import DEFAULT_REPO, PI05Model
    from benchmarks.sim.lerobot_smolvla import LIBERO_CAMERAS

    model = PI05Model(
        os.environ.get("VLA_PI05_REPO", DEFAULT_REPO),
        torch.device("cuda"),
        LIBERO_CAMERAS,
        tokenizer=os.environ.get("VLA_PI05_TOKENIZER") or None,
    )
    yield model
    model.policy.to("cpu")
    torch.cuda.empty_cache()


def _smolvla_policy_opts() -> dict[str, str]:
    repo, cameras, vlm = _smolvla_source()
    opts = {"repo": repo, "cameras": ",".join(f"{k}={v}" for k, v in cameras.items())}
    if vlm:
        opts["vlm"] = vlm
    return opts


def _synthetic_obs(resolution: int = 256) -> tuple[dict[str, torch.Tensor], str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    images = {
        "agentview": torch.rand(3, resolution, resolution, generator=generator),
        "wrist": torch.rand(3, resolution, resolution, generator=generator),
    }
    state = torch.randn(8, generator=generator) * 0.1
    return images, "put the bowl on the plate", state


def _check_split_matches_native(model, atol: float) -> None:
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
    torch.testing.assert_close(split, native, atol=atol, rtol=atol)


def test_smolvla_split_matches_native(smolvla_model):
    _check_split_matches_native(smolvla_model, atol=1e-3)


def test_pi05_split_matches_native(pi05_model):
    _check_split_matches_native(pi05_model, atol=1e-3)


def _rtc_inputs(model, delay: int = 5, start: int = 10, overlap: int = 30):
    """Shared observation, noise and a plausible previous tail (env space)."""
    images, text, state = _synthetic_obs()
    config = model.config
    noise = torch.randn(
        1, config.chunk_size, config.max_action_dim,
        generator=torch.Generator().manual_seed(1),
    ).to(model.device)
    other = torch.randn(
        1, config.chunk_size, config.max_action_dim,
        generator=torch.Generator().manual_seed(2),
    ).to(model.device)
    embedding = model.encode_prefix(images, text, state)
    prev = model.denoise(embedding, noise=other)[start:start + overlap]
    return images, text, state, noise, embedding, prev, delay


def _check_rtc_matches_lerobot(model) -> None:
    from lerobot.configs.types import RTCAttentionSchedule
    from lerobot.policies.rtc.configuration_rtc import RTCConfig as LeRobotRTCConfig

    from vla_runtime.action_head.base import ActionHeadContext
    from vla_runtime.action_head.rtc import RTCConfig, prepare_guidance

    images, text, state, noise, embedding, prev, delay = _rtc_inputs(model)
    overlap = prev.shape[0]

    guidance = prepare_guidance(
        ActionHeadContext(prev_actions=prev, inference_delay_steps=delay),
        model.config.chunk_size,
        model.action_dim,
        RTCConfig(max_guidance_weight=5.0, prefix_schedule="linear", jacobian="identity"),
    )
    ours = model.denoise(embedding, noise=noise.clone(), guidance=guidance)
    unguided = model.denoise(embedding, noise=noise.clone())

    policy = model.policy
    saved = policy.config.rtc_config
    policy.config.rtc_config = LeRobotRTCConfig(
        enabled=True,
        prefix_attention_schedule=RTCAttentionSchedule.LINEAR,
        max_guidance_weight=5.0,
        execution_horizon=overlap,
    )
    policy.init_rtc_processor()
    try:
        batch = model.preprocess(images, text, state)
        reference = model._postprocess(
            policy.predict_action_chunk(
                batch,
                noise=noise.clone(),
                inference_delay=delay,
                prev_chunk_left_over=model.normalize_actions(prev).unsqueeze(0),
                execution_horizon=overlap,
            )
        )
    finally:
        policy.config.rtc_config = saved
        policy.init_rtc_processor()

    torch.testing.assert_close(ours, reference, atol=2e-3, rtol=2e-3)
    # Guidance did something, and it pulled the frozen steps toward the tail.
    assert not torch.allclose(ours, unguided, atol=1e-3)
    frozen_ours = (ours[:delay] - prev[:delay]).abs().mean()
    frozen_plain = (unguided[:delay] - prev[:delay]).abs().mean()
    assert frozen_ours < frozen_plain


def test_smolvla_rtc_matches_lerobot_rtc_processor(smolvla_model):
    _check_rtc_matches_lerobot(smolvla_model)


def test_pi05_rtc_matches_lerobot_rtc_processor(pi05_model):
    _check_rtc_matches_lerobot(pi05_model)


def _rtc_head_latency(model, record_property, label: str) -> None:
    """Head cost per mode: unguided, identity (no backward), vjp (one backward per step)."""
    from vla_runtime.action_head.base import ActionHeadContext
    from vla_runtime.action_head.rtc import RTCConfig, prepare_guidance

    _, _, _, noise, embedding, prev, delay = _rtc_inputs(model)
    context = ActionHeadContext(prev_actions=prev, inference_delay_steps=delay)

    def timed(guidance) -> float:
        for _ in range(2):
            model.denoise(embedding, noise=noise.clone(), guidance=guidance)
        torch.cuda.synchronize()
        samples = []
        for _ in range(10):
            t0 = time.perf_counter()
            out = model.denoise(embedding, noise=noise.clone(), guidance=guidance)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)
            assert torch.isfinite(out).all()
        return sorted(samples)[len(samples) // 2]

    chunk, dim = model.config.chunk_size, model.action_dim
    results = {
        "unguided": timed(None),
        "identity": timed(prepare_guidance(context, chunk, dim, RTCConfig(jacobian="identity"))),
        "vjp": timed(prepare_guidance(context, chunk, dim, RTCConfig(jacobian="vjp"))),
    }
    for mode, ms in results.items():
        record_property(f"head_p50_ms_{mode}", round(ms, 2))
    print(f"\n{label} expert P50 (ms): {results}")


def test_smolvla_rtc_head_latency(smolvla_model, record_property):
    _rtc_head_latency(smolvla_model, record_property, "SmolVLA")


def test_pi05_rtc_head_latency(pi05_model, record_property):
    _rtc_head_latency(pi05_model, record_property, "pi05")


def test_smolvla_condition_size(smolvla_model, record_property):
    _record_condition_size(smolvla_model, record_property)


def test_pi05_condition_size(pi05_model, record_property):
    _record_condition_size(pi05_model, record_property)


def _record_condition_size(model, record_property) -> None:
    """Bytes crossing the VLM → head boundary per frame (ADR-005 split-point data)."""
    images, text, state = _synthetic_obs()
    embedding = model.encode_prefix(images, text, state)
    assert embedding.nbytes > 0
    # Recorded (junit property), not asserted against a target.
    record_property("condition_nbytes", embedding.nbytes)
    record_property("condition_layers", embedding.num_layers)
    record_property("prefix_len", embedding.seq_len)
    print(
        f"\n{embedding.model_id}: condition {embedding.nbytes / 1e6:.2f} MB, "
        f"{embedding.num_layers} layers, prefix {embedding.seq_len}"
    )


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
    bench = make_policy("smolvla", spec, _smolvla_policy_opts())
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
