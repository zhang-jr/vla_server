"""Closed-loop sim checks on real simulators and models — remote GPU host only.

What must hold before any LIBERO / ManiSkill number goes into BENCHMARKS.md:

  1. The SmolVLA split (encode → per_layer_kv → expert denoise) reproduces
     LeRobot's own predict_action_chunk with shared noise. Otherwise a
     native-vs-runtime success gap would be an adapter bug, not a runtime effect.
     Same for the π0.5 split (PaliGemma prefix → Gemma expert).
  2. The runtime's RTC guidance on SmolVLA and π0.5 (``identity`` Jacobian) reproduces
     LeRobot's built-in RTCProcessor with shared noise; the ``vjp`` mode runs
     and its head latency is recorded.
  3. A batched expert pass over several robots' conditions (ADR-022, fleet head
     batching) matches each row denoised alone with the same noise, including
     RTC rows mixed with unguided ones and right-padded prefixes.
     The π0.5 expert replayed from CUDA Graphs (V4) equals the eager loop bit
     for bit, and its capturable step equals LeRobot's ``denoise_step``. The
     expert's sdpa attention stays within bf16 noise of eager, and leaves
     batches below its threshold untouched.
  4. Each simulator wrapper produces the observation format the policies read.
  5. One short episode per mode runs end to end on LIBERO.

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
        )[0]
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


def _check_batched_head_matches_rows(model) -> None:
    """ADR-022: one batched expert pass == every row alone, same noise.

    Batching changes the GEMM shapes, so under bf16 the kernels — and the
    rounding — differ from the one-row pass: π0.5 rows move by up to ~1e-2
    (mean ~3e-4) on actions of scale ~1, against ~0.9 for a different noise
    sample. That is bounded by a few bf16 ulps (eps 7.8e-3), and closed-loop
    success is the acceptance (CLAUDE.md principle 3). Everything batching
    could get *wrong* is checked exactly within one pass: rows never leak
    into each other, and a padded prefix decodes like the unpadded one.
    """
    from vla_runtime.action_head.base import ActionHeadContext
    from vla_runtime.action_head.rtc import RTCConfig, prepare_guidance
    from vla_runtime.vlm.base import Embedding, LayerKV

    images, text, state = _synthetic_obs()
    config = model.config
    first = model.encode_prefix(images, text, state)
    second = model.encode_prefix(
        {k: v.flip(-1) for k, v in images.items()}, "open the top drawer", state + 0.05
    )
    # The first condition with 7 extra masked positions of garbage: it must
    # decode exactly like the first, and it forces the others to be padded.
    pad = 7
    generator = torch.Generator(device=model.device).manual_seed(4)

    def garbage(t: torch.Tensor) -> torch.Tensor:
        extra = torch.randn(
            t.shape[0], pad, t.shape[2], generator=generator, device=t.device, dtype=t.dtype
        )
        return torch.cat([t, extra], dim=1)

    padded = Embedding(
        kind="per_layer_kv",
        layer_kv=tuple(LayerKV(key=garbage(kv.key), value=garbage(kv.value)) for kv in first.layer_kv),
        attention_mask=torch.cat([first.attention_mask, first.attention_mask.new_zeros(pad)]),
        model_id=first.model_id,
        metadata=first.metadata,
    )
    rows = [first, second, padded, first]

    other = torch.randn(
        1, config.chunk_size, config.max_action_dim, generator=torch.Generator().manual_seed(2)
    ).to(model.device)
    prev = model.denoise(second, noise=other)[10:40]
    guided = prepare_guidance(
        ActionHeadContext(prev_actions=prev, inference_delay_steps=5),
        config.chunk_size,
        model.action_dim,
        RTCConfig(),
    )
    guidances = [None, guided, None, None]
    noise = torch.randn(
        len(rows), config.chunk_size, config.max_action_dim,
        generator=torch.Generator().manual_seed(1),
    ).to(model.device)
    noise[2] = noise[0]
    noise[3] = noise[0]

    batched = model.denoise_batch(rows, noise=noise.clone(), guidances=guidances)
    assert len(batched) == len(rows)
    # Same pass, same kernels: a repeated row is bit-identical (no leakage
    # between rows), and masked padding changes nothing.
    assert torch.equal(batched[3], batched[0])
    torch.testing.assert_close(batched[2], batched[0], atol=1e-5, rtol=0)
    # Against each row alone, including the RTC row among unguided ones.
    for i, (row, g) in enumerate(zip(rows[:3], guidances[:3])):
        alone = model.denoise(row, noise=noise[i : i + 1].clone(), guidance=g)
        torch.testing.assert_close(batched[i], alone, atol=3e-2, rtol=0)
        assert (batched[i] - alone).abs().mean() < 1e-3
    assert not torch.allclose(batched[0], batched[1], atol=1e-1)


def test_smolvla_batched_head_matches_rows(smolvla_model):
    _check_batched_head_matches_rows(smolvla_model)


def test_pi05_batched_head_matches_rows(pi05_model):
    _check_batched_head_matches_rows(pi05_model)


def _pi05_rows(model) -> tuple[list, torch.Tensor]:
    """Three rows over two conditions, and their noise."""
    images, text, state = _synthetic_obs()
    first = model.encode_prefix(images, text, state)
    second = model.encode_prefix(
        {k: v.flip(-1) for k, v in images.items()}, "open the top drawer", state + 0.05
    )
    config = model.config
    noise = torch.randn(
        3, config.chunk_size, config.max_action_dim, generator=torch.Generator().manual_seed(1)
    ).to(model.device)
    return [first, second, first], noise


def test_pi05_velocity_matches_denoise_step(pi05_model):
    """The capturable ``_velocity`` is LeRobot's ``denoise_step``, bit for bit."""
    from transformers import DynamicCache

    model = pi05_model
    rows, noise = _pi05_rows(model)
    with torch.inference_mode():
        layers, pad_masks = model._prefix_layers(rows, clone=False)
        cache = DynamicCache(tuple((key, value, None) for key, value in layers))
        t = torch.full((len(rows),), 0.7, device=model.device)
        ours = model._velocity(pad_masks, layers, noise, t)
        ref = model.policy.model.denoise_step(pad_masks, cache, noise, t)
    assert torch.equal(ours, ref)


def test_pi05_cuda_graph_matches_eager(pi05_model):
    """V4: graph replay == eager, bit for bit, on fresh inputs and per batch size."""
    from benchmarks.sim.cuda_graph import GraphCache

    model = pi05_model
    rows, noise = _pi05_rows(model)
    swapped = [rows[1], rows[0], rows[1]]
    model._graphs = GraphCache(model._graph_loop)
    try:
        graphed = model.denoise_batch(rows, noise=noise.clone())
        # Same shapes, new data: the captured graph must read its inputs.
        replayed = model.denoise_batch(swapped, noise=noise.flip(0).clone())
        single = model.denoise(rows[0], noise=noise[:1].clone())
        num_graphs = model._graphs.num_graphs
    finally:
        model._graphs = None
    assert num_graphs == 2
    for got, want in (
        (graphed, model.denoise_batch(rows, noise=noise.clone())),
        (replayed, model.denoise_batch(swapped, noise=noise.flip(0).clone())),
        ([single], [model.denoise(rows[0], noise=noise[:1].clone())]),
    ):
        for g, w in zip(got, want, strict=True):
            assert torch.equal(g, w)


def test_pi05_expert_sdpa_close_to_eager(pi05_model):
    """sdpa from batch 2: batch 1 stays bit-identical, batch 3 within bf16 noise."""
    model = pi05_model
    rows, noise = _pi05_rows(model)
    eager = model.denoise_batch(rows, noise=noise.clone())
    eager_single = model.denoise(rows[0], noise=noise[:1].clone())
    model.expert_sdpa_min_batch = 2
    try:
        sdpa = model.denoise_batch(rows, noise=noise.clone())
        sdpa_single = model.denoise(rows[0], noise=noise[:1].clone())
    finally:
        model.expert_sdpa_min_batch = None
    assert torch.equal(sdpa_single, eager_single)
    for s, e in zip(sdpa, eager, strict=True):
        assert (s - e).abs().max() < 5e-2
        assert not torch.equal(s, e)


def test_pi05_cuda_graph_head_latency(pi05_model, record_property):
    """V4: expert loop P50, eager vs CUDA Graph, per batch size."""
    from benchmarks.sim.cuda_graph import GraphCache

    model = pi05_model
    rows, _ = _pi05_rows(model)
    config = model.config

    def timed(batch: int) -> float:
        conditions = [rows[0]] * batch
        noise = torch.randn(batch, config.chunk_size, config.max_action_dim, device=model.device)
        for _ in range(3):
            model.denoise_batch(conditions, noise=noise)
        samples = []
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.denoise_batch(conditions, noise=noise)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)
        return sorted(samples)[len(samples) // 2]

    results = {}
    for batch in (1, 2, 4, 8):
        results[f"eager_b{batch}"] = timed(batch)
    model._graphs = GraphCache(model._graph_loop)
    try:
        for batch in (1, 2, 4, 8):
            results[f"graph_b{batch}"] = timed(batch)
        model._graphs = GraphCache(model._graph_loop)
        model.expert_sdpa_min_batch = 2
        for batch in (2, 4, 8):
            results[f"graph_sdpa_b{batch}"] = timed(batch)
    finally:
        model._graphs = None
        model.expert_sdpa_min_batch = None
    for key, ms in results.items():
        record_property(f"head_p50_ms_{key}", round(ms, 2))
    print(f"\npi05 expert P50 (ms): {results}")
    assert results["graph_b1"] < 0.7 * results["eager_b1"]


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
