"""LeRobot prefix-KV policies (SmolVLA, π0.5) as benchmark policies.

The adapters themselves live in the package (``vla_runtime.models``): the
model-agnostic backbone / expert head in ``prefix_kv``, the models in
``smolvla`` and ``pi05``, option parsing in ``registry``. This module wraps a
loaded model for the sim harness: it checks the model against the
environment, picks the head frequency from the env's control rate, and adds
the model's native inference as the ``native`` mode's reference.

Policy options (``--policy-opt key=value``): see ``vla_runtime.models.registry``
for the model and RTC options; in addition

    n_action_steps   override the checkpoint's value (actions executed per chunk
                     in the blocking modes)
    vlm_budget_ms    backbone latency budget reported to the scheduler (default 2000)
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import torch

from benchmarks.sim.envs import EnvSpec, Observation
from benchmarks.sim.policies import BenchPolicy
from vla_runtime.action_head.rtc import RTCConfig
from vla_runtime.models.prefix_kv import PrefixKVModel, prefix_kv_policy
from vla_runtime.models.registry import load_model, pop_rtc_options


def check_cameras(cameras: Mapping[str, str], spec: EnvSpec) -> None:
    missing = set(cameras) - set(spec.camera_names)
    if missing:
        raise ValueError(
            f"camera map uses {sorted(missing)} but {spec.name} provides "
            f"{list(spec.camera_names)}; set --policy-opt cameras=env_cam=lerobot_key,..."
        )


def wrap_prefix_kv_policy(
    model: PrefixKVModel,
    spec: EnvSpec,
    *,
    vlm_budget_ms: float,
    rtc: bool,
    rtc_config: RTCConfig,
    n_action_steps: int | None,
    info: dict[str, Any],
) -> BenchPolicy:
    """Prefix-KV model → Policy (prefix state, per_layer_kv, iterative) → BenchPolicy."""
    if model.action_dim != spec.action_dim:
        raise ValueError(
            f"{model.repo} outputs {model.action_dim}-D actions but {spec.name} takes "
            f"{spec.action_dim}-D"
        )
    steps = n_action_steps or int(model.config.n_action_steps)
    policy, backbone = prefix_kv_policy(
        model,
        vlm_budget_ms=vlm_budget_ms,
        frequency_hz=spec.control_hz / max(1, steps),
        rtc=rtc,
        rtc_config=rtc_config,
    )

    async def native(obs: Observation) -> torch.Tensor:
        state = obs["robot_state"].extra["policy_state"]
        return await asyncio.to_thread(
            model.native_chunk, obs["images"], obs["instruction"], state
        )

    def close() -> None:
        backbone.close()
        model.policy.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return BenchPolicy(
        name=policy.name,
        policy=policy,
        n_action_steps=steps,
        native=native,
        info={
            "repo": model.repo,
            "chunk_size": int(model.config.chunk_size),
            "n_action_steps": steps,
            "num_denoise_steps": model.num_steps,
            **info,
            "rtc": rtc_config.model_dump() if rtc else None,
        },
        close=close,
    )


def build_lerobot(name: str, spec: EnvSpec, options: dict[str, str]) -> BenchPolicy:
    """Load a LeRobot prefix-KV model and wrap it as a BenchPolicy."""
    opts = dict(options)
    n_action_steps = opts.pop("n_action_steps", None)
    vlm_budget_ms = float(opts.pop("vlm_budget_ms", "2000"))
    rtc, rtc_config = pop_rtc_options(opts)
    model, info = load_model(name, opts, check_cameras=lambda cams: check_cameras(cams, spec))
    return wrap_prefix_kv_policy(
        model,
        spec,
        vlm_budget_ms=vlm_budget_ms,
        rtc=rtc,
        rtc_config=rtc_config,
        n_action_steps=int(n_action_steps) if n_action_steps else None,
        info=info,
    )


def build_smolvla(spec: EnvSpec, options: dict[str, str]) -> BenchPolicy:
    """Load SmolVLA and wrap it as a BenchPolicy."""
    return build_lerobot("smolvla", spec, options)
