"""Load a model adapter by name from ``key=value`` options.

The sim benchmark (``--policy-opt``) and the policy server (``--policy-opt``)
share these names, so one option string means the same thing in both.

Models:
    smolvla   LeRobot SmolVLA, split at the prefix-KV boundary (GPU)
    pi05      LeRobot π0.5, split at the PaliGemma prefix-KV boundary (GPU)

Options (both models unless noted):
    repo             checkpoint directory or repo id
                     (default HuggingFaceVLA/smolvla_libero / lerobot/pi05_libero_finetuned)
    device           default cuda
    cameras          robot_cam=lerobot_key pairs joined by ',' (default: LIBERO,
                     agentview→observation.images.image, wrist→observation.images.image2)
    vlm              smolvla: local SmolVLM2 directory replacing the checkpoint's
                     ``vlm_model_name`` (backbone config + tokenizer), for offline hosts
    tokenizer        pi05: local PaliGemma tokenizer directory replacing
                     ``google/paligemma-3b-pt-224``, for offline hosts
    cuda_graph       pi05: 1 replays the unguided expert loop from CUDA Graphs (default 0)
    vision_bf16      pi05: 1 runs SigLIP and the projector in bf16 instead of fp32 (default 0)
    expert_sdpa      pi05: batch size from which the expert uses sdpa attention
                     (default 0: never)

RTC options (``pop_rtc_options``): ``rtc`` (1 declares supports_rtc and guides
the expert), ``rtc_beta``, ``rtc_schedule``, ``rtc_jacobian``.

LeRobot is imported only when a model is loaded.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from vla_runtime.action_head.rtc import RTCConfig
from vla_runtime.models._lerobot import LIBERO_CAMERAS, parse_camera_map
from vla_runtime.models.prefix_kv import PrefixKVModel

MODEL_NAMES = ("smolvla", "pi05")

CameraCheck = Callable[[Mapping[str, str]], None]


def pop_rtc_options(opts: dict[str, str]) -> tuple[bool, RTCConfig]:
    """Take ``rtc`` / ``rtc_beta`` / ``rtc_schedule`` / ``rtc_jacobian`` out of ``opts``."""
    rtc = opts.pop("rtc", "0").lower() not in ("0", "false", "no", "")
    fields: dict[str, Any] = {}
    if "rtc_beta" in opts:
        fields["max_guidance_weight"] = float(opts.pop("rtc_beta"))
    if "rtc_schedule" in opts:
        fields["prefix_schedule"] = opts.pop("rtc_schedule")
    if "rtc_jacobian" in opts:
        fields["jacobian"] = opts.pop("rtc_jacobian")
    return rtc, RTCConfig(**fields)


def _flag(opts: dict[str, str], key: str) -> bool:
    return opts.pop(key, "0").lower() in ("1", "true", "yes")


def load_model(
    name: str,
    options: Mapping[str, str],
    check_cameras: CameraCheck | None = None,
) -> tuple[PrefixKVModel, dict[str, Any]]:
    """Load a prefix-KV model adapter.

    Args:
        name: ``smolvla`` or ``pi05``.
        options: Model options (module docstring). RTC and scheduling options
            must already be popped by the caller.
        check_cameras: Called with the camera map before any weights load, so
            a mapping the caller cannot serve fails fast.

    Returns:
        (model, info): info records the options that shaped the model, for
        result files and handshakes.

    Raises:
        ValueError: Unknown model or option, or a camera map the checkpoint
            does not take.
    """
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name!r}; expected one of {list(MODEL_NAMES)}")
    opts = dict(options)
    device = torch.device(opts.pop("device", "cuda"))
    cameras = parse_camera_map(opts.pop("cameras")) if "cameras" in opts else dict(LIBERO_CAMERAS)
    info: dict[str, Any] = {"device": str(device), "cameras": cameras}

    if name == "smolvla":
        from vla_runtime.models.smolvla import DEFAULT_REPO, SmolVLAModel

        repo = opts.pop("repo", DEFAULT_REPO)
        vlm = opts.pop("vlm", None)
        _reject_unknown(name, opts)
        if check_cameras is not None:
            check_cameras(cameras)
        return SmolVLAModel(repo, device, cameras, vlm=vlm), info

    from vla_runtime.models.pi05 import DEFAULT_REPO, PI05Model

    repo = opts.pop("repo", DEFAULT_REPO)
    tokenizer = opts.pop("tokenizer", None)
    cuda_graph = _flag(opts, "cuda_graph")
    vision_bf16 = _flag(opts, "vision_bf16")
    expert_sdpa = int(opts.pop("expert_sdpa", "0"))
    _reject_unknown(name, opts)
    if check_cameras is not None:
        check_cameras(cameras)
    info.update(cuda_graph=cuda_graph, vision_bf16=vision_bf16, expert_sdpa=expert_sdpa)
    model = PI05Model(
        repo,
        device,
        cameras,
        tokenizer=tokenizer,
        cuda_graph=cuda_graph,
        vision_bf16=vision_bf16,
        expert_sdpa_min_batch=expert_sdpa or None,
    )
    return model, info


def _reject_unknown(name: str, opts: Mapping[str, str]) -> None:
    if opts:
        raise ValueError(f"unknown --policy-opt for {name}: {sorted(opts)}")
