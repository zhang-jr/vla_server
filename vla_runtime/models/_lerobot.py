"""LeRobot conventions shared by the LeRobot model adapters.

Feature keys of LeRobot's processor pipeline and the camera mapping from a
robot's (or simulator's) camera names to a checkpoint's image feature keys.
"""

from __future__ import annotations

try:
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
except ImportError:  # older layout / constants moved
    OBS_LANGUAGE_TOKENS = "observation.language.tokens"
    OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"

OBS_STATE = "observation.state"


# LIBERO convention, used by the LIBERO-finetuned SmolVLA / π0.5 checkpoints:
# robot camera name → LeRobot image feature key.
LIBERO_CAMERAS = {
    "agentview": "observation.images.image",
    "wrist": "observation.images.image2",
}


def parse_camera_map(value: str) -> dict[str, str]:
    """``"a=observation.images.x,b=observation.images.y"`` → dict."""
    mapping: dict[str, str] = {}
    for pair in value.split(","):
        if not pair:
            continue
        src, sep, dst = pair.partition("=")
        if not sep:
            raise ValueError(f"camera mapping {pair!r} is not env_cam=lerobot_key")
        mapping[src.strip()] = dst.strip()
    return mapping
