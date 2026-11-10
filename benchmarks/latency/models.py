"""Model adapters for the latency benchmark — each one is the model's *native* path.

Every adapter takes the same raw observation a robot driver produces (uint8
camera frames at sensor resolution, float state, instruction) and returns the
actions on the host, so resize / normalize / tokenize, all forward passes and
the device→host copy are inside the timed call. Nothing here goes through the
runtime's split (that is ``benchmarks/sim``); this is the baseline a
requirement is checked against.

Models and the Python environment (``env group``) each needs — the stacks pin
incompatible ``transformers`` versions, so ``run.py`` starts one process per
model with that group's interpreter (``docker/Dockerfile`` builds them all):

==============  =========  ===================================================
model           env group  implementation
==============  =========  ===================================================
mock            (any)      sleeps; CPU plumbing check
openvla         openvla    HF ``openvla/openvla-7b``, ``predict_action`` (1 action / call)
openvla_oft     oft        ``openvla-oft`` repo, ``get_vla_action`` (L1 head, chunk 8)
pi0             lerobot    LeRobot ``PI0Policy.predict_action_chunk``
pi05            lerobot    LeRobot ``PI05Policy.predict_action_chunk``
smolvla         lerobot    LeRobot ``SmolVLAPolicy.predict_action_chunk``
openpi          openpi     Physical Intelligence ``openpi`` (JAX), ``Policy.infer``
==============  =========  ===================================================

Per-model options come in as ``--opt <model>.<key>=<value>``; each adapter
documents its keys and rejects unknown ones.
"""

from __future__ import annotations

import importlib
import json
import os
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np

from benchmarks.latency.harness import LatencyModel, Observation, StageTimer

DEFAULT_INSTRUCTION = "pick up the red block and put it in the bowl"


def synthetic_image(rng: np.random.Generator, hw: tuple[int, int]) -> np.ndarray:
    """Deterministic camera-like frame: smooth gradients plus sensor noise.

    Compute cost of these models does not depend on pixel content (fixed
    token counts), so a synthetic frame times the same as a real one.
    """
    h, w = hw
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    base = np.stack([x * np.ones_like(y), y * np.ones_like(x), 0.5 * (x + y)], axis=-1)
    noise = rng.normal(0.0, 0.05, size=(h, w, 3)).astype(np.float32)
    return (np.clip(base + noise, 0.0, 1.0) * 255).astype(np.uint8)


class Options:
    """``key=value`` options of one adapter; every key must be consumed.

    Args:
        model: Adapter name, for error messages.
        values: Raw option strings.
    """

    def __init__(self, model: str, values: dict[str, str]) -> None:
        self.model = model
        self._values = dict(values)
        self._used: set[str] = set()

    def get(self, key: str, default: str) -> str:
        self._used.add(key)
        return self._values.get(key, default)

    def get_int(self, key: str, default: int) -> int:
        return int(self.get(key, str(default)))

    def get_float(self, key: str, default: float) -> float:
        return float(self.get(key, str(default)))

    def get_bool(self, key: str, default: bool) -> bool:
        return self.get(key, str(default)).strip().lower() in ("1", "true", "yes", "on")

    def prefixed(self, prefix: str) -> dict[str, Any]:
        """``cfg.num_steps=5`` → ``{"num_steps": 5}`` (JSON-decoded when possible)."""
        out: dict[str, Any] = {}
        for key, raw in self._values.items():
            if key.startswith(prefix):
                self._used.add(key)
                try:
                    out[key[len(prefix):]] = json.loads(raw)
                except json.JSONDecodeError:
                    out[key[len(prefix):]] = raw
        return out

    def check_consumed(self) -> None:
        unknown = sorted(set(self._values) - self._used)
        if unknown:
            raise ValueError(f"unknown option(s) for {self.model}: {', '.join(unknown)}")


def _torch_sync() -> Callable[[], None] | None:
    import torch

    return torch.cuda.synchronize if torch.cuda.is_available() else None


def _images(rng: np.random.Generator, hw: tuple[int, int], cameras: list[str]) -> dict[str, np.ndarray]:
    return {cam: synthetic_image(rng, hw) for cam in cameras}


# ── mock ──────────────────────────────────────────────────────────────────────


class MockModel:
    """Sleeps instead of computing — exercises the harness on any machine.

    Options:
        preprocess_ms (2), model_ms (20), chunk (50), action_dim (7), jitter_ms (0)
    """

    name = "mock"
    sync = None

    def __init__(self, opts: Options) -> None:
        self.preprocess_ms = opts.get_float("preprocess_ms", 2.0)
        self.model_ms = opts.get_float("model_ms", 20.0)
        self.chunk = opts.get_int("chunk", 50)
        self.action_dim = opts.get_int("action_dim", 7)
        self.jitter_ms = opts.get_float("jitter_ms", 0.0)
        self._rng = np.random.default_rng(0)

    def load(self) -> None:
        return None

    def make_observation(self, rng: np.random.Generator, image_hw: tuple[int, int]) -> Observation:
        return Observation(
            images={"base": synthetic_image(rng, image_hw)},
            state=np.zeros(self.action_dim, dtype=np.float32),
            instruction=DEFAULT_INSTRUCTION,
        )

    def infer(self, obs: Observation, timer: StageTimer) -> np.ndarray:
        with timer.stage("preprocess"):
            time.sleep(self.preprocess_ms / 1e3)
        with timer.stage("model"):
            jitter = abs(self._rng.normal(0.0, self.jitter_ms)) if self.jitter_ms else 0.0
            time.sleep((self.model_ms + jitter) / 1e3)
        return np.zeros((self.chunk, self.action_dim), dtype=np.float32)

    def describe(self) -> dict[str, Any]:
        return {"checkpoint": "mock", "stages": "preprocess + model = total"}


# ── OpenVLA ───────────────────────────────────────────────────────────────────


class OpenVLAModel:
    """OpenVLA-7B through its Hugging Face remote code (README usage).

    One call = one 7-DoF action (7 autoregressive tokens); OpenVLA has no
    action chunking, so control rate equals inference rate.

    Options:
        repo (openvla/openvla-7b), unnorm_key (bridge_orig),
        attn (sdpa | flash_attention_2 | eager), device (cuda:0)
    """

    name = "openvla"

    def __init__(self, opts: Options) -> None:
        self.repo = opts.get("repo", "openvla/openvla-7b")
        self.unnorm_key = opts.get("unnorm_key", "bridge_orig")
        self.attn = opts.get("attn", "sdpa")
        self.device = opts.get("device", "cuda:0")
        self.sync: Callable[[], None] | None = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(self.repo, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.repo,
            attn_implementation=self.attn,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        self.vla.eval()
        self.sync = _torch_sync()

    def make_observation(self, rng: np.random.Generator, image_hw: tuple[int, int]) -> Observation:
        return Observation(
            images=_images(rng, image_hw, ["primary"]),
            state=np.zeros(7, dtype=np.float32),
            instruction=DEFAULT_INSTRUCTION,
        )

    def infer(self, obs: Observation, timer: StageTimer) -> np.ndarray:
        from PIL import Image

        torch = self._torch
        with torch.inference_mode():
            with timer.stage("preprocess"):
                prompt = f"In: What action should the robot take to {obs.instruction}?\nOut:"
                image = Image.fromarray(obs.images["primary"])
                inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)
            with timer.stage("generate"):
                action = self.vla.predict_action(
                    **inputs, unnorm_key=self.unnorm_key, do_sample=False
                )
        return np.asarray(action, dtype=np.float32)[None]

    def describe(self) -> dict[str, Any]:
        return {
            "checkpoint": self.repo,
            "dtype": "bf16",
            "attn": self.attn,
            "unnorm_key": self.unnorm_key,
            "cameras": ["primary"],
            "stages": "preprocess + generate = total (generate includes unnormalize + D2H)",
        }


# ── OpenVLA-OFT ───────────────────────────────────────────────────────────────

_SHADOW_LINK_BYTES = 16 * 2**20


def shadow_checkpoint(src: str, root: str | None = None) -> str:
    """Writable stand-in for a local checkpoint directory.

    ``openvla_utils.get_vla`` rewrites ``config.json`` and copies its own
    ``modeling_prismatic.py`` into the checkpoint, which must not touch a
    shared (often read-only) model zoo. Large files are symlinked; small ones
    are copied, so those writes land in the shadow and never follow a link
    back into the source.

    Args:
        src: Checkpoint directory.
        root: Where shadows live (default ``$VLA_BENCH_SCRATCH`` or the temp dir).

    Returns:
        The shadow directory, rebuilt from ``src`` on every call.
    """
    import shutil
    import tempfile

    root = root or os.environ.get("VLA_BENCH_SCRATCH") or tempfile.gettempdir()
    dst = os.path.join(root, "vla_ckpt_shadow", os.path.basename(os.path.normpath(src)))
    shutil.rmtree(dst, ignore_errors=True)
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        out_dir = os.path.join(dst, os.path.relpath(dirpath, src))
        os.makedirs(out_dir, exist_ok=True)
        for filename in filenames:
            source = os.path.join(dirpath, filename)
            target = os.path.join(out_dir, filename)
            if os.path.getsize(source) >= _SHADOW_LINK_BYTES:
                os.symlink(os.path.abspath(source), target)
            else:
                shutil.copy2(source, target)
    return dst


class OpenVLAOFTModel:
    """OpenVLA-OFT via the ``openvla-oft`` repo's README inference path.

    Parallel decoding + continuous L1 action head, one call = an action chunk.
    ``prismatic.vla.constants`` picks the robot platform (chunk size, action /
    proprio dims) by scanning ``sys.argv`` for "libero" / "aloha" / "bridge";
    the default LIBERO checkpoint name puts "libero" on the command line.

    Options:
        repo (moojink/openvla-7b-oft-finetuned-libero-spatial),
        unnorm_key (libero_spatial_no_noops), num_images (2), use_proprio (true),
        use_film (false), center_crop (true), load_in_8bit (false), load_in_4bit (false)
    """

    name = "openvla_oft"

    def __init__(self, opts: Options) -> None:
        self.repo = opts.get("repo", "moojink/openvla-7b-oft-finetuned-libero-spatial")
        self.cfg = SimpleNamespace(
            model_family="openvla",
            pretrained_checkpoint=self.repo,
            use_l1_regression=True,
            use_diffusion=False,
            num_diffusion_steps_train=50,
            num_diffusion_steps_inference=50,
            use_film=opts.get_bool("use_film", False),
            num_images_in_input=opts.get_int("num_images", 2),
            use_proprio=opts.get_bool("use_proprio", True),
            center_crop=opts.get_bool("center_crop", True),
            num_open_loop_steps=8,
            lora_rank=32,
            unnorm_key=opts.get("unnorm_key", "libero_spatial_no_noops"),
            load_in_8bit=opts.get_bool("load_in_8bit", False),
            load_in_4bit=opts.get_bool("load_in_4bit", False),
            task_suite_name="libero_spatial",
            seed=7,
        )
        self.sync: Callable[[], None] | None = None

    def load(self) -> None:
        from experiments.robot import openvla_utils
        from prismatic.vla import constants

        if os.path.isdir(self.repo):
            self.cfg.pretrained_checkpoint = shadow_checkpoint(self.repo)
        self.cfg.num_open_loop_steps = constants.NUM_ACTIONS_CHUNK
        self.proprio_dim = constants.PROPRIO_DIM
        self.vla = openvla_utils.get_vla(self.cfg)
        self.processor = openvla_utils.get_processor(self.cfg)
        self.action_head = openvla_utils.get_action_head(self.cfg, llm_dim=self.vla.llm_dim)
        self.proprio_projector = (
            openvla_utils.get_proprio_projector(
                self.cfg, llm_dim=self.vla.llm_dim, proprio_dim=self.proprio_dim
            )
            if self.cfg.use_proprio
            else None
        )
        self._get_vla_action = openvla_utils.get_vla_action
        self.sync = _torch_sync()

    def make_observation(self, rng: np.random.Generator, image_hw: tuple[int, int]) -> Observation:
        cameras = ["full_image", "wrist_image"][: self.cfg.num_images_in_input]
        return Observation(
            images=_images(rng, image_hw, cameras),
            state=np.zeros(self.proprio_dim, dtype=np.float32),
            instruction=DEFAULT_INSTRUCTION,
        )

    def infer(self, obs: Observation, timer: StageTimer) -> np.ndarray:
        observation: dict[str, Any] = {**obs.images, "state": obs.state}
        with timer.stage("get_vla_action"):
            actions = self._get_vla_action(
                self.cfg,
                self.vla,
                self.processor,
                observation,
                obs.instruction,
                action_head=self.action_head,
                proprio_projector=self.proprio_projector,
                use_film=self.cfg.use_film,
            )
        return np.asarray(actions, dtype=np.float32)

    def describe(self) -> dict[str, Any]:
        return {
            "checkpoint": self.repo,
            "dtype": "int8" if self.cfg.load_in_8bit else "int4" if self.cfg.load_in_4bit else "bf16",
            "unnorm_key": self.cfg.unnorm_key,
            "num_images": self.cfg.num_images_in_input,
            "use_proprio": self.cfg.use_proprio,
            "chunk": self.cfg.num_open_loop_steps,
            "stages": "get_vla_action = total (resize, processor, forward, head, unnormalize)",
        }


# ── LeRobot: π0 / π0.5 / SmolVLA ─────────────────────────────────────────────

LEROBOT_PRESETS: dict[str, tuple[str, str]] = {
    "pi0": ("pi0", "lerobot/pi0_base"),
    "pi05": ("pi05", "lerobot/pi05_base"),
    "smolvla": ("smolvla", "lerobot/smolvla_base"),
}


class LeRobotModel:
    """A LeRobot policy through its processor pipeline + ``predict_action_chunk``.

    Same path as ``benchmarks/sim/lerobot_smolvla.py::SmolVLAModel.native_chunk``:
    preprocessor (normalize, tokenize, to device) → chunk → postprocessor
    (unnormalize) → host. Cameras and state size come from the checkpoint's
    ``input_features``. π0 / π0.5 tokenizers come from the gated
    ``google/paligemma-3b-pt-224`` repo: set ``HF_TOKEN`` for an account
    that accepted its license.

    Options:
        repo (preset default), type (preset policy type), device (cuda),
        tokenizer — local dir / repo replacing the preprocessor's ``tokenizer_name``
        (e.g. a local copy of the PaliGemma tokenizer for offline runs),
        cfg.<field>=<json> — override any policy config field before loading,
        e.g. ``cfg.dtype=bfloat16``, ``cfg.num_inference_steps=5``, ``cfg.compile_model=true``,
        ``cfg.vlm_model_name=/models/SmolVLM2-500M-Video-Instruct`` (SmolVLA backbone)
    """

    def __init__(self, name: str, opts: Options) -> None:
        policy_type, repo = LEROBOT_PRESETS[name]
        self.name = name
        self.policy_type = opts.get("type", policy_type)
        self.repo = opts.get("repo", repo)
        self.device = opts.get("device", "cuda")
        self.tokenizer = opts.get("tokenizer", None)
        self.overrides = opts.prefixed("cfg.")
        self.sync: Callable[[], None] | None = None

    def load(self) -> None:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self._torch = torch
        config = PreTrainedConfig.from_pretrained(self.repo)
        for key, value in self.overrides.items():
            if not hasattr(config, key):
                raise ValueError(f"{type(config).__name__} has no field {key!r}")
            setattr(config, key, value)
        config.device = self.device
        policy = get_policy_class(self.policy_type).from_pretrained(self.repo, config=config)
        policy.to(self.device)
        policy.eval()
        self.policy = policy
        self.config = policy.config
        pre_overrides: dict[str, dict[str, Any]] = {"device_processor": {"device": self.device}}
        if self.tokenizer:
            pre_overrides["tokenizer_processor"] = {"tokenizer_name": self.tokenizer}
        self.pre, self.post = make_pre_post_processors(
            policy.config,
            pretrained_path=self.repo,
            preprocessor_overrides=pre_overrides,
        )
        self.sync = _torch_sync()

    def _features(self, kind: str) -> dict[str, Any]:
        return {
            key: feature
            for key, feature in self.config.input_features.items()
            if getattr(feature.type, "value", str(feature.type)).upper() == kind
        }

    def make_observation(self, rng: np.random.Generator, image_hw: tuple[int, int]) -> Observation:
        cameras = sorted(self._features("VISUAL"))
        states = self._features("STATE")
        state_dim = sum(int(np.prod(f.shape)) for f in states.values())
        return Observation(
            images=_images(rng, image_hw, cameras),
            state=rng.uniform(-0.5, 0.5, size=state_dim).astype(np.float32),
            instruction=DEFAULT_INSTRUCTION,
            extra={"state_keys": sorted(states)},
        )

    def infer(self, obs: Observation, timer: StageTimer) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            with timer.stage("preprocess"):
                frame: dict[str, Any] = {
                    key: torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
                    for key, img in obs.images.items()
                }
                for key in obs.extra["state_keys"]:
                    frame[key] = torch.from_numpy(obs.state).unsqueeze(0)
                frame["task"] = [obs.instruction]
                batch = self.pre(frame)
            with timer.stage("predict_chunk"):
                chunk = self.policy.predict_action_chunk(batch)
            with timer.stage("postprocess"):
                actions = self.post(chunk)[0].float().cpu().numpy()
        return actions

    def describe(self) -> dict[str, Any]:
        cfg = self.config
        return {
            "checkpoint": self.repo,
            "policy_type": self.policy_type,
            "dtype": str(getattr(cfg, "dtype", "float32")),
            "chunk": getattr(cfg, "chunk_size", None),
            "n_action_steps": getattr(cfg, "n_action_steps", None),
            "denoise_steps": getattr(cfg, "num_inference_steps", getattr(cfg, "num_steps", None)),
            "compile": getattr(cfg, "compile_model", None),
            "tokenizer": self.tokenizer,
            "cameras": sorted(self._features("VISUAL")),
            "overrides": self.overrides,
            "stages": "preprocess + predict_chunk + postprocess = total",
        }


# ── openpi (JAX) ──────────────────────────────────────────────────────────────

_OPENPI_EXAMPLES = {
    "droid": ("openpi.policies.droid_policy", "make_droid_example"),
    "libero": ("openpi.policies.libero_policy", "make_libero_example"),
    "aloha": ("openpi.policies.aloha_policy", "make_aloha_example"),
}


class OpenPIModel:
    """Physical Intelligence's reference implementation (JAX), README usage.

    The example observation comes from the config family's ``make_*_example``
    with its HWC frames swapped for raw-resolution ones, so the policy's own
    resize transform is timed too. ``model_inner`` is openpi's own
    ``policy_timing.infer_ms`` (model only), nested inside ``infer``.

    Options:
        config (pi05_droid), checkpoint (gs://openpi-assets/checkpoints/<config>),
        family (droid | libero | aloha; default: guessed from the config name)
    """

    name = "openpi"
    sync = None  # infer() returns host numpy arrays

    def __init__(self, opts: Options) -> None:
        self.config_name = opts.get("config", "pi05_droid")
        self.checkpoint = opts.get("checkpoint", f"gs://openpi-assets/checkpoints/{self.config_name}")
        guess = next((f for f in _OPENPI_EXAMPLES if f in self.config_name), "droid")
        self.family = opts.get("family", guess)
        if self.family not in _OPENPI_EXAMPLES:
            raise ValueError(f"openpi family {self.family!r} not in {sorted(_OPENPI_EXAMPLES)}")

    def load(self) -> None:
        # Without this JAX grabs 75% of the GPU up front and the memory column is meaningless.
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        from openpi.policies import policy_config
        from openpi.shared import download
        from openpi.training import config as openpi_config

        config = openpi_config.get_config(self.config_name)
        checkpoint_dir = download.maybe_download(self.checkpoint)
        self.policy = policy_config.create_trained_policy(config, checkpoint_dir)
        module, fn = _OPENPI_EXAMPLES[self.family]
        self._make_example = getattr(importlib.import_module(module), fn)

    def make_observation(self, rng: np.random.Generator, image_hw: tuple[int, int]) -> Observation:
        example = dict(self._make_example())
        images: dict[str, np.ndarray] = {}
        for key, value in example.items():
            if isinstance(value, np.ndarray) and value.dtype == np.uint8 and value.ndim == 3 and value.shape[-1] == 3:
                images[key] = synthetic_image(rng, image_hw)
                example[key] = images[key]
        return Observation(
            images=images,
            state=np.zeros(0, dtype=np.float32),
            instruction=str(example.get("prompt", DEFAULT_INSTRUCTION)),
            extra={"example": example},
        )

    def infer(self, obs: Observation, timer: StageTimer) -> np.ndarray:
        with timer.stage("infer"):
            out = self.policy.infer(dict(obs.extra["example"]))
        inner = out.get("policy_timing", {}).get("infer_ms")
        if inner is not None:
            timer.record("model_inner", float(inner))
        return np.asarray(out["actions"], dtype=np.float32)

    def describe(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "config": self.config_name,
            "family": self.family,
            "stages": "infer = total; model_inner (openpi's own model time) is nested in it",
        }


# ── Registry ──────────────────────────────────────────────────────────────────

ENV_GROUPS: dict[str, str] = {
    "mock": "mock",
    "openvla": "openvla",
    "openvla_oft": "oft",
    "pi0": "lerobot",
    "pi05": "lerobot",
    "smolvla": "lerobot",
    "openpi": "openpi",
}


def make_model(name: str, options: dict[str, str]) -> LatencyModel:
    """Adapter by name; raises ValueError on an unknown model or option."""
    opts = Options(name, options)
    model: LatencyModel
    if name == "mock":
        model = MockModel(opts)
    elif name == "openvla":
        model = OpenVLAModel(opts)
    elif name == "openvla_oft":
        model = OpenVLAOFTModel(opts)
    elif name in LEROBOT_PRESETS:
        model = LeRobotModel(name, opts)
    elif name == "openpi":
        model = OpenPIModel(opts)
    else:
        raise ValueError(f"unknown model {name!r}; expected one of {', '.join(ENV_GROUPS)}")
    opts.check_consumed()
    return model
