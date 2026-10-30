"""π0.5 (LeRobot) split at the PaliGemma prefix → action expert boundary.

Same structure as ``benchmarks/sim/lerobot_smolvla.py``: ``PI05Model.encode_prefix``
runs PaliGemma over images + language and returns its per-layer K/V as a
``per_layer_kv`` condition (ADR-019); ``denoise`` runs the Gemma action expert's
flow-matching Euler loop over that cache. Together they reproduce
``PI05Pytorch.sample_actions`` step for step, which
``benchmarks/benchmark_sim.py::test_pi05_split_matches_native`` checks against
LeRobot's own ``predict_action_chunk`` with shared noise.

Capabilities: π0.5 discretizes the normalized state into the language prompt
(``pi05_prepare_state_tokenizer_processor_step``), so state sits in the prefix →
``state_placement="prefix"`` → coupled frequency (ADR-018), like SmolVLA.

RTC (ADR-015, ``--policy-opt rtc=1``): the shared ``euler_denoise`` loop, same
time convention as SmolVLA (1 = noise → 0 = data).

Built against LeRobot 0.6 (``lerobot.policies.common`` helpers, transformers
``DynamicCache`` prefix). It relies on these internals, all used verbatim by
``sample_actions``: ``PI05Policy._preprocess_images``, ``model.embed_prefix``,
``model.paligemma_with_expert.forward``, ``model.denoise_step``,
``model.sample_noise``, ``make_att_2d_masks`` and ``prepare_attention_masks_4d``.
``denoise_step`` clones the cache it is given, so one rebuilt cache serves all
steps of a call.

Policy options (``--policy-opt key=value``):
    repo             checkpoint directory or repo id (default lerobot/pi05_libero_finetuned)
    device           default cuda
    cameras          env_cam=lerobot_key pairs joined by ',' (default: LIBERO
                     agentview→observation.images.image, wrist→observation.images.image2)
    n_action_steps   override the checkpoint's value
    vlm_budget_ms    backbone latency budget reported to the scheduler
    tokenizer        local PaliGemma tokenizer directory replacing the checkpoint's
                     ``google/paligemma-3b-pt-224``, for offline hosts
    rtc, rtc_beta, rtc_schedule, rtc_jacobian   as for smolvla
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from benchmarks.sim.envs import EnvSpec
from benchmarks.sim.lerobot_smolvla import (
    LIBERO_CAMERAS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    check_cameras,
    euler_denoise,
    parse_camera_map,
    pop_rtc_options,
    wrap_prefix_kv_policy,
)
from benchmarks.sim.policies import BenchPolicy
from vla_runtime.action_head.rtc import RTCGuidance
from vla_runtime.vlm.base import Embedding, LayerKV

DEFAULT_REPO = "lerobot/pi05_libero_finetuned"


class PI05Model:
    """The loaded LeRobot π0.5 policy and its processors, shared by backbone and head.

    Args:
        repo: Hugging Face repo id or local checkpoint directory.
        device: Where the model runs.
        camera_map: Env camera name → LeRobot image feature key. Image features
            the map leaves out are fed as LeRobot's masked empty cameras.
        tokenizer: Local PaliGemma tokenizer directory; None keeps the
            checkpoint's tokenizer id.
    """

    def __init__(
        self,
        repo: str,
        device: torch.device,
        camera_map: Mapping[str, str],
        tokenizer: str | None = None,
    ) -> None:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05 import modeling_pi05

        self.repo = repo
        self.device = device
        self.camera_map = dict(camera_map)
        self._make_att_2d_masks = modeling_pi05.make_att_2d_masks
        self._prepare_attention_masks_4d = modeling_pi05.prepare_attention_masks_4d
        config = PreTrainedConfig.from_pretrained(repo)
        # Training-time settings: compile changes kernels (a V4 measurement of
        # its own), checkpointing only matters with gradients.
        config.compile_model = False
        config.gradient_checkpointing = False
        config.device = str(device)
        pre_overrides: dict[str, dict[str, Any]] = {"device_processor": {"device": str(device)}}
        if tokenizer:
            pre_overrides["tokenizer_processor"] = {"tokenizer_name": tokenizer}
        self.policy = modeling_pi05.PI05Policy.from_pretrained(repo, config=config)
        self.policy.to(device)
        self.policy.eval()
        # Inference only; RTC's VJP must not build graphs for the weights.
        self.policy.requires_grad_(False)
        self.config = self.policy.config
        self.pre, self.post = make_pre_post_processors(
            self.policy.config,
            pretrained_path=repo,
            preprocessor_overrides=pre_overrides,
        )
        expected = set(getattr(self.config, "image_features", {}) or {})
        mapped = set(self.camera_map.values())
        if expected and not mapped <= expected:
            raise ValueError(
                f"camera map targets {sorted(mapped - expected)} which {repo} does not "
                f"take; it expects {sorted(expected)} (set --policy-opt cameras=...)"
            )

    @property
    def model_id(self) -> str:
        return f"pi05:{self.repo}"

    @property
    def num_steps(self) -> int:
        return int(self.config.num_inference_steps)

    @property
    def action_dim(self) -> int:
        return int(self.config.action_feature.shape[0])

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Executed-space actions [n, action_dim] → the model's normalized space."""
        for step in getattr(self.post, "steps", ()):
            if hasattr(step, "_normalize_action"):
                batch = actions.to(self.device, dtype=torch.float32).unsqueeze(0)
                return step._normalize_action(batch, inverse=False)[0]
        raise ValueError(f"{self.repo} postprocessor has no action unnormalizer to invert")

    def preprocess(
        self, images: Mapping[str, torch.Tensor], text: str, policy_state: torch.Tensor
    ) -> dict[str, Any]:
        """One observation → the normalized batch with state tokenized into the prompt."""
        frame: dict[str, Any] = {}
        for src, dst in self.camera_map.items():
            if src not in images:
                raise KeyError(f"camera {src!r} missing from observation (has {sorted(images)})")
            frame[dst] = images[src].unsqueeze(0).float()
        frame[OBS_STATE] = policy_state.unsqueeze(0).float()
        frame["task"] = [text]
        return self.pre(frame)

    @torch.inference_mode()
    def encode_prefix(
        self, images: Mapping[str, torch.Tensor], text: str, policy_state: torch.Tensor
    ) -> Embedding:
        """PaliGemma prefix forward → per-layer K/V condition (what sample_actions caches)."""
        batch = self.preprocess(images, text, policy_state)
        model = self.policy.model
        imgs, img_masks = self.policy._preprocess_images(batch)
        prefix_embs, pad_masks, att_masks = model.embed_prefix(
            imgs, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        )
        att_2d = self._make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_4d = self._prepare_attention_masks_4d(att_2d)
        model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = (
            "eager"
        )
        _, past_key_values = model.paligemma_with_expert.forward(
            attention_mask=att_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        entries = list(past_key_values)  # (keys [B, H, L, D], values, sliding_window)
        return Embedding(
            kind="per_layer_kv",
            layer_kv=tuple(LayerKV(key=k[0], value=v[0]) for k, v, _ in entries),
            attention_mask=pad_masks[0].bool(),
            model_id=self.model_id,
            metadata={"sliding_windows": [sw for _, _, sw in entries]},
        )

    def _prefix_cache(self, embedding: Embedding, clone: bool) -> Any:
        from transformers import DynamicCache

        return DynamicCache(
            tuple(
                (
                    (kv.key.clone() if clone else kv.key).unsqueeze(0),
                    (kv.value.clone() if clone else kv.value).unsqueeze(0),
                    sw,
                )
                for kv, sw in zip(
                    embedding.layer_kv, embedding.metadata["sliding_windows"], strict=True
                )
            )
        )

    def denoise(
        self,
        embedding: Embedding,
        noise: torch.Tensor | None = None,
        guidance: RTCGuidance | None = None,
    ) -> torch.Tensor:
        """Action expert Euler loop over a cached prefix → [chunk_size, action_dim] (CPU)."""
        if guidance is not None and guidance.jacobian == "vjp":
            with torch.no_grad():
                return self._denoise(embedding, noise, guidance, detach_cache=True)
        with torch.inference_mode():
            return self._denoise(embedding, noise, guidance, detach_cache=False)

    def _denoise(
        self,
        embedding: Embedding,
        noise: torch.Tensor | None,
        guidance: RTCGuidance | None,
        detach_cache: bool,
    ) -> torch.Tensor:
        model = self.policy.model
        config = self.config
        # Inference-mode tensors cannot enter an autograd graph; copies can.
        past_key_values = self._prefix_cache(embedding, clone=detach_cache)
        pad_masks = embedding.attention_mask.unsqueeze(0)
        if detach_cache:
            pad_masks = pad_masks.clone()
        device = pad_masks.device
        if noise is None:
            noise = model.sample_noise((1, config.chunk_size, config.max_action_dim), device)
        target = None
        if guidance is not None:
            target = torch.zeros(config.chunk_size, config.max_action_dim, device=device)
            target[:, : self.action_dim] = self.normalize_actions(guidance.target)
        model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        x_t = euler_denoise(
            lambda x, t: model.denoise_step(
                prefix_pad_masks=pad_masks, past_key_values=past_key_values, x_t=x, timestep=t
            ),
            noise,
            self.num_steps,
            target,
            guidance,
        )
        return self._postprocess(x_t[:, :, : self.action_dim])

    @torch.inference_mode()
    def native_chunk(
        self,
        images: Mapping[str, torch.Tensor],
        text: str,
        policy_state: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """LeRobot's own monolithic path — the reference the split must match."""
        batch = self.preprocess(images, text, policy_state)
        actions = self.policy.predict_action_chunk(batch, noise=noise)
        return self._postprocess(actions)

    def _postprocess(self, actions: torch.Tensor) -> torch.Tensor:
        out = self.post(actions)
        return out[0].float().cpu()


def build_pi05(spec: EnvSpec, options: dict[str, str]) -> BenchPolicy:
    """Load π0.5 and wrap it as a BenchPolicy (see module docstring for options)."""
    opts = dict(options)
    repo = opts.pop("repo", DEFAULT_REPO)
    device = torch.device(opts.pop("device", "cuda"))
    cameras = parse_camera_map(opts.pop("cameras")) if "cameras" in opts else dict(LIBERO_CAMERAS)
    n_action_steps = opts.pop("n_action_steps", None)
    vlm_budget_ms = float(opts.pop("vlm_budget_ms", "2000"))
    tokenizer = opts.pop("tokenizer", None)
    rtc, rtc_config = pop_rtc_options(opts)
    if opts:
        raise ValueError(f"unknown --policy-opt for pi05: {sorted(opts)}")
    check_cameras(cameras, spec)

    model = PI05Model(repo, device, cameras, tokenizer=tokenizer)
    return wrap_prefix_kv_policy(
        model,
        spec,
        vlm_budget_ms=vlm_budget_ms,
        rtc=rtc,
        rtc_config=rtc_config,
        n_action_steps=int(n_action_steps) if n_action_steps else None,
        info={"device": str(device), "cameras": cameras},
    )
