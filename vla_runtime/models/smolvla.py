"""SmolVLA (LeRobot) split at the VLM → action expert boundary.

``SmolVLAModel.encode_prefix()`` runs the VLM over the prefix (images +
language + state) and returns its per-layer K/V as a ``per_layer_kv``
condition (ADR-019); ``denoise()`` / ``denoise_batch()`` run the action
expert's flow-matching loop over that cache. Together they reproduce
``VLAFlowMatching.sample_actions`` step for step, which
``benchmarks/benchmark_sim.py::test_smolvla_split_matches_native`` checks
numerically against LeRobot's own ``predict_action_chunk`` with shared noise.
The generic backbone / head / policy wrapping is ``vla_runtime.models.prefix_kv``.

Capabilities: SmolVLA feeds state into the prefix → ``state_placement="prefix"``
→ coupled frequency (ADR-018). The scheduler therefore passes state to
encode(), and the head never re-runs on a cached condition.

RTC (ADR-015): SmolVLA integrates time from 1 (noise) to 0, so each step hands
the guidance ``tau = 1 - time`` and the sign-flipped velocity
(``euler_denoise``). The previous actions arrive in the executed (env) space
and are normalized with the checkpoint's own action statistics before guiding.
``benchmarks/benchmark_sim.py`` checks the ``identity`` mode against LeRobot's
built-in ``RTCProcessor`` with shared noise.

Built against LeRobot's processor API (``make_pre_post_processors``, 0.4+).
It relies on these SmolVLA internals, all used verbatim by ``sample_actions``:
``prepare_images``, ``prepare_state``, ``model.embed_prefix``,
``model.vlm_with_expert.forward``, ``model.denoise_step``,
``model.sample_noise`` and ``make_att_2d_masks``. Two prefix-cache layouts are
handled: LeRobot < 0.6 passes ``fill_kv_cache=True`` and returns a
``{layer: {"key_states", "value_states"}}`` dict ([B, L, H, D]); 0.6+ fills a
transformers ``DynamicCache`` ([B, H, L, D]) that ``denoise_step`` appends to
and crops back in place, so a fresh one is rebuilt for every denoise call. If
a LeRobot upgrade renames any of them, the alignment test fails loudly rather
than the benchmark silently measuring a different computation.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from vla_runtime.action_head.rtc import RTCGuidance
from vla_runtime.models._lerobot import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from vla_runtime.models.prefix_kv import (
    batch_guidance,
    euler_denoise,
    stack_prefix_kv,
    wait_for_gpu,
)
from vla_runtime.vlm.base import Embedding, LayerKV

DEFAULT_REPO = "HuggingFaceVLA/smolvla_libero"


class SmolVLAModel:
    """The loaded LeRobot policy and its processors, shared by backbone and head.

    Args:
        repo: Hugging Face repo id or local checkpoint directory.
        device: Where the model runs.
        camera_map: Env camera name → LeRobot image feature key.
        vlm: Local SmolVLM2 directory replacing the checkpoint's ``vlm_model_name``
            for both the backbone config and the tokenizer; None keeps the repo id.
    """

    def __init__(
        self,
        repo: str,
        device: torch.device,
        camera_map: Mapping[str, str],
        vlm: str | None = None,
    ) -> None:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import (
            SmolVLAPolicy,
            make_att_2d_masks,
        )

        self.repo = repo
        self.device = device
        self.camera_map = dict(camera_map)
        config = PreTrainedConfig.from_pretrained(repo)
        pre_overrides: dict[str, dict[str, Any]] = {"device_processor": {"device": str(device)}}
        if vlm:
            config.vlm_model_name = vlm
            pre_overrides["tokenizer_processor"] = {"tokenizer_name": vlm}
        self.policy = SmolVLAPolicy.from_pretrained(repo, config=config)
        self.policy.to(device)
        self.policy.eval()
        # Inference only; RTC's VJP must not build graphs for the weights.
        self.policy.requires_grad_(False)
        self.config = self.policy.config
        self.make_att_2d_masks = make_att_2d_masks
        self._legacy_cache = "fill_kv_cache" in inspect.signature(
            self.policy.model.vlm_with_expert.forward
        ).parameters
        self.pre, self.post = make_pre_post_processors(
            self.policy.config,
            pretrained_path=repo,
            preprocessor_overrides=pre_overrides,
        )
        if not getattr(self.config, "use_cache", True):
            raise ValueError(
                f"{repo} has use_cache=False; the split adapter needs the prefix KV cache"
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
        return f"smolvla:{self.repo}"

    @property
    def num_steps(self) -> int:
        return int(self.config.num_steps)

    @property
    def action_dim(self) -> int:
        return int(self.config.action_feature.shape[0])

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Executed-space actions [n, action_dim] → the model's normalized space.

        Inverse of the postprocessor's unnormalizer, using the same statistics.
        """
        for step in getattr(self.post, "steps", ()):
            if hasattr(step, "_normalize_action"):
                batch = actions.to(self.device, dtype=torch.float32).unsqueeze(0)
                return step._normalize_action(batch, inverse=False)[0]
        raise ValueError(f"{self.repo} postprocessor has no action unnormalizer to invert")

    def preprocess(
        self, images: Mapping[str, torch.Tensor], text: str, policy_state: torch.Tensor
    ) -> dict[str, Any]:
        """One observation → the normalized, tokenized batch the model reads."""
        frame: dict[str, Any] = {}
        for src, dst in self.camera_map.items():
            if src not in images:
                raise KeyError(f"camera {src!r} missing from observation (has {sorted(images)})")
            frame[dst] = images[src].unsqueeze(0).float()
        frame[OBS_STATE] = policy_state.unsqueeze(0).float()
        frame["task"] = [text]
        batch = self.pre(frame)
        prepare = getattr(self.policy, "_prepare_batch", None)
        return prepare(batch) if prepare is not None else batch

    @torch.inference_mode()
    def encode_prefix(
        self, images: Mapping[str, torch.Tensor], text: str, policy_state: torch.Tensor
    ) -> Embedding:
        """Prefix forward pass → per-layer K/V condition (what sample_actions caches)."""
        batch = self.preprocess(images, text, policy_state)
        model = self.policy.model
        imgs, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        prefix_embs, pad_masks, att_masks = model.embed_prefix(
            imgs,
            img_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state=state,
        )
        att_2d = self.make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        extra = {"fill_kv_cache": True} if self._legacy_cache else {}
        _, past_key_values = model.vlm_with_expert.forward(
            attention_mask=att_2d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            **extra,
        )
        prefix_len = pad_masks.shape[1]
        if self._legacy_cache:
            layer_indices = sorted(past_key_values)
            entries = [
                (past_key_values[i]["key_states"], past_key_values[i]["value_states"])
                for i in layer_indices
            ]
        else:
            layer_indices = list(range(len(past_key_values.layers)))
            entries = [(layer.keys, layer.values) for layer in past_key_values.layers]
        layout = _kv_layout(entries[0][0], prefix_len)
        layers = tuple(
            LayerKV(key=_heads_first(key, layout), value=_heads_first(value, layout))
            for key, value in entries
        )
        return Embedding(
            kind="per_layer_kv",
            layer_kv=layers,
            attention_mask=pad_masks[0].bool(),
            model_id=self.model_id,
            metadata={"layer_indices": layer_indices, "kv_layout": layout},
        )

    def _prefix_cache(
        self, embeddings: Sequence[Embedding], clone: bool
    ) -> tuple[Any, torch.Tensor]:
        """Rebuild the model's prefix cache from per_layer_kv conditions, one row each.

        Returns:
            (cache, prefix pad masks [B, L]).
        """
        layout = embeddings[0].metadata["kv_layout"]
        layer_indices = embeddings[0].metadata["layer_indices"]
        if any(
            e.metadata["kv_layout"] != layout or e.metadata["layer_indices"] != layer_indices
            for e in embeddings
        ):
            raise ValueError("batched SmolVLA conditions disagree on their K/V layout")
        layers, pad_masks = stack_prefix_kv(embeddings, clone)
        pairs = [
            (
                i,
                key.transpose(1, 2) if layout == "BLHD" else key,
                value.transpose(1, 2) if layout == "BLHD" else value,
            )
            for i, (key, value) in zip(layer_indices, layers)
        ]
        if self._legacy_cache:
            return {i: {"key_states": k, "value_states": v} for i, k, v in pairs}, pad_masks
        from transformers import DynamicCache

        cache = DynamicCache()
        for i, k, v in pairs:
            # DynamicCache stores [B, H, L, D].
            cache.update(_to_bhld(k, layout), _to_bhld(v, layout), i)
        return cache, pad_masks

    def denoise(
        self,
        embedding: Embedding,
        noise: torch.Tensor | None = None,
        guidance: RTCGuidance | None = None,
    ) -> torch.Tensor:
        """Action expert Euler loop over a cached prefix → [chunk_size, action_dim] (CPU).

        With ``guidance`` (RTC), every step's velocity is guided toward the
        previous chunk's tail; the ``vjp`` mode needs autograd, so it runs
        outside inference mode on a copy of the cached K/V.
        """
        return self.denoise_batch([embedding], noise, [guidance])[0]

    def denoise_batch(
        self,
        embeddings: Sequence[Embedding],
        noise: torch.Tensor | None = None,
        guidances: Sequence[RTCGuidance | None] | None = None,
    ) -> list[torch.Tensor]:
        """``denoise`` for several rows at once (ADR-022) → one CPU tensor per row.

        Args:
            embeddings: One per_layer_kv condition per row.
            noise: ``[B, chunk_size, max_action_dim]``; sampled when None.
            guidances: One RTC guidance (or None) per row.
        """
        guidances = list(guidances) if guidances is not None else [None] * len(embeddings)
        if any(g is not None and g.jacobian == "vjp" for g in guidances):
            with torch.no_grad():
                return self._denoise(embeddings, noise, guidances, detach_cache=True)
        with torch.inference_mode():
            return self._denoise(embeddings, noise, guidances, detach_cache=False)

    def _denoise(
        self,
        embeddings: Sequence[Embedding],
        noise: torch.Tensor | None,
        guidances: Sequence[RTCGuidance | None],
        detach_cache: bool,
    ) -> list[torch.Tensor]:
        model = self.policy.model
        config = self.config
        # Inference-mode tensors cannot enter an autograd graph; copies can.
        past_key_values, pad_masks = self._prefix_cache(embeddings, clone=detach_cache)
        device = pad_masks.device
        shape = (len(embeddings), config.chunk_size, config.max_action_dim)
        if noise is None:
            noise = model.sample_noise(shape, device)
        # Env space → normalized, zero-padded to the expert's max_action_dim.
        target, guidance = batch_guidance(
            guidances, self.normalize_actions, shape, self.action_dim, device
        )
        x_t = euler_denoise(
            lambda x, t: model.denoise_step(pad_masks, past_key_values, x, t),
            noise,
            config.num_steps,
            target,
            guidance,
        )
        actions = x_t[:, :, : self.action_dim]
        return self._postprocess(actions)

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
        return self._postprocess(actions)[0]

    def _postprocess(self, actions: torch.Tensor) -> list[torch.Tensor]:
        wait_for_gpu(actions)
        out = self.post(actions)
        return list(out.float().cpu().unbind(0))


def _kv_layout(key: torch.Tensor, prefix_len: int) -> str:
    """SmolVLA caches [B, L, H, D]; accept [B, H, L, D] too, refuse anything else."""
    if key.dim() == 4 and key.shape[1] == prefix_len:
        return "BLHD"
    if key.dim() == 4 and key.shape[2] == prefix_len:
        return "BHLD"
    raise ValueError(
        f"unexpected prefix K/V shape {tuple(key.shape)} for prefix length {prefix_len}"
    )


def _heads_first(t: torch.Tensor, layout: str) -> torch.Tensor:
    """Cached tensor (batch 1) → LayerKV layout [heads, seq, head_dim]."""
    return t[0].transpose(0, 1) if layout == "BLHD" else t[0]


def _batch_first(t: torch.Tensor, layout: str) -> torch.Tensor:
    """Inverse of _heads_first."""
    return (t.transpose(0, 1) if layout == "BLHD" else t).unsqueeze(0)


def _to_bhld(t: torch.Tensor, layout: str) -> torch.Tensor:
    """Batch-first cached tensor in ``layout`` → [B, H, L, D]."""
    return t.transpose(1, 2) if layout == "BLHD" else t
