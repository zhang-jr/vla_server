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
time convention as SmolVLA (1 = noise → 0 = data). Head batching (ADR-022):
``denoise_batch``, as for SmolVLA.

CUDA Graph (V4, ``--policy-opt cuda_graph=1``): the unguided Euler loop is
captured once per batch size and replayed (``benchmarks/sim/cuda_graph.py``).
The expert is launch-bound, so this cuts the 10-step loop from 89 to 33 ms at
batch 1 with bit-identical output. RTC-guided calls stay eager.

bf16 vision (``--policy-opt vision_bf16=1``): LeRobot keeps SigLIP and the
projector in fp32 so training never toggles dtypes; at inference that is a
third of the prefix's GPU time in fp32 GEMMs. bf16 weights take the prefix
from 67 to 51 ms; actions move by up to ~6e-3, so it is accepted on closed-loop
success (CLAUDE.md principle 3 allows the VLM lower precision, not the head).

Expert sdpa (``--policy-opt expert_sdpa=N``): ``denoise_step`` pins the expert
to eager attention, whose ``repeat_kv`` and elementwise softmax are about a
quarter of a graphed batch-4 pass. From batch N on the expert runs sdpa with a
boolean mask instead (graphed B=4 72 -> 51 ms, B=8 109 -> 70 ms); below N it
stays eager, which is faster at batch 1. sdpa reorders the softmax, so actions
move by ~5e-3 and the option is accepted on closed-loop success.

Built against LeRobot 0.6 (``lerobot.policies.common`` helpers, transformers
``DynamicCache``). It relies on these internals, all used verbatim by
``sample_actions``: ``PI05Policy._preprocess_images``, ``model.embed_prefix``,
``model.paligemma_with_expert.forward``, ``model.sample_noise``,
``create_sinusoidal_pos_embedding``, ``make_att_2d_masks`` and
``prepare_attention_masks_4d``. ``_velocity`` restates ``model.denoise_step``
in a capturable form; ``benchmarks/benchmark_sim.py`` checks it bit for bit.

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
    cuda_graph       1 replays the unguided expert loop from CUDA Graphs (default 0)
    vision_bf16      1 runs SigLIP and the projector in bf16 instead of fp32 (default 0)
    expert_sdpa      batch size from which the expert uses sdpa attention (default 0: never)
"""

from __future__ import annotations

import functools
import types
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

from benchmarks.sim.cuda_graph import GraphCache
from benchmarks.sim.envs import EnvSpec
from benchmarks.sim.lerobot_smolvla import (
    LIBERO_CAMERAS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    batch_guidance,
    check_cameras,
    euler_denoise,
    parse_camera_map,
    pop_rtc_options,
    stack_prefix_kv,
    wait_for_gpu,
    wrap_prefix_kv_policy,
)
from benchmarks.sim.policies import BenchPolicy
from vla_runtime.action_head.rtc import RTCGuidance
from vla_runtime.vlm.base import Embedding, LayerKV

DEFAULT_REPO = "lerobot/pi05_libero_finetuned"


@functools.cache
def _prefix_cache_type() -> type:
    """A read-only prefix cache for the expert (transformers imported lazily)."""
    from transformers import DynamicCache

    class PrefixCache(DynamicCache):
        """Prefix K/V that every denoise step reads and none extends.

        ``update`` returns prefix ++ suffix without storing it: the one copy
        attention needs, where ``DynamicCache`` plus LeRobot's per-step clone
        make three.
        """

        def __init__(self, layers: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> None:
            super().__init__()
            self._prefix = layers

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):  # noqa: ANN001
            key, value = self._prefix[layer_idx]
            return (
                torch.cat([key, key_states], dim=-2),
                torch.cat([value, value_states], dim=-2),
            )

        def get_seq_length(self, layer_idx: int = 0) -> int:
            return self._prefix[layer_idx][0].shape[-2]

    return PrefixCache


class PI05Model:
    """The loaded LeRobot π0.5 policy and its processors, shared by backbone and head.

    Args:
        repo: Hugging Face repo id or local checkpoint directory.
        device: Where the model runs.
        camera_map: Env camera name → LeRobot image feature key. Image features
            the map leaves out are fed as LeRobot's masked empty cameras.
        tokenizer: Local PaliGemma tokenizer directory; None keeps the
            checkpoint's tokenizer id.
        cuda_graph: Replay the unguided expert loop from CUDA Graphs (V4).
        vision_bf16: Run the vision tower and projector in bf16.
        expert_sdpa_min_batch: Batch size from which the expert attends with
            sdpa instead of eager; None keeps eager throughout.
    """

    def __init__(
        self,
        repo: str,
        device: torch.device,
        camera_map: Mapping[str, str],
        tokenizer: str | None = None,
        cuda_graph: bool = False,
        vision_bf16: bool = False,
        expert_sdpa_min_batch: int | None = None,
    ) -> None:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05 import modeling_pi05

        self.repo = repo
        self.device = device
        self.camera_map = dict(camera_map)
        self._make_att_2d_masks = modeling_pi05.make_att_2d_masks
        self._prepare_attention_masks_4d = modeling_pi05.prepare_attention_masks_4d
        self._time_embedding = modeling_pi05.create_sinusoidal_pos_embedding
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
        self.vision_bf16 = vision_bf16
        if vision_bf16:
            _vision_to_bf16(self.policy.model.paligemma_with_expert)
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
        # embed_suffix builds this mask from a Python list on every step: a
        # host copy, which a CUDA Graph cannot capture.
        self._suffix_att = torch.tensor(
            [1] + [0] * (self.config.chunk_size - 1),
            dtype=self.policy.model.action_in_proj.weight.dtype,
            device=device,
        )
        self.expert_sdpa_min_batch = expert_sdpa_min_batch
        self._graphs = GraphCache(self._graph_loop) if cuda_graph else None

    @property
    def cuda_graph(self) -> bool:
        return self._graphs is not None

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

    def _prefix_layers(
        self, embeddings: Sequence[Embedding], clone: bool
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Per-row conditions → per-layer prefix K/V [B, H, L, D] and pad masks [B, L]."""
        for e in embeddings:
            if any(sw is not None for sw in e.metadata["sliding_windows"]):
                raise ValueError("pi05 prefix cache has sliding-window layers; not supported")
        return stack_prefix_kv(embeddings, clone)

    def _velocity(
        self,
        pad_masks: torch.Tensor,
        layers: Sequence[tuple[torch.Tensor, torch.Tensor]],
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """``PI05Pytorch.denoise_step`` without host copies or per-step cache clones."""
        model = self.policy.model
        config = self.config
        time_emb = self._time_embedding(
            timestep,
            model.action_in_proj.out_features,
            min_period=config.min_period,
            max_period=config.max_period,
            device=timestep.device,
        ).type(dtype=timestep.dtype)
        action_emb = model.action_in_proj(x_t)
        adarms_cond = F.silu(model.time_mlp_out(F.silu(model.time_mlp_in(time_emb))))

        batch_size, suffix_len = action_emb.shape[:2]
        prefix_len = pad_masks.shape[1]
        suffix_pad = torch.ones(batch_size, suffix_len, dtype=torch.bool, device=timestep.device)
        suffix_att = self._suffix_att[None, :].expand(batch_size, suffix_len)
        full_att_2d = torch.cat(
            [
                pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len),
                self._make_att_2d_masks(suffix_pad, suffix_att),
            ],
            dim=2,
        )
        position_ids = torch.sum(pad_masks, dim=-1)[:, None] + torch.cumsum(suffix_pad, dim=1) - 1
        # The choice depends only on the batch size, so each captured graph has one.
        sdpa = self.expert_sdpa_min_batch is not None and batch_size >= self.expert_sdpa_min_batch
        model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = (
            "sdpa" if sdpa else "eager"
        )
        # sdpa rejects the fp32 additive mask eager takes; it reads the boolean one.
        attention_mask = (
            full_att_2d[:, None] if sdpa else self._prepare_attention_masks_4d(full_att_2d)
        )
        outputs, _ = model.paligemma_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=_prefix_cache_type()(layers),
            inputs_embeds=[None, action_emb],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs[1][:, -config.chunk_size :].to(dtype=torch.float32)
        return model.action_out_proj(suffix_out)

    def _graph_loop(
        self, pad_masks: torch.Tensor, noise: torch.Tensor, *kv: torch.Tensor
    ) -> torch.Tensor:
        """The unguided Euler loop over flat K/V inputs, as ``GraphCache`` captures it."""
        layers = list(zip(kv[0::2], kv[1::2], strict=True))
        return euler_denoise(
            lambda x, t: self._velocity(pad_masks, layers, x, t), noise, self.num_steps, None, None
        )

    def denoise(
        self,
        embedding: Embedding,
        noise: torch.Tensor | None = None,
        guidance: RTCGuidance | None = None,
    ) -> torch.Tensor:
        """Action expert Euler loop over a cached prefix → [chunk_size, action_dim] (CPU)."""
        return self.denoise_batch([embedding], noise, [guidance])[0]

    def denoise_batch(
        self,
        embeddings: Sequence[Embedding],
        noise: torch.Tensor | None = None,
        guidances: Sequence[RTCGuidance | None] | None = None,
    ) -> list[torch.Tensor]:
        """``denoise`` for several rows at once (ADR-022) → one CPU tensor per row."""
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
        layers, pad_masks = self._prefix_layers(embeddings, clone=detach_cache)
        device = pad_masks.device
        shape = (len(embeddings), config.chunk_size, config.max_action_dim)
        if noise is None:
            noise = model.sample_noise(shape, device)
        target, guidance = batch_guidance(
            guidances, self.normalize_actions, shape, self.action_dim, device
        )
        if self._graphs is not None and guidance is None:
            x_t = self._graphs(pad_masks, noise, *(t for kv in layers for t in kv))
        else:
            x_t = euler_denoise(
                lambda x, t: self._velocity(pad_masks, layers, x, t),
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
        return self._postprocess(actions)[0]

    def _postprocess(self, actions: torch.Tensor) -> list[torch.Tensor]:
        wait_for_gpu(actions)
        out = self.post(actions)
        return list(out.float().cpu().unbind(0))


def _vision_to_bf16(paligemma_with_expert: Any) -> None:
    """Cast SigLIP + projector to bf16 and feed them bf16 images.

    ``PaliGemmaWithExpertModel.embed_image`` casts every image to fp32 for the
    fp32 tower; this instance's replacement casts to bf16 and returns the
    features in the caller's dtype, as the original does.
    """
    model = paligemma_with_expert.paligemma.model
    model.vision_tower.to(torch.bfloat16)
    model.multi_modal_projector.to(torch.bfloat16)

    def embed_image(self: Any, image: torch.Tensor) -> torch.Tensor:
        features = self.paligemma.model.get_image_features(image.to(torch.bfloat16))
        return features.pooler_output.to(image.dtype)

    paligemma_with_expert.embed_image = types.MethodType(embed_image, paligemma_with_expert)


def _flag(opts: dict[str, str], key: str) -> bool:
    return opts.pop(key, "0").lower() in ("1", "true", "yes")


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
    cuda_graph = _flag(opts, "cuda_graph")
    vision_bf16 = _flag(opts, "vision_bf16")
    expert_sdpa = int(opts.pop("expert_sdpa", "0"))
    if opts:
        raise ValueError(f"unknown --policy-opt for pi05: {sorted(opts)}")
    check_cameras(cameras, spec)

    model = PI05Model(
        repo,
        device,
        cameras,
        tokenizer=tokenizer,
        cuda_graph=cuda_graph,
        vision_bf16=vision_bf16,
        expert_sdpa_min_batch=expert_sdpa or None,
    )
    return wrap_prefix_kv_policy(
        model,
        spec,
        vlm_budget_ms=vlm_budget_ms,
        rtc=rtc,
        rtc_config=rtc_config,
        n_action_steps=int(n_action_steps) if n_action_steps else None,
        info={
            "device": str(device),
            "cameras": cameras,
            "cuda_graph": cuda_graph,
            "vision_bf16": vision_bf16,
            "expert_sdpa": expert_sdpa,
        },
    )
