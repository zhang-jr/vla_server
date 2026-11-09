"""SmolVLA (LeRobot) split at the VLM → action expert boundary.

The first real model adapter: ``SmolVLABackbone.encode()`` runs the VLM over
the prefix (images + language + state) and returns its per-layer K/V as a
``per_layer_kv`` condition (ADR-019); ``SmolVLAExpertHead.predict()`` runs the
flow-matching denoising loop of the action expert over that cache. Together
they reproduce ``VLAFlowMatching.sample_actions`` step for step, which
``benchmarks/benchmark_sim.py::test_smolvla_split_matches_native`` checks
numerically against LeRobot's own ``predict_action_chunk`` with shared noise.

Capabilities: SmolVLA feeds state into the prefix → ``state_placement="prefix"``
→ coupled frequency (ADR-018). The scheduler therefore passes state to
encode(), and the head never re-runs on a cached condition.

RTC (ADR-015, ``--policy-opt rtc=1``): the expert's Euler loop runs the
runtime's guidance (``vla_runtime.action_head.rtc``). SmolVLA integrates time
from 1 (noise) to 0, so each step hands the guidance ``tau = 1 - time`` and the
sign-flipped velocity. The previous actions arrive in the executed (env) space
and are normalized with the checkpoint's own action statistics before guiding.
``benchmarks/benchmark_sim.py`` checks the ``identity`` mode against LeRobot's
built-in ``RTCProcessor`` with shared noise.

Head batching (ADR-022): ``denoise_batch`` runs the expert over several robots'
conditions at once — per-layer K/V stacked along the batch, shorter prefixes
right-padded and masked out, one noise / RTC row each. ``denoise`` is the
one-row case of the same code, so the native-alignment tests cover it too.

The model-agnostic half — ``PrefixKVBackbone``, ``PrefixKVExpertHead``, the
``euler_denoise`` loop and ``wrap_prefix_kv_policy`` — is shared with the π0.5
adapter (``benchmarks/sim/lerobot_pi05.py``); only ``SmolVLAModel`` is SmolVLA-specific.

Built against LeRobot's processor API (``make_pre_post_processors``, 0.4+).
It relies on these SmolVLA internals, all used verbatim by ``sample_actions``:
``prepare_images``, ``prepare_state``, ``model.embed_prefix``,
``model.vlm_with_expert.forward``, ``model.denoise_step``,
``model.sample_noise`` and ``make_att_2d_masks``. Two prefix-cache layouts are
handled: LeRobot < 0.6 passes ``fill_kv_cache=True`` and returns a
``{layer: {"key_states", "value_states"}}`` dict ([B, L, H, D]); 0.6+ fills a
transformers ``DynamicCache`` ([B, H, L, D]) that ``denoise_step`` appends to
and crops back in place, so a fresh one is rebuilt for every denoise call. If a LeRobot upgrade renames
any of them, the alignment test fails loudly rather than the benchmark
silently measuring a different computation.

Policy options (``--policy-opt key=value``):
    repo             checkpoint (default HuggingFaceVLA/smolvla_libero)
    device           default cuda
    cameras          env_cam=lerobot_key pairs joined by ',' (default: LIBERO
                     agentview→observation.images.image, wrist→observation.images.image2)
    n_action_steps   override the checkpoint's value
    vlm_budget_ms    backbone latency budget reported to the scheduler
    vlm              local SmolVLM2 directory replacing the checkpoint's
                     ``vlm_model_name`` (backbone config + tokenizer), for
                     offline hosts, e.g. /models/SmolVLM2-500M-Video-Instruct
    rtc              1 to declare supports_rtc and guide the expert (default 0)
    rtc_beta         max guidance weight (default RTCConfig: 5)
    rtc_schedule     zeros / exp / linear / ones (default zeros)
    rtc_jacobian     identity / vjp (default identity)
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import inspect
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

import torch

from benchmarks.sim.envs import EnvSpec, Observation
from benchmarks.sim.policies import BenchPolicy
from vla_runtime.action_head.base import ActionChunk, ActionHeadContext, ActionHeadType, RobotState
from vla_runtime.action_head.rtc import RTCConfig, RTCGuidance, guided_velocity, prepare_guidance
from vla_runtime.capabilities import PolicyCapabilities
from vla_runtime.errors import ActionHeadFailureError, BackendError, ErrorContext
from vla_runtime.observability.tracer import get_tracer
from vla_runtime.policy import Policy
from vla_runtime.vlm.base import Embedding, LayerKV, TokenStream

_tracer = get_tracer("benchmarks.smolvla")

try:
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
except ImportError:  # older layout / constants moved
    OBS_LANGUAGE_TOKENS = "observation.language.tokens"
    OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"

OBS_STATE = "observation.state"
DEFAULT_REPO = "HuggingFaceVLA/smolvla_libero"
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
        out = self.post(actions)
        return list(out.float().cpu().unbind(0))


def stack_prefix_kv(
    embeddings: Sequence[Embedding], clone: bool = False
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Per-row per_layer_kv conditions → per-layer ``[B, H, L, D]`` K/V and ``[B, L]`` pad mask.

    Rows shorter than the longest prefix are right-padded with zeros and
    masked out, so the expert attends to exactly the positions it would see
    row by row. One row is passed through without padding; ``clone`` copies
    it (stacking already copies several rows).

    Raises:
        ValueError: If the rows disagree on layer count or K/V shape.
    """
    first = embeddings[0]
    num_layers = len(first.layer_kv)
    longest = max(e.attention_mask.shape[0] for e in embeddings)
    for e in embeddings:
        if len(e.layer_kv) != num_layers or any(
            kv.key.shape[0::2] != ref.key.shape[0::2] for kv, ref in zip(e.layer_kv, first.layer_kv)
        ):
            raise ValueError("batched conditions disagree on layer count or K/V shape")

    def rows(get: Callable[[Embedding], torch.Tensor], pad_dim: int, fill: Any) -> torch.Tensor:
        if len(embeddings) == 1:
            t = get(first)
            return (t.clone() if clone else t).unsqueeze(0)
        padded = []
        for e in embeddings:
            t = get(e)
            missing = longest - t.shape[pad_dim]
            if missing:
                pad_shape = list(t.shape)
                pad_shape[pad_dim] = missing
                t = torch.cat([t, t.new_full(pad_shape, fill)], dim=pad_dim)
            padded.append(t)
        return torch.stack(padded)

    layers = [
        (
            rows(lambda e, i=i: e.layer_kv[i].key, 1, 0),
            rows(lambda e, i=i: e.layer_kv[i].value, 1, 0),
        )
        for i in range(num_layers)
    ]
    return layers, rows(lambda e: e.attention_mask, 0, False)


def batch_guidance(
    guidances: Sequence[RTCGuidance | None],
    normalize: Callable[[torch.Tensor], torch.Tensor],
    shape: tuple[int, int, int],
    action_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, RTCGuidance | None]:
    """Per-row RTC guidance → the batched target and weights ``euler_denoise`` takes.

    Unguided rows get zero weights, which leaves their velocity untouched, so
    guided and unguided rows can share a batch.

    Args:
        guidances: One guidance (or None) per row.
        normalize: Executed-space actions → the model's normalized space.
        shape: ``(B, chunk_size, max_action_dim)``.
        action_dim: Leading action columns the target fills.
        device: Where the target lives.

    Returns:
        (normalized target ``[B, chunk, max_action_dim]``, guidance with
        ``[B, chunk]`` weights), or (None, None) when no row is guided.

    Raises:
        ValueError: If guided rows disagree on beta or the Jacobian mode.
    """
    active = [g for g in guidances if g is not None]
    if not active:
        return None, None
    first = active[0]
    if any(
        (g.max_guidance_weight, g.jacobian) != (first.max_guidance_weight, first.jacobian)
        for g in active
    ):
        raise ValueError("batched RTC rows disagree on max_guidance_weight or jacobian")
    target = torch.zeros(shape, device=device)
    weights = torch.zeros(shape[0], shape[1])
    for row, g in enumerate(guidances):
        if g is not None:
            target[row, :, :action_dim] = normalize(g.target)
            weights[row] = g.weights
    return target, dataclasses.replace(first, target=target, weights=weights)


def euler_denoise(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    noise: torch.Tensor,
    num_steps: int,
    target: torch.Tensor | None,
    guidance: RTCGuidance | None,
) -> torch.Tensor:
    """LeRobot / openpi Euler loop from time 1 (noise) to 0, optionally RTC-guided.

    Same schedule as LeRobot's ``euler_integrate``: time is recomputed per
    step, not accumulated, so a split adapter reproduces it bit for bit.

    Args:
        velocity_fn: ``(x_t, time[1]) → v_t`` in the model's convention.
        noise: ``x`` at time 1, ``[B, chunk_size, max_action_dim]``.
        num_steps: Euler steps.
        target: Normalized, padded previous actions ``[B, chunk, max_action_dim]``;
            required with ``guidance``.
        guidance: RTC guidance with ``[B, chunk]`` weights, or None for the plain loop.
    """
    device = noise.device
    dt = -1.0 / num_steps
    x_t = noise
    for step in range(num_steps):
        time = 1.0 + step * dt
        # A fill, not a host copy: the unguided loop must be CUDA Graph capturable.
        step_time = torch.full((noise.shape[0],), time, dtype=torch.float32, device=device)
        if guidance is None:
            v_t = velocity_fn(x_t, step_time)
        else:
            # Paper convention: tau = 1 - time, velocity points noise → data.
            v_t = -guided_velocity(
                lambda x, t=step_time: -velocity_fn(x, t),
                x_t,
                1.0 - time,
                target,
                guidance.weights,
                guidance.max_guidance_weight,
                guidance.jacobian,
            )
        x_t = x_t + dt * v_t
    return x_t


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


class PrefixKVModel(Protocol):
    """What the generic backbone / head need from a LeRobot prefix-KV adapter.

    Implemented by ``SmolVLAModel`` and ``benchmarks.sim.lerobot_pi05.PI05Model``.
    """

    repo: str
    policy: Any
    config: Any

    @property
    def model_id(self) -> str: ...

    @property
    def num_steps(self) -> int: ...

    @property
    def action_dim(self) -> int: ...

    def encode_prefix(
        self, images: Mapping[str, torch.Tensor], text: str, policy_state: torch.Tensor
    ) -> Embedding: ...

    def denoise(
        self,
        embedding: Embedding,
        noise: torch.Tensor | None = None,
        guidance: RTCGuidance | None = None,
    ) -> torch.Tensor: ...

    def denoise_batch(
        self,
        embeddings: Sequence[Embedding],
        noise: torch.Tensor | None = None,
        guidances: Sequence[RTCGuidance | None] | None = None,
    ) -> list[torch.Tensor]: ...

    def native_chunk(
        self,
        images: Mapping[str, torch.Tensor],
        text: str,
        policy_state: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


def _policy_state(
    state: RobotState | None, context: ErrorContext, model_id: str
) -> torch.Tensor:
    if state is None or "policy_state" not in state.extra:
        raise BackendError(
            f"{model_id} reads robot state in its prefix but encode() got no "
            "robot_state.extra['policy_state']; the env must provide it",
            context,
        )
    return state.extra["policy_state"]


class PrefixKVBackbone:
    """VLMBackend over a LeRobot VLM prefix: images + text + state → per-layer K/V.

    GPU work runs on one dedicated thread so the event loop never blocks, and
    concurrent fleet requests are serialized on the shared weights.
    """

    def __init__(self, model: PrefixKVModel, latency_budget_ms: float) -> None:
        self._model = model
        self._budget_ms = latency_budget_ms
        family = model.model_id.split(":", 1)[0]
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{family}-vlm")

    @property
    def model_id(self) -> str:
        return self._model.model_id

    @property
    def latency_budget_ms(self) -> float:
        return self._budget_ms

    @property
    def kv_cache_enabled(self) -> bool:
        return True

    async def encode(
        self,
        images: Mapping[str, torch.Tensor],
        text: str,
        state: RobotState | None = None,
    ) -> Embedding:
        context = ErrorContext(trace_id=_tracer.trace_id, module_name="benchmarks.prefix_kv")
        policy_state = _policy_state(state, context, self.model_id)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self._model.encode_prefix, images, text, policy_state
        )

    async def generate(
        self,
        images: Mapping[str, torch.Tensor],
        text: str,
        state: RobotState | None = None,
    ) -> TokenStream:
        raise NotImplementedError(f"{self.model_id} has no discrete action tokens")

    async def warmup(self) -> None:
        pass

    async def shutdown(self) -> None:
        # Schedulers call this on stop(); the model outlives them within a run.
        pass

    def close(self) -> None:
        self._executor.shutdown(wait=True)


class PrefixKVExpertHead:
    """ActionHeadRuntime over a LeRobot action expert (flow matching, iterative).

    Instances spawned for different robots share the one loaded model; the
    expert keeps no state between calls, so nothing per robot needs copying
    and any instance can serve a batch of robots (``predict_batch``, ADR-022).
    """

    accepted_condition_kinds: tuple[str, ...] = ("per_layer_kv",)
    supports_rtc: bool = True

    def __init__(
        self,
        model: PrefixKVModel,
        frequency_hz: float = 10.0,
        rtc_config: RTCConfig | None = None,
    ) -> None:
        self._model = model
        self._frequency_hz = frequency_hz
        self._rtc_config = rtc_config or RTCConfig()

    def spawn(self) -> PrefixKVExpertHead:
        return copy.copy(self)

    def shares_weights_with(self, other: object) -> bool:
        return getattr(other, "_model", None) is self._model

    @property
    def head_type(self) -> ActionHeadType:
        return "flow_matching"

    @property
    def frequency_hz(self) -> float:
        return self._frequency_hz

    @property
    def chunk_size(self) -> int:
        return int(self._model.config.chunk_size)

    @property
    def action_dim(self) -> int:
        return self._model.action_dim

    def predict(
        self,
        embedding: Embedding,
        robot_state: RobotState,
        context: ActionHeadContext | None = None,
    ) -> ActionChunk:
        error_context = ErrorContext(
            trace_id=_tracer.trace_id, module_name="benchmarks.prefix_kv.head"
        )
        try:
            guidance = prepare_guidance(
                context, self.chunk_size, self.action_dim, self._rtc_config
            )
            actions = self._model.denoise(embedding, guidance=guidance)
        except Exception as exc:
            raise ActionHeadFailureError("flow_matching", exc, error_context) from exc
        return self._chunk(actions, guidance)

    def predict_batch(
        self,
        embeddings: Sequence[Embedding],
        robot_states: Sequence[RobotState],
        contexts: Sequence[ActionHeadContext | None],
    ) -> list[ActionChunk]:
        """One expert pass over several robots' conditions (BatchedActionHead, ADR-022)."""
        error_context = ErrorContext(
            trace_id=_tracer.trace_id, module_name="benchmarks.prefix_kv.head"
        )
        try:
            guidances = [
                prepare_guidance(c, self.chunk_size, self.action_dim, self._rtc_config)
                for c in contexts
            ]
            rows = self._model.denoise_batch(embeddings, guidances=guidances)
        except Exception as exc:
            raise ActionHeadFailureError("flow_matching", exc, error_context) from exc
        return [self._chunk(actions, g) for actions, g in zip(rows, guidances, strict=True)]

    def _chunk(self, actions: torch.Tensor, guidance: RTCGuidance | None) -> ActionChunk:
        metadata: dict[str, Any] = {
            "n_steps": self._model.num_steps,
            "rtc": guidance is not None,
        }
        if guidance is not None:
            metadata["rtc_delay_steps"] = guidance.delay_steps
            metadata["rtc_overlap_steps"] = guidance.overlap_steps
        return ActionChunk(actions=actions, head_type="flow_matching", metadata=metadata)

    def load_weights(self, checkpoint_path: str) -> None:
        raise NotImplementedError("weights come from the LeRobot checkpoint; rebuild the adapter")

    def to_device(self, device: torch.device) -> None:
        self._model.policy.to(device)


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
    backbone = PrefixKVBackbone(model, latency_budget_ms=vlm_budget_ms)
    steps = n_action_steps or int(model.config.n_action_steps)
    head = PrefixKVExpertHead(
        model,
        frequency_hz=spec.control_hz / max(1, steps),
        rtc_config=rtc_config,
    )
    caps = PolicyCapabilities(
        state_placement="prefix",
        condition_kind="per_layer_kv",
        head_structure="iterative",
        supports_rtc=rtc,
    )
    name = model.model_id + ("+rtc" if rtc else "")
    policy = Policy(name=name, capabilities=caps, vlm_backend=backbone, head_prototype=head)

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
        name=name,
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


def build_smolvla(spec: EnvSpec, options: dict[str, str]) -> BenchPolicy:
    """Load SmolVLA and wrap it as a BenchPolicy (see module docstring for options)."""
    opts = dict(options)
    repo = opts.pop("repo", DEFAULT_REPO)
    device = torch.device(opts.pop("device", "cuda"))
    cameras = parse_camera_map(opts.pop("cameras")) if "cameras" in opts else dict(LIBERO_CAMERAS)
    n_action_steps = opts.pop("n_action_steps", None)
    vlm_budget_ms = float(opts.pop("vlm_budget_ms", "2000"))
    vlm = opts.pop("vlm", None)
    rtc, rtc_config = pop_rtc_options(opts)
    if opts:
        raise ValueError(f"unknown --policy-opt for smolvla: {sorted(opts)}")
    check_cameras(cameras, spec)

    model = SmolVLAModel(repo, device, cameras, vlm=vlm)
    return wrap_prefix_kv_policy(
        model,
        spec,
        vlm_budget_ms=vlm_budget_ms,
        rtc=rtc,
        rtc_config=rtc_config,
        n_action_steps=int(n_action_steps) if n_action_steps else None,
        info={"device": str(device), "cameras": cameras},
    )
