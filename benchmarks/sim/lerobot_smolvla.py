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

Built against LeRobot's processor API (``make_pre_post_processors``, 0.4+).
It relies on these SmolVLA internals, all used verbatim by ``sample_actions``:
``prepare_images``, ``prepare_state``, ``model.embed_prefix``,
``model.vlm_with_expert.forward(fill_kv_cache=True)``, ``model.denoise_step``,
``model.sample_noise`` and ``make_att_2d_masks``. If a LeRobot upgrade renames
any of them, the alignment test fails loudly rather than the benchmark
silently measuring a different computation.

Policy options (``--policy-opt key=value``):
    repo             checkpoint (default HuggingFaceVLA/smolvla_libero)
    device           default cuda
    cameras          env_cam=lerobot_key pairs joined by ',' (default: LIBERO
                     agentview→observation.images.image, wrist→observation.images.image2)
    n_action_steps   override the checkpoint's value
    vlm_budget_ms    backbone latency budget reported to the scheduler
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

from benchmarks.sim.envs import EnvSpec, Observation
from benchmarks.sim.policies import BenchPolicy
from vla_runtime.action_head.base import ActionChunk, ActionHeadType, RobotState
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
    """

    def __init__(self, repo: str, device: torch.device, camera_map: Mapping[str, str]) -> None:
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import (
            SmolVLAPolicy,
            make_att_2d_masks,
        )

        self.repo = repo
        self.device = device
        self.camera_map = dict(camera_map)
        self.policy = SmolVLAPolicy.from_pretrained(repo)
        self.policy.to(device)
        self.policy.eval()
        self.config = self.policy.config
        self.make_att_2d_masks = make_att_2d_masks
        self.pre, self.post = make_pre_post_processors(
            self.policy.config,
            pretrained_path=repo,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
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
    def action_dim(self) -> int:
        return int(self.config.action_feature.shape[0])

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
        _, past_key_values = model.vlm_with_expert.forward(
            attention_mask=att_2d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        prefix_len = pad_masks.shape[1]
        layer_indices = sorted(past_key_values)
        layout = _kv_layout(past_key_values[layer_indices[0]]["key_states"], prefix_len)
        layers = tuple(
            LayerKV(
                key=_heads_first(past_key_values[i]["key_states"], layout),
                value=_heads_first(past_key_values[i]["value_states"], layout),
            )
            for i in layer_indices
        )
        return Embedding(
            kind="per_layer_kv",
            layer_kv=layers,
            attention_mask=pad_masks[0].bool(),
            model_id=f"smolvla:{self.repo}",
            metadata={"layer_indices": layer_indices, "kv_layout": layout},
        )

    @torch.inference_mode()
    def denoise(self, embedding: Embedding, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Action expert Euler loop over a cached prefix → [chunk_size, action_dim] (CPU)."""
        model = self.policy.model
        config = self.config
        layout = embedding.metadata["kv_layout"]
        past_key_values = {
            i: {
                "key_states": _batch_first(kv.key, layout),
                "value_states": _batch_first(kv.value, layout),
            }
            for i, kv in zip(embedding.metadata["layer_indices"], embedding.layer_kv)
        }
        pad_masks = embedding.attention_mask.unsqueeze(0)
        device = pad_masks.device
        if noise is None:
            noise = model.sample_noise((1, config.chunk_size, config.max_action_dim), device)
        dt = torch.tensor(-1.0 / config.num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            v_t = model.denoise_step(pad_masks, past_key_values, x_t, time.expand(1))
            x_t = x_t + dt * v_t
            time = time + dt
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
        return self._postprocess(actions)

    def _postprocess(self, actions: torch.Tensor) -> torch.Tensor:
        out = self.post(actions)
        return out[0].float().cpu()


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


def _policy_state(state: RobotState | None, context: ErrorContext) -> torch.Tensor:
    if state is None or "policy_state" not in state.extra:
        raise BackendError(
            "SmolVLA reads robot state in its prefix but encode() got no "
            "robot_state.extra['policy_state']; the env must provide it",
            context,
        )
    return state.extra["policy_state"]


class SmolVLABackbone:
    """VLMBackend over SmolVLA's VLM: prefix → per-layer K/V.

    GPU work runs on one dedicated thread so the event loop never blocks, and
    concurrent fleet requests are serialized on the shared weights.
    """

    def __init__(self, model: SmolVLAModel, latency_budget_ms: float) -> None:
        self._model = model
        self._budget_ms = latency_budget_ms
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="smolvla-vlm")

    @property
    def model_id(self) -> str:
        return f"smolvla:{self._model.repo}"

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
        context = ErrorContext(trace_id=_tracer.trace_id, module_name="benchmarks.smolvla")
        policy_state = _policy_state(state, context)
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
        raise NotImplementedError("SmolVLA has no discrete action tokens")

    async def warmup(self) -> None:
        pass

    async def shutdown(self) -> None:
        # Schedulers call this on stop(); the model outlives them within a run.
        pass

    def close(self) -> None:
        self._executor.shutdown(wait=True)


class SmolVLAExpertHead:
    """ActionHeadRuntime over SmolVLA's action expert (flow matching, iterative).

    Instances spawned for different robots share the one loaded model; the
    expert keeps no state between calls, so nothing per robot needs copying.
    """

    accepted_condition_kinds: tuple[str, ...] = ("per_layer_kv",)

    def __init__(self, model: SmolVLAModel, frequency_hz: float = 10.0) -> None:
        self._model = model
        self._frequency_hz = frequency_hz

    def spawn(self) -> SmolVLAExpertHead:
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

    def predict(self, embedding: Embedding, robot_state: RobotState) -> ActionChunk:
        context = ErrorContext(trace_id=_tracer.trace_id, module_name="benchmarks.smolvla.head")
        try:
            actions = self._model.denoise(embedding)
        except Exception as exc:
            raise ActionHeadFailureError("flow_matching", exc, context) from exc
        return ActionChunk(
            actions=actions,
            head_type="flow_matching",
            metadata={"n_steps": int(self._model.config.num_steps)},
        )

    def load_weights(self, checkpoint_path: str) -> None:
        raise NotImplementedError("weights come from the LeRobot checkpoint; rebuild the adapter")

    def to_device(self, device: torch.device) -> None:
        self._model.policy.to(device)


def build_smolvla(spec: EnvSpec, options: dict[str, str]) -> BenchPolicy:
    """Load SmolVLA and wrap it as a BenchPolicy (see module docstring for options)."""
    opts = dict(options)
    repo = opts.pop("repo", DEFAULT_REPO)
    device = torch.device(opts.pop("device", "cuda"))
    cameras = parse_camera_map(opts.pop("cameras")) if "cameras" in opts else dict(LIBERO_CAMERAS)
    n_action_steps = opts.pop("n_action_steps", None)
    vlm_budget_ms = float(opts.pop("vlm_budget_ms", "2000"))
    if opts:
        raise ValueError(f"unknown --policy-opt for smolvla: {sorted(opts)}")
    missing = set(cameras) - set(spec.camera_names)
    if missing:
        raise ValueError(
            f"camera map uses {sorted(missing)} but {spec.name} provides "
            f"{list(spec.camera_names)}; set --policy-opt cameras=env_cam=lerobot_key,..."
        )

    model = SmolVLAModel(repo, device, cameras)
    if model.action_dim != spec.action_dim:
        raise ValueError(
            f"{repo} outputs {model.action_dim}-D actions but {spec.name} takes {spec.action_dim}-D"
        )
    backbone = SmolVLABackbone(model, latency_budget_ms=vlm_budget_ms)
    head = SmolVLAExpertHead(model, frequency_hz=spec.control_hz / max(1, int(model.config.n_action_steps)))
    caps = PolicyCapabilities(
        state_placement="prefix",
        condition_kind="per_layer_kv",
        head_structure="iterative",
    )
    policy = Policy(name=f"smolvla:{repo}", capabilities=caps, vlm_backend=backbone, head_prototype=head)

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

    steps = int(n_action_steps) if n_action_steps else int(model.config.n_action_steps)
    return BenchPolicy(
        name=f"smolvla:{repo}",
        policy=policy,
        n_action_steps=steps,
        native=native,
        info={
            "repo": repo,
            "device": str(device),
            "chunk_size": int(model.config.chunk_size),
            "n_action_steps": steps,
            "num_denoise_steps": int(model.config.num_steps),
            "cameras": cameras,
        },
        close=close,
    )
