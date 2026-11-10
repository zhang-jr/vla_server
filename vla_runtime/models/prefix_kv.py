"""Prefix-KV split of a VLA: the VLM prefix → per-layer K/V → the action expert.

π0.5 and SmolVLA are each one network: a VLM runs over the prefix (camera
views, instruction and, for these two models, the robot state), and a
flow-matching action expert attends over every layer's prefix K/V while it
denoises. This module is the model-agnostic half of that split:

- ``PrefixKVBackbone`` (VLMBackend): runs the model's prefix pass and returns
  its per-layer K/V as a ``per_layer_kv`` condition (ADR-019).
- ``PrefixKVExpertHead`` (ActionHeadRuntime, plus WeightSharingHead,
  RealTimeChunkingHead and BatchedActionHead): runs the expert's denoising
  loop over that condition, for one robot or a batch of robots (ADR-022),
  with optional RTC guidance (ADR-015).
- ``euler_denoise``, ``stack_prefix_kv``, ``batch_guidance``: the loop and the
  batching helpers both model adapters share.
- ``prefix_kv_policy``: a loaded model → a validated ``Policy`` (prefix state,
  per_layer_kv, iterative → coupled, ADR-018).

The model-specific half implements ``PrefixKVModel``:
``vla_runtime.models.smolvla.SmolVLAModel`` and ``vla_runtime.models.pi05.PI05Model``.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

import torch

from vla_runtime.action_head.base import ActionChunk, ActionHeadContext, ActionHeadType, RobotState
from vla_runtime.action_head.rtc import RTCConfig, RTCGuidance, guided_velocity, prepare_guidance
from vla_runtime.capabilities import PolicyCapabilities
from vla_runtime.errors import ActionHeadFailureError, BackendError, ErrorContext
from vla_runtime.observability.tracer import get_tracer
from vla_runtime.policy import Policy
from vla_runtime.vlm.base import Embedding, TokenStream

_tracer = get_tracer("models.prefix_kv")


def wait_for_gpu(tensor: torch.Tensor) -> None:
    """Block until ``tensor``'s stream is done, with the GIL released.

    A blocking device-to-host copy waits for the GPU while holding the GIL:
    a graphed expert pass would stall every other Python thread (the VLM
    stage, env workers, the event loop) for its whole 33-72 ms.
    """
    if tensor.is_cuda:
        torch.cuda.current_stream(tensor.device).synchronize()


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


class PrefixKVModel(Protocol):
    """What the generic backbone / head need from a LeRobot prefix-KV adapter.

    Implemented by ``vla_runtime.models.smolvla.SmolVLAModel`` and
    ``vla_runtime.models.pi05.PI05Model``.
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
        context = ErrorContext(trace_id=_tracer.trace_id, module_name="models.prefix_kv")
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
            trace_id=_tracer.trace_id, module_name="models.prefix_kv.head"
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
            trace_id=_tracer.trace_id, module_name="models.prefix_kv.head"
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


def prefix_kv_policy(
    model: PrefixKVModel,
    *,
    vlm_budget_ms: float,
    frequency_hz: float,
    rtc: bool = False,
    rtc_config: RTCConfig | None = None,
) -> tuple[Policy, PrefixKVBackbone]:
    """A loaded prefix-KV model → Policy (prefix state, per_layer_kv, iterative).

    Args:
        model: The loaded model both halves share.
        vlm_budget_ms: Backbone latency budget reported to the scheduler.
        frequency_hz: The head's nominal inference rate (control rate divided
            by the actions executed per chunk).
        rtc: Declare ``supports_rtc`` so schedulers pass RTC contexts.
        rtc_config: Guidance parameters; defaults to ``RTCConfig()``.

    Returns:
        (policy, backbone). Call ``backbone.close()`` when done with the
        policy; it owns the backbone's GPU thread.
    """
    backbone = PrefixKVBackbone(model, latency_budget_ms=vlm_budget_ms)
    head = PrefixKVExpertHead(model, frequency_hz=frequency_hz, rtc_config=rtc_config)
    caps = PolicyCapabilities(
        state_placement="prefix",
        condition_kind="per_layer_kv",
        head_structure="iterative",
        supports_rtc=rtc,
    )
    name = model.model_id + ("+rtc" if rtc else "")
    policy = Policy(name=name, capabilities=caps, vlm_backend=backbone, head_prototype=head)
    return policy, backbone
