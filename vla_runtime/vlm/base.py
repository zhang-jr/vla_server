"""VLM Backend Protocol and shared data types.

All VLM adapters must implement VLMBackend. The Protocol is intentionally
minimal — implementations may expose extra methods, but callers (AsyncScheduler,
ActionHead) must only rely on what is defined here.

Data flow:
    ActionRequest → VLMBackend.encode() → Embedding → ActionHeadRuntime.predict()
    ActionRequest → VLMBackend.generate() → TokenStream  (discrete-token models only)

What crosses the VLM → action head boundary depends on the model (ADR-019):
a pooled vector, the prefix token sequence, per-layer prefix K/V (pi0-style
joint attention), or named projected features (GR00T-style). ``Embedding``
carries any of these, tagged with its ``kind``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal, Mapping, Optional, Protocol, runtime_checkable

import torch

from vla_runtime.action_head.base import RobotState

ConditionKind = Literal["pooled", "prefix_tokens", "per_layer_kv", "projected_features"]


# ── Shared data types ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LayerKV:
    """Cached prefix keys/values for one transformer layer.

    Attributes:
        key: [num_kv_heads, seq_len, head_dim].
        value: [num_kv_heads, seq_len, head_dim].
    """

    key: torch.Tensor
    value: torch.Tensor

    def __post_init__(self) -> None:
        if self.key.shape != self.value.shape:
            raise ValueError(
                f"LayerKV key/value shape mismatch: {tuple(self.key.shape)} "
                f"vs {tuple(self.value.shape)}"
            )
        if self.key.dim() != 3:
            raise ValueError(
                f"LayerKV expects [num_kv_heads, seq_len, head_dim], got "
                f"{tuple(self.key.shape)}"
            )

    @property
    def seq_len(self) -> int:
        return self.key.shape[1]

    @property
    def nbytes(self) -> int:
        return _tensor_nbytes(self.key) + _tensor_nbytes(self.value)


@dataclass
class Embedding:
    """Conditioning produced by VLMBackend.encode() for the action head.

    Consumed by ActionHeadRuntime.predict(); never shared across requests or
    robots. The name is kept for interface stability, but the payload is
    whatever the model's action head attends over — see ``kind``.

    Attributes:
        data: Hidden states. [hidden_dim] for ``pooled``, [seq_len, hidden_dim]
            for ``prefix_tokens``. Optional for the other kinds (e.g. the
            final-layer prefix states some adapters keep alongside K/V).
        kind: What the payload is. Inferred from ``data`` when omitted:
            1-D → ``pooled``, 2-D → ``prefix_tokens``.
        layer_kv: Per-layer prefix K/V, required for ``per_layer_kv``.
        features: Named conditioning tensors, required for
            ``projected_features`` (e.g. {"backbone": ..., "vl_proj": ...}).
        attention_mask: [seq_len] bool, True = visible prefix position.
            Heads must honour it; padded language tokens are masked out.
        created_at: Unix timestamp (seconds) at creation — used to decide
            whether a reused condition is still within the model's
            staleness tolerance.
        frame_id: Monotonically increasing counter per robot/request stream.
        model_id: Identifier of the VLM backend that produced this embedding.
        metadata: Backend-specific extras.

    Raises:
        ValueError: If the payload does not match ``kind``.
    """

    data: Optional[torch.Tensor] = None
    kind: Optional[ConditionKind] = None
    layer_kv: tuple[LayerKV, ...] = ()
    features: dict[str, torch.Tensor] = field(default_factory=dict)
    attention_mask: Optional[torch.Tensor] = None
    created_at: float = field(default_factory=time.time)
    frame_id: int = 0
    model_id: str = "unknown"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind is None:
            if self.data is None:
                raise ValueError("Embedding needs `kind` when `data` is not given")
            self.kind = "pooled" if self.data.dim() == 1 else "prefix_tokens"
        self.layer_kv = tuple(self.layer_kv)
        self._validate()

    def _validate(self) -> None:
        kind = self.kind
        if kind == "pooled":
            if self.data is None or self.data.dim() != 1:
                raise ValueError("pooled condition needs 1-D `data` [hidden_dim]")
        elif kind == "prefix_tokens":
            if self.data is None or self.data.dim() != 2:
                raise ValueError(
                    "prefix_tokens condition needs 2-D `data` [seq_len, hidden_dim]"
                )
        elif kind == "per_layer_kv":
            if not self.layer_kv:
                raise ValueError("per_layer_kv condition needs non-empty `layer_kv`")
            seq_lens = {kv.seq_len for kv in self.layer_kv}
            if len(seq_lens) != 1:
                raise ValueError(
                    f"per_layer_kv layers disagree on prefix length: {sorted(seq_lens)}"
                )
        elif kind == "projected_features":
            if not self.features:
                raise ValueError("projected_features condition needs non-empty `features`")
        else:
            raise ValueError(f"unknown condition kind {kind!r}")

        if self.attention_mask is not None:
            if self.attention_mask.dtype != torch.bool or self.attention_mask.dim() != 1:
                raise ValueError("attention_mask must be a 1-D bool tensor [seq_len]")
            seq_len = self.seq_len
            if seq_len is not None and self.attention_mask.shape[0] != seq_len:
                raise ValueError(
                    f"attention_mask length {self.attention_mask.shape[0]} "
                    f"!= prefix length {seq_len}"
                )

    @property
    def dim(self) -> int:
        """Hidden width of ``data``.

        Raises:
            ValueError: If this condition carries no ``data`` tensor.
        """
        if self.data is None:
            raise ValueError(f"{self.kind} condition has no `data`; `dim` is undefined")
        return self.data.shape[-1]

    @property
    def seq_len(self) -> Optional[int]:
        """Prefix length, or None for a pooled vector / unsequenced features."""
        if self.kind == "per_layer_kv":
            return self.layer_kv[0].seq_len
        if self.data is not None and self.data.dim() == 2:
            return self.data.shape[0]
        return None

    @property
    def num_layers(self) -> int:
        return len(self.layer_kv)

    @property
    def nbytes(self) -> int:
        """Payload size — what an edge-cloud split at this boundary must ship."""
        total = sum(kv.nbytes for kv in self.layer_kv)
        total += sum(_tensor_nbytes(t) for t in self.features.values())
        if self.data is not None:
            total += _tensor_nbytes(self.data)
        if self.attention_mask is not None:
            total += _tensor_nbytes(self.attention_mask)
        return total

    @property
    def age_ms(self) -> float:
        return (time.time() - self.created_at) * 1000.0

    def is_stale(self, max_age_ms: float) -> bool:
        return self.age_ms > max_age_ms


def _tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


@dataclass
class TokenChunk:
    """A batch of generated tokens from the VLM (discrete-token models only)."""

    token_ids: list[int]
    log_probs: Optional[list[float]] = None
    is_final: bool = False


# TokenStream is an async iterator of TokenChunk
TokenStream = AsyncIterator[TokenChunk]


# ── VLMBackend Protocol ───────────────────────────────────────────────────────

@runtime_checkable
class VLMBackend(Protocol):
    """Protocol that every VLM backend adapter must satisfy.

    Implementations:
        - vllm_backend.VLLMBackend
        - sglang_backend.SGLangBackend
        - llamacpp_backend.LlamaCppBackend  (edge)
        - lmdeploy_backend.LMDeployBackend

    All methods are async; the backend must not block the event loop.
    Robot state normally travels to ActionHeadRuntime as its own tensor channel.
    It reaches encode() only for models trained with state in the prefix
    (PolicyCapabilities.state_placement == "prefix"); the scheduler decides,
    backends never pull state in on their own.
    """

    @property
    def model_id(self) -> str:
        """Unique identifier for this backend / model combination."""
        ...

    @property
    def latency_budget_ms(self) -> float:
        """Maximum allowed encode/generate latency in milliseconds.

        The AsyncScheduler will trigger on_vlm_timeout if this is exceeded.
        """
        ...

    @property
    def kv_cache_enabled(self) -> bool:
        """Whether this backend supports KV cache reuse across frames."""
        ...

    async def encode(
        self,
        images: Mapping[str, torch.Tensor],
        text: str,
        state: Optional[RobotState] = None,
    ) -> Embedding:
        """Encode camera views + instruction (+ state, if prefix-state) into a condition.

        Args:
            images: Preprocessed views keyed by camera name, each [C, H, W].
                Iteration order is the order the model expects.
            text: Natural language instruction string.
            state: Robot state, passed only for prefix-state models.

        Returns:
            Embedding whose ``kind`` matches the model's condition_kind.

        Raises:
            VLMTimeoutError: If encoding exceeds latency_budget_ms.
            BackendError: For any other backend-level failure.
        """
        ...

    async def generate(
        self,
        images: Mapping[str, torch.Tensor],
        text: str,
        state: Optional[RobotState] = None,
    ) -> TokenStream:
        """Token-level generation for discrete-action-token models.

        Only implement for discrete-token VLA models (e.g. OpenVLA, RT-2).
        Flow matching / diffusion heads use encode() instead.

        Args:
            images: Preprocessed views keyed by camera name.
            text: Instruction string.
            state: Robot state, passed only for prefix-state models.

        Returns:
            Async iterator of TokenChunk; caller must drain the stream.
        """
        ...

    async def warmup(self) -> None:
        """Run a dummy forward pass to JIT-compile kernels.

        Should be called once before serving begins. Backends that do not
        need warmup should implement as a no-op.
        """
        ...

    async def shutdown(self) -> None:
        """Release GPU memory and background threads gracefully."""
        ...


# ── ActionRequest ─────────────────────────────────────────────────────────────

@dataclass
class ActionRequest:
    """Input to the VLA inference pipeline — one observation of one robot.

    Attributes:
        request_id: Unique ID for this request (caller-assigned or auto-generated).
        images: Preprocessed views keyed by camera name, each [C, H, W]. VLA
            policies commonly take several (e.g. base + wrist); insertion order
            is the order handed to the backbone.
        instruction: Natural language task instruction.
        robot_state: Structured proprioception. Where it is consumed (head or
            backbone prefix) is a property of the model, not of the request.
        frame_id: Monotonically increasing frame counter for this robot stream.
        timestamp: Unix time (seconds) when this request was created.

    Raises:
        ValueError: If no camera view is given.
    """

    request_id: str
    images: dict[str, torch.Tensor]
    instruction: str
    robot_state: RobotState
    frame_id: int = 0
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.images:
            raise ValueError("ActionRequest needs at least one camera view")
