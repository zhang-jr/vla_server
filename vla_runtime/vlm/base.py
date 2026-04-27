"""VLM Backend Protocol and shared data types.

All VLM adapters must implement VLMBackend. The Protocol is intentionally
minimal — implementations may expose extra methods, but callers (AsyncScheduler,
ActionHead) must only rely on what is defined here.

Data flow:
    ActionRequest → VLMBackend.encode() → Embedding → ActionHeadRuntime.predict()
    ActionRequest → VLMBackend.generate() → TokenStream  (discrete-token models only)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional, Protocol, runtime_checkable

import torch


# ── Shared data types ─────────────────────────────────────────────────────────

@dataclass
class Embedding:
    """Shared embedding produced by VLMBackend.encode().

    Consumed by ActionHeadRuntime.predict(); never shared across requests.

    Attributes:
        data: Feature tensor, shape [seq_len, hidden_dim] or [hidden_dim].
        created_at: Unix timestamp (seconds) at creation — used to detect
            stale embeddings in fallback paths.
        frame_id: Monotonically increasing counter per robot/request stream.
        model_id: Identifier of the VLM backend that produced this embedding.
        metadata: Optional dict for backend-specific extras (e.g. attention mask).
    """

    data: torch.Tensor
    created_at: float = field(default_factory=time.time)
    frame_id: int = 0
    model_id: str = "unknown"
    metadata: dict = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return self.data.shape[-1]

    @property
    def age_ms(self) -> float:
        return (time.time() - self.created_at) * 1000.0

    def is_stale(self, max_age_ms: float) -> bool:
        return self.age_ms > max_age_ms


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
    Robot state is passed as a separate tensor through ActionHeadRuntime —
    never embed robot state in the prompt (Anti-Pattern in CLAUDE.md).
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

    async def encode(self, image: torch.Tensor, text: str) -> Embedding:
        """Encode image + instruction into a shared embedding.

        Args:
            image: Preprocessed image tensor, shape [C, H, W] or [B, C, H, W].
            text: Natural language instruction string.

        Returns:
            Embedding object ready for ActionHeadRuntime consumption.

        Raises:
            VLMTimeoutError: If encoding exceeds latency_budget_ms.
            BackendError: For any other backend-level failure.
        """
        ...

    async def generate(self, image: torch.Tensor, text: str) -> TokenStream:
        """Token-level generation for discrete-action-token models.

        Only implement for discrete-token VLA models (e.g. OpenVLA, RT-2).
        Flow matching / diffusion heads use encode() instead.

        Args:
            image: Preprocessed image tensor.
            text: Instruction string.

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
    """Input to the VLA inference pipeline.

    Attributes:
        request_id: Unique ID for this request (caller-assigned or auto-generated).
        image: Preprocessed image tensor [C, H, W].
        instruction: Natural language task instruction.
        robot_state: Current joint positions / velocities / gripper state,
            passed directly to ActionHeadRuntime (NOT tokenized into the prompt).
        frame_id: Monotonically increasing frame counter for this robot stream.
        timestamp: Unix time (seconds) when this request was created.
    """

    request_id: str
    image: torch.Tensor
    instruction: str
    robot_state: torch.Tensor          # shape: [state_dim]
    frame_id: int = 0
    timestamp: float = field(default_factory=time.time)
