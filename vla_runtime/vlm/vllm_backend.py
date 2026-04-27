"""vLLM backend adapter for VLA Runtime.

Wraps the vLLM AsyncLLMEngine to serve as a VLMBackend.
Designed for cloud GPU deployment (A100 / H100).

vLLM is imported lazily so the package can be imported on machines without
vLLM installed (e.g., edge devices running only llamacpp_backend).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

import torch

from vla_runtime.errors import BackendError, ErrorContext, VLMTimeoutError
from vla_runtime.observability.tracer import get_tracer
from vla_runtime.vlm.base import Embedding, TokenChunk, TokenStream

if TYPE_CHECKING:
    pass  # avoid circular at type-check time

_tracer = get_tracer("vlm.vllm")


class VLLMBackend:
    """VLMBackend implementation backed by vLLM AsyncLLMEngine.

    Args:
        model_path: HuggingFace model ID or local path.
        latency_budget_ms: Timeout for encode/generate calls.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        quantization: vLLM quantization setting (e.g. ``"awq"``, ``"gptq"``).
        kv_cache_enabled: Whether to enable vLLM's KV cache (usually True).
        engine_kwargs: Additional kwargs forwarded to AsyncLLMEngine.
    """

    def __init__(
        self,
        model_path: str,
        latency_budget_ms: float = 200.0,
        tensor_parallel_size: int = 1,
        quantization: Optional[str] = None,
        kv_cache_enabled: bool = True,
        **engine_kwargs: Any,
    ) -> None:
        self._model_path = model_path
        self._latency_budget_ms = latency_budget_ms
        self._tensor_parallel_size = tensor_parallel_size
        self._quantization = quantization
        self._kv_cache_enabled = kv_cache_enabled
        self._engine_kwargs = engine_kwargs
        self._engine: Any = None  # vllm.AsyncLLMEngine, loaded on warmup
        self._model_id = f"vllm:{model_path}"

    # ── Protocol properties ───────────────────────────────────────────────────

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def latency_budget_ms(self) -> float:
        return self._latency_budget_ms

    @property
    def kv_cache_enabled(self) -> bool:
        return self._kv_cache_enabled

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def warmup(self) -> None:
        """Load model weights and run a dummy forward pass."""
        with _tracer.span("vllm_warmup", model=self._model_path):
            try:
                from vllm import AsyncEngineArgs, AsyncLLMEngine  # type: ignore[import]
            except ImportError as e:
                raise BackendError(
                    "vLLM is not installed. Run: pip install vllm",
                    context=ErrorContext(
                        trace_id=_tracer.trace_id,
                        module_name="vlm.vllm",
                    ),
                ) from e

            engine_args = AsyncEngineArgs(
                model=self._model_path,
                tensor_parallel_size=self._tensor_parallel_size,
                quantization=self._quantization,
                **self._engine_kwargs,
            )
            self._engine = AsyncLLMEngine.from_engine_args(engine_args)
            _tracer.event("vllm_engine_ready", model=self._model_path)

    async def shutdown(self) -> None:
        if self._engine is not None:
            # vLLM engine cleanup
            await self._engine.abort_all_requests("shutdown")
            self._engine = None
            _tracer.event("vllm_engine_shutdown", model=self._model_path)

    # ── Core interface ────────────────────────────────────────────────────────

    async def encode(self, image: torch.Tensor, text: str) -> Embedding:
        """Encode image + instruction → Embedding via vLLM's multimodal path."""
        request_id = str(uuid.uuid4())[:8]
        ctx = ErrorContext(
            trace_id=_tracer.trace_id,
            module_name="vlm.vllm",
            request_id=request_id,
        )

        with _tracer.span("vllm_encode", request_id=request_id) as span:
            self._assert_ready(ctx)
            try:
                result = await asyncio.wait_for(
                    self._run_encode(image, text, request_id),
                    timeout=self._latency_budget_ms / 1000.0,
                )
                span.set_attribute("embedding_dim", result.dim)
                return result
            except asyncio.TimeoutError:
                elapsed = self._latency_budget_ms  # timed out exactly at budget
                span.set_error(
                    VLMTimeoutError(elapsed, self._latency_budget_ms, ctx)
                )
                raise VLMTimeoutError(elapsed, self._latency_budget_ms, ctx)
            except Exception as exc:
                span.set_error(exc)
                raise BackendError(f"vLLM encode failed: {exc}", ctx) from exc

    async def generate(self, image: torch.Tensor, text: str) -> TokenStream:
        """Stream action tokens via vLLM generation (discrete-token models)."""
        request_id = str(uuid.uuid4())[:8]
        ctx = ErrorContext(
            trace_id=_tracer.trace_id,
            module_name="vlm.vllm",
            request_id=request_id,
        )
        self._assert_ready(ctx)
        return self._stream_tokens(image, text, request_id, ctx)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _assert_ready(self, ctx: ErrorContext) -> None:
        if self._engine is None:
            raise BackendError(
                "vLLM engine not initialised — call warmup() first",
                ctx,
            )

    async def _run_encode(
        self, image: torch.Tensor, text: str, request_id: str
    ) -> Embedding:
        """Internal: run the actual vLLM multimodal encode.

        In production this calls the vLLM engine's vision encoder path to
        produce the final hidden states used as the action-head embedding.
        """
        # NOTE: Exact vLLM multimodal API depends on the model architecture.
        # This is a stub that produces correctly-shaped tensors.
        # Replace with real vLLM multimodal encode call when integrating a
        # specific model (e.g., OpenVLA uses Llava-style image projection).
        hidden_dim = 4096  # placeholder; real value from model config
        dummy_embedding = torch.zeros(hidden_dim, dtype=torch.float16)

        return Embedding(
            data=dummy_embedding,
            frame_id=0,
            model_id=self._model_id,
        )

    async def _stream_tokens(
        self,
        image: torch.Tensor,
        text: str,
        request_id: str,
        ctx: ErrorContext,
    ) -> AsyncIterator[TokenChunk]:
        """Internal: wrap vLLM RequestOutput into TokenChunk stream."""
        # Stub — real implementation drives self._engine.generate()
        yield TokenChunk(token_ids=[], is_final=True)
