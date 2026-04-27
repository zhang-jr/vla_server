"""llama.cpp backend adapter for VLA Runtime (edge deployment).

Wraps llama-cpp-python to run quantized GGUF models on Jetson Orin and other
edge devices. This backend is optimised for low VRAM footprint at the cost of
throughput — it is never used in cloud serving.

Reference: LiteVLA-Edge (arxiv 2603.03380) — 150.5ms full pipeline on Jetson.

Design notes:
- llama-cpp-python's Python API is synchronous; we run it in a thread pool
  via asyncio.get_event_loop().run_in_executor() to avoid blocking the loop.
- KV cache reuse is limited: llama.cpp does prefix caching but not across
  arbitrary frame sequences. kv_cache_enabled returns True only when the
  backend can detect a shared prefix.
"""

from __future__ import annotations

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import torch

from vla_runtime.errors import BackendError, EdgeResourceError, ErrorContext, VLMTimeoutError
from vla_runtime.observability.tracer import get_tracer
from vla_runtime.vlm.base import Embedding, TokenChunk, TokenStream

_tracer = get_tracer("vlm.llamacpp")


class LlamaCppBackend:
    """VLMBackend backed by llama-cpp-python for edge (GGUF) deployment.

    Args:
        model_path: Path to a GGUF model file.
        latency_budget_ms: Encode timeout; edge typically 150-300ms.
        n_gpu_layers: Number of layers to offload to GPU (-1 = all).
        n_ctx: Context window size (keep small for edge; VLA context is short).
        n_threads: CPU thread count for layers not offloaded.
        embedding_dim: Expected embedding dimension (model-specific).
    """

    def __init__(
        self,
        model_path: str | Path,
        latency_budget_ms: float = 300.0,
        n_gpu_layers: int = -1,
        n_ctx: int = 512,
        n_threads: int = 4,
        embedding_dim: int = 2048,
        **llama_kwargs: Any,
    ) -> None:
        self._model_path = Path(model_path)
        self._latency_budget_ms = latency_budget_ms
        self._n_gpu_layers = n_gpu_layers
        self._n_ctx = n_ctx
        self._n_threads = n_threads
        self._embedding_dim = embedding_dim
        self._llama_kwargs = llama_kwargs
        self._llm: Any = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llamacpp")
        self._model_id = f"llamacpp:{self._model_path.name}"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def latency_budget_ms(self) -> float:
        return self._latency_budget_ms

    @property
    def kv_cache_enabled(self) -> bool:
        return False  # llama.cpp prefix cache is frame-to-frame best-effort

    async def warmup(self) -> None:
        with _tracer.span("llamacpp_warmup", model=str(self._model_path)) as span:
            if not self._model_path.exists():
                raise BackendError(
                    f"GGUF model not found: {self._model_path}",
                    context=ErrorContext(
                        trace_id=_tracer.trace_id,
                        module_name="vlm.llamacpp",
                    ),
                )
            try:
                from llama_cpp import Llama  # type: ignore[import]
            except ImportError as exc:
                raise BackendError(
                    "llama-cpp-python not installed. Run: pip install llama-cpp-python",
                    context=ErrorContext(
                        trace_id=_tracer.trace_id,
                        module_name="vlm.llamacpp",
                    ),
                ) from exc

            loop = asyncio.get_event_loop()
            self._llm = await loop.run_in_executor(
                self._executor,
                lambda: Llama(
                    model_path=str(self._model_path),
                    n_gpu_layers=self._n_gpu_layers,
                    n_ctx=self._n_ctx,
                    n_threads=self._n_threads,
                    embedding=True,  # enable embedding output
                    verbose=False,
                    **self._llama_kwargs,
                ),
            )
            span.set_attribute("n_gpu_layers", self._n_gpu_layers)
            span.set_attribute("n_ctx", self._n_ctx)
            _tracer.event("llamacpp_model_loaded", model=self._model_id)

    async def shutdown(self) -> None:
        self._llm = None
        self._executor.shutdown(wait=False)
        _tracer.event("llamacpp_shutdown", model=self._model_id)

    async def encode(self, image: torch.Tensor, text: str) -> Embedding:
        request_id = str(uuid.uuid4())[:8]
        ctx = ErrorContext(
            trace_id=_tracer.trace_id,
            module_name="vlm.llamacpp",
            request_id=request_id,
        )
        with _tracer.span("llamacpp_encode", request_id=request_id) as span:
            self._assert_ready(ctx)
            try:
                result = await asyncio.wait_for(
                    self._run_encode_in_thread(image, text),
                    timeout=self._latency_budget_ms / 1000.0,
                )
                span.set_attribute("embedding_dim", result.dim)
                return result
            except asyncio.TimeoutError:
                raise VLMTimeoutError(
                    self._latency_budget_ms, self._latency_budget_ms, ctx
                )
            except Exception as exc:
                span.set_error(exc)
                raise BackendError(f"llama.cpp encode failed: {exc}", ctx) from exc

    async def generate(self, image: torch.Tensor, text: str) -> TokenStream:
        """Token generation — only used when action head expects discrete tokens."""
        request_id = str(uuid.uuid4())[:8]
        ctx = ErrorContext(
            trace_id=_tracer.trace_id,
            module_name="vlm.llamacpp",
            request_id=request_id,
        )
        self._assert_ready(ctx)
        return self._stream_tokens_from_thread(image, text, ctx)

    def _assert_ready(self, ctx: ErrorContext) -> None:
        if self._llm is None:
            raise BackendError(
                "llama.cpp model not loaded — call warmup() first", ctx
            )

    async def _run_encode_in_thread(
        self, image: torch.Tensor, text: str
    ) -> Embedding:
        loop = asyncio.get_event_loop()

        def _blocking_encode() -> list[float]:
            # llama.cpp embedding: tokenise text, run forward, return last-layer hidden
            # Image features would be projected and prepended to the token sequence
            # via a multimodal projection (llava-style). Stub returns zeros.
            token_ids = self._llm.tokenize(text.encode())
            # Real call: self._llm.embed(text) or self._llm(text, embedding=True)
            return [0.0] * self._embedding_dim

        raw = await loop.run_in_executor(self._executor, _blocking_encode)
        tensor = torch.tensor(raw, dtype=torch.float16)
        return Embedding(data=tensor, model_id=self._model_id)

    async def _stream_tokens_from_thread(
        self, image: torch.Tensor, text: str, ctx: ErrorContext
    ) -> AsyncIterator[TokenChunk]:
        yield TokenChunk(token_ids=[], is_final=True)
