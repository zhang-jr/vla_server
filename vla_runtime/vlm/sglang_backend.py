"""SGLang backend adapter for VLA Runtime.

SGLang provides a RadixAttention-based serving engine with a Python frontend
language. It is a cloud-preferred alternative to vLLM, especially for
multi-call workloads with shared prefixes.

SGLang is imported lazily — see vllm_backend.py for rationale.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, AsyncIterator, Optional

import torch

from vla_runtime.errors import BackendError, ErrorContext, VLMTimeoutError
from vla_runtime.observability.tracer import get_tracer
from vla_runtime.vlm.base import Embedding, TokenChunk, TokenStream

_tracer = get_tracer("vlm.sglang")


class SGLangBackend:
    """VLMBackend implementation backed by SGLang Runtime.

    Args:
        model_path: HuggingFace model ID or local path.
        latency_budget_ms: Timeout for encode/generate calls.
        tp_size: Tensor parallel size.
        quantization: Quantization mode (e.g. ``"awq"``).
        server_args: Additional kwargs forwarded to SGLang Runtime.
    """

    def __init__(
        self,
        model_path: str,
        latency_budget_ms: float = 200.0,
        tp_size: int = 1,
        quantization: Optional[str] = None,
        **server_args: Any,
    ) -> None:
        self._model_path = model_path
        self._latency_budget_ms = latency_budget_ms
        self._tp_size = tp_size
        self._quantization = quantization
        self._server_args = server_args
        self._runtime: Any = None
        self._model_id = f"sglang:{model_path}"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def latency_budget_ms(self) -> float:
        return self._latency_budget_ms

    @property
    def kv_cache_enabled(self) -> bool:
        # SGLang RadixAttention provides prefix-sharing KV reuse
        return True

    async def warmup(self) -> None:
        with _tracer.span("sglang_warmup", model=self._model_path):
            try:
                import sglang as sgl  # type: ignore[import]
            except ImportError as exc:
                raise BackendError(
                    "SGLang is not installed. Run: pip install sglang",
                    context=ErrorContext(
                        trace_id=_tracer.trace_id,
                        module_name="vlm.sglang",
                    ),
                ) from exc

            self._runtime = sgl.Runtime(
                model_path=self._model_path,
                tp_size=self._tp_size,
                **self._server_args,
            )
            _tracer.event("sglang_runtime_ready", model=self._model_path)

    async def shutdown(self) -> None:
        if self._runtime is not None:
            self._runtime.shutdown()
            self._runtime = None
            _tracer.event("sglang_runtime_shutdown", model=self._model_path)

    async def encode(self, image: torch.Tensor, text: str) -> Embedding:
        request_id = str(uuid.uuid4())[:8]
        ctx = ErrorContext(
            trace_id=_tracer.trace_id,
            module_name="vlm.sglang",
            request_id=request_id,
        )
        with _tracer.span("sglang_encode", request_id=request_id) as span:
            self._assert_ready(ctx)
            try:
                result = await asyncio.wait_for(
                    self._run_encode(image, text),
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
                raise BackendError(f"SGLang encode failed: {exc}", ctx) from exc

    async def generate(self, image: torch.Tensor, text: str) -> TokenStream:
        request_id = str(uuid.uuid4())[:8]
        ctx = ErrorContext(
            trace_id=_tracer.trace_id,
            module_name="vlm.sglang",
            request_id=request_id,
        )
        self._assert_ready(ctx)
        return self._stream_tokens(image, text, ctx)

    def _assert_ready(self, ctx: ErrorContext) -> None:
        if self._runtime is None:
            raise BackendError(
                "SGLang runtime not initialised — call warmup() first", ctx
            )

    async def _run_encode(self, image: torch.Tensor, text: str) -> Embedding:
        # Stub: replace with real SGLang multimodal encode call
        hidden_dim = 4096
        dummy = torch.zeros(hidden_dim, dtype=torch.float16)
        return Embedding(data=dummy, model_id=self._model_id)

    async def _stream_tokens(
        self,
        image: torch.Tensor,
        text: str,
        ctx: ErrorContext,
    ) -> AsyncIterator[TokenChunk]:
        yield TokenChunk(token_ids=[], is_final=True)
