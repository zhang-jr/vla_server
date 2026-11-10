"""LMDeploy backend adapter for VLA Runtime.

LMDeploy (from OpenMMLab) provides a TurboMind inference engine with support
for W4A16 / W8A16 quantization and efficient KV cache. It is a cloud-preferred
alternative that benchmarks competitively with vLLM on Llama-family models.

LMDeploy is imported lazily — see vllm_backend.py for rationale.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, AsyncIterator, Mapping, Optional

import torch

from vla_runtime.action_head.base import RobotState
from vla_runtime.errors import BackendError, ErrorContext, VLMTimeoutError
from vla_runtime.observability.tracer import get_tracer
from vla_runtime.vlm.base import Embedding, TokenChunk, TokenStream

_tracer = get_tracer("vlm.lmdeploy")


class LMDeployBackend:
    """VLMBackend backed by LMDeploy TurboMind engine.

    Args:
        model_path: HuggingFace model ID or local path.
        latency_budget_ms: Encode timeout.
        tp: Tensor parallel degree.
        cache_max_entry_count: KV cache capacity ratio (0.0-1.0).
        quant_policy: LMDeploy quant policy (0=FP16, 4=W4A16, 8=W8A16).
        engine_config: Additional kwargs forwarded to TurbomindEngineConfig.
    """

    def __init__(
        self,
        model_path: str,
        latency_budget_ms: float = 200.0,
        tp: int = 1,
        cache_max_entry_count: float = 0.8,
        quant_policy: int = 0,
        **engine_config: Any,
    ) -> None:
        self._model_path = model_path
        self._latency_budget_ms = latency_budget_ms
        self._tp = tp
        self._cache_max_entry_count = cache_max_entry_count
        self._quant_policy = quant_policy
        self._engine_config = engine_config
        self._pipeline: Any = None
        self._model_id = f"lmdeploy:{model_path}"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def latency_budget_ms(self) -> float:
        return self._latency_budget_ms

    @property
    def kv_cache_enabled(self) -> bool:
        return True  # TurboMind has continuous-batch KV cache

    async def warmup(self) -> None:
        with _tracer.span("lmdeploy_warmup", model=self._model_path):
            try:
                from lmdeploy import pipeline, TurbomindEngineConfig  # type: ignore[import]
            except ImportError as exc:
                raise BackendError(
                    "LMDeploy not installed. Run: pip install lmdeploy",
                    context=ErrorContext(
                        trace_id=_tracer.trace_id,
                        module_name="vlm.lmdeploy",
                    ),
                ) from exc

            engine_cfg = TurbomindEngineConfig(
                tp=self._tp,
                cache_max_entry_count=self._cache_max_entry_count,
                quant_policy=self._quant_policy,
                **self._engine_config,
            )
            self._pipeline = pipeline(self._model_path, backend_config=engine_cfg)
            _tracer.event("lmdeploy_pipeline_ready", model=self._model_path)

    async def shutdown(self) -> None:
        self._pipeline = None
        _tracer.event("lmdeploy_shutdown", model=self._model_id)

    async def encode(
        self,
        images: Mapping[str, torch.Tensor],
        text: str,
        state: Optional[RobotState] = None,
    ) -> Embedding:
        request_id = str(uuid.uuid4())[:8]
        ctx = ErrorContext(
            trace_id=_tracer.trace_id,
            module_name="vlm.lmdeploy",
            request_id=request_id,
        )
        with _tracer.span("lmdeploy_encode", request_id=request_id) as span:
            self._assert_ready(ctx)
            try:
                result = await asyncio.wait_for(
                    self._run_encode(images, text),
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
                raise BackendError(f"LMDeploy encode failed: {exc}", ctx) from exc

    async def generate(
        self,
        images: Mapping[str, torch.Tensor],
        text: str,
        state: Optional[RobotState] = None,
    ) -> TokenStream:
        request_id = str(uuid.uuid4())[:8]
        ctx = ErrorContext(
            trace_id=_tracer.trace_id,
            module_name="vlm.lmdeploy",
            request_id=request_id,
        )
        self._assert_ready(ctx)
        return self._stream_tokens(images, text, ctx)

    def _assert_ready(self, ctx: ErrorContext) -> None:
        if self._pipeline is None:
            raise BackendError(
                "LMDeploy pipeline not initialised — call warmup() first", ctx
            )

    async def _run_encode(self, images: Mapping[str, torch.Tensor], text: str) -> Embedding:
        # Stub: replace with real LMDeploy multimodal encode call
        hidden_dim = 4096
        dummy = torch.zeros(hidden_dim, dtype=torch.float16)
        return Embedding(data=dummy, model_id=self._model_id)

    async def _stream_tokens(
        self, images: Mapping[str, torch.Tensor], text: str, ctx: ErrorContext
    ) -> AsyncIterator[TokenChunk]:
        yield TokenChunk(token_ids=[], is_final=True)
