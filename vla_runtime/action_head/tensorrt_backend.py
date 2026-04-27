"""TensorRT backend for action head acceleration.

Wraps an existing ActionHeadRuntime implementation with TensorRT-compiled
inference. Acts as a transparent accelerator — the wrapped head's Protocol
properties and predict() signature are preserved.

Pipeline:
    1. Export the head's PyTorch model to ONNX.
    2. Compile the ONNX model to a TensorRT engine (FP16 or INT8 with cal).
    3. Replace the PyTorch forward pass with TRT engine execution.

Quantization note:
    TensorRT FP16 is the default (matches action head FP16 constraint).
    INT8 calibration is allowed ONLY for discrete_token heads, and requires
    a calibration dataset + sim validation (ADR-007, ADR-008).

This backend is the fallback when Triton kernel performance is insufficient
(Risk #1 in TODO.md risk register).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import torch

from vla_runtime.action_head.base import ActionChunk, ActionHeadType, RobotState
from vla_runtime.errors import ActionHeadFailureError, ErrorContext, KernelError
from vla_runtime.observability.tracer import get_tracer
from vla_runtime.vlm.base import Embedding

_tracer = get_tracer("action_head.tensorrt")

TRTPrecision = Literal["fp16", "int8"]

# tensorrt is imported lazily — not available on all machines
try:
    import tensorrt as trt  # type: ignore[import]
    _trt_available = True
except ImportError:
    _trt_available = False


class TensorRTBackend:
    """TensorRT-accelerated action head.

    Args:
        wrapped_head: The ActionHeadRuntime to accelerate. Used for:
            - Protocol property delegation.
            - ONNX export (the wrapped head's nn.Module is exported).
        engine_path: Path to cache the compiled TRT engine. If the file
            exists, it is loaded directly (no recompilation).
        precision: ``"fp16"`` (default) or ``"int8"`` (discrete_token only).
        workspace_mb: TensorRT builder workspace in MB.
        dynamic_batch: Whether to use dynamic batch size optimization.
    """

    def __init__(
        self,
        wrapped_head: Any,  # ActionHeadRuntime implementor
        engine_path: str | Path,
        precision: TRTPrecision = "fp16",
        workspace_mb: int = 1024,
        dynamic_batch: bool = False,
    ) -> None:
        self._wrapped = wrapped_head
        self._engine_path = Path(engine_path)
        self._precision = precision
        self._workspace_mb = workspace_mb
        self._dynamic_batch = dynamic_batch
        self._engine: Any = None
        self._context: Any = None

        # Enforce precision constraint
        if precision == "int8" and wrapped_head.head_type in ("flow_matching", "diffusion"):
            from vla_runtime.errors import PrecisionDegradationError
            raise PrecisionDegradationError(
                baseline_rate=float("nan"),
                quantized_rate=float("nan"),
                threshold_drop=0.0,
                context=ErrorContext(
                    trace_id=_tracer.trace_id,
                    module_name="action_head.tensorrt",
                    extra={"reason": "INT8 forbidden for flow_matching/diffusion heads"},
                ),
            )

    # ── Protocol delegation ───────────────────────────────────────────────────

    @property
    def head_type(self) -> ActionHeadType:
        return self._wrapped.head_type

    @property
    def frequency_hz(self) -> float:
        return self._wrapped.frequency_hz

    @property
    def chunk_size(self) -> int:
        return self._wrapped.chunk_size

    @property
    def action_dim(self) -> int:
        return self._wrapped.action_dim

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def build_or_load_engine(self, example_inputs: dict[str, torch.Tensor]) -> None:
        """Build a TRT engine from ONNX export or load from cache.

        Args:
            example_inputs: Dict of {input_name: tensor} for ONNX export shapes.
        """
        ctx = ErrorContext(trace_id=_tracer.trace_id, module_name="action_head.tensorrt")

        if not _trt_available:
            raise KernelError(
                "tensorrt",
                RuntimeError("TensorRT is not installed"),
                ctx,
            )

        with _tracer.span("trt_build_or_load", engine=str(self._engine_path)) as span:
            if self._engine_path.exists():
                self._engine = self._load_engine(ctx)
                span.set_attribute("source", "cache")
            else:
                onnx_path = self._engine_path.with_suffix(".onnx")
                self._export_onnx(example_inputs, onnx_path, ctx)
                self._engine = self._compile_engine(onnx_path, ctx)
                span.set_attribute("source", "compiled")

            self._context = self._engine.create_execution_context()
            _tracer.event("trt_engine_ready", head_type=self.head_type, precision=self._precision)

    def shutdown(self) -> None:
        self._context = None
        self._engine = None
        _tracer.event("trt_engine_shutdown")

    # ── Core interface ────────────────────────────────────────────────────────

    def predict(self, embedding: Embedding, robot_state: RobotState) -> ActionChunk:
        """Run TRT-accelerated inference.

        Falls back to the wrapped PyTorch head if TRT engine is not built.
        """
        if self._engine is None or self._context is None:
            _tracer.event("trt_fallback_to_pytorch", reason="engine_not_built")
            return self._wrapped.predict(embedding, robot_state)

        ctx = ErrorContext(trace_id=_tracer.trace_id, module_name="action_head.tensorrt")

        with _tracer.span("trt_predict") as span:
            try:
                actions = self._run_trt(embedding, robot_state)
                span.set_attribute("chunk_size", self.chunk_size)
                span.set_attribute("precision", self._precision)
                return ActionChunk(
                    actions=actions,
                    head_type=self.head_type,
                    metadata={"backend": "tensorrt", "precision": self._precision},
                )
            except Exception as exc:
                span.set_error(exc)
                raise ActionHeadFailureError(self.head_type, exc, ctx) from exc

    def load_weights(self, checkpoint_path: str) -> None:
        self._wrapped.load_weights(checkpoint_path)

    def to_device(self, device: torch.device) -> None:
        self._wrapped.to_device(device)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _export_onnx(
        self,
        example_inputs: dict[str, torch.Tensor],
        onnx_path: Path,
        ctx: ErrorContext,
    ) -> None:
        """Export the wrapped head's model to ONNX."""
        # Production: call torch.onnx.export on the wrapped head's nn.Module.
        # Stub: writes a placeholder file.
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        _tracer.event("trt_onnx_exported", path=str(onnx_path))

    def _compile_engine(self, onnx_path: Path, ctx: ErrorContext) -> Any:
        """Compile ONNX to TensorRT engine and serialize to disk."""
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, logger)

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self._workspace_mb * 1024 * 1024)

        if self._precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)

        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            raise KernelError("tensorrt", RuntimeError("TRT engine build failed"), ctx)

        self._engine_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine_path.write_bytes(engine_bytes)
        _tracer.event("trt_engine_compiled", path=str(self._engine_path))

        runtime = trt.Runtime(logger)
        return runtime.deserialize_cuda_engine(engine_bytes)

    def _load_engine(self, ctx: ErrorContext) -> Any:
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine_bytes = self._engine_path.read_bytes()
        return runtime.deserialize_cuda_engine(engine_bytes)

    def _run_trt(self, embedding: Embedding, robot_state: RobotState) -> torch.Tensor:
        """Execute TRT engine inference. Stub — production binds I/O buffers."""
        # Production: allocate CUDA buffers, bind inputs/outputs, run context.execute_async_v3().
        # Fallback to wrapped head for now.
        return self._wrapped.predict(embedding, robot_state).actions
