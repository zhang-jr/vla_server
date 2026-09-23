"""VLA Runtime exception hierarchy.

All exceptions carry trace_id, module_name, and request_id for structured
observability. Never catch and suppress — always re-raise with context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ErrorContext:
    trace_id: str
    module_name: str
    request_id: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [f"trace_id={self.trace_id}", f"module={self.module_name}"]
        if self.request_id:
            parts.append(f"request_id={self.request_id}")
        parts.extend(f"{k}={v}" for k, v in self.extra.items())
        return " ".join(parts)


# ── Root ──────────────────────────────────────────────────────────────────────

class VLARuntimeError(Exception):
    """Base class for all VLA Runtime errors."""

    def __init__(self, message: str, context: Optional[ErrorContext] = None) -> None:
        self.context = context
        suffix = f" [{context}]" if context else ""
        super().__init__(f"{message}{suffix}")


# ── Backend ───────────────────────────────────────────────────────────────────

class BackendError(VLARuntimeError):
    """Errors originating from model backends."""


class VLMTimeoutError(BackendError):
    """VLM backbone did not return within the latency budget."""

    def __init__(
        self,
        timeout_ms: float,
        budget_ms: float,
        context: Optional[ErrorContext] = None,
    ) -> None:
        self.timeout_ms = timeout_ms
        self.budget_ms = budget_ms
        super().__init__(
            f"VLM timed out after {timeout_ms:.1f}ms (budget={budget_ms:.1f}ms)",
            context,
        )


class ActionHeadFailureError(BackendError):
    """Action head raised an unrecoverable error."""

    def __init__(
        self,
        head_type: str,
        cause: Optional[Exception] = None,
        context: Optional[ErrorContext] = None,
    ) -> None:
        self.head_type = head_type
        self.cause = cause
        super().__init__(
            f"Action head '{head_type}' failed: {cause}",
            context,
        )


# ── Pipeline ──────────────────────────────────────────────────────────────────

class PipelineError(VLARuntimeError):
    """Errors within the async inference pipeline."""


class BudgetExceededError(PipelineError):
    """End-to-end latency exceeded action_execution_time.

    This violates the hard constraint: inference_latency < action_execution_time.
    The scheduler must trigger a degraded path rather than block.
    """

    def __init__(
        self,
        latency_ms: float,
        budget_ms: float,
        context: Optional[ErrorContext] = None,
    ) -> None:
        self.latency_ms = latency_ms
        self.budget_ms = budget_ms
        super().__init__(
            f"Latency budget exceeded: {latency_ms:.1f}ms > {budget_ms:.1f}ms",
            context,
        )


class EmbeddingStaleError(PipelineError):
    """A reused condition is older than the model tolerates.

    The tolerance is ``PolicyCapabilities.max_condition_staleness_ms`` —
    calibrated per model and task suite by sim success rate (ADR-018), not a
    frame count, because how long a condition stays usable depends on the
    model and the scene, not on how many requests arrived meanwhile.
    """

    def __init__(
        self,
        age_ms: float,
        max_age_ms: float,
        context: Optional[ErrorContext] = None,
    ) -> None:
        self.age_ms = age_ms
        self.max_age_ms = max_age_ms
        super().__init__(
            f"Condition stale: {age_ms:.1f}ms old (max={max_age_ms:.1f}ms)",
            context,
        )


class PolicyCapabilityError(PipelineError):
    """An operation the loaded policy's declared capabilities do not support.

    Example: asking a coupled-frequency model to re-run its action head on a
    cached condition, which yields no new information for that model (ADR-018).
    """


# ── Quantization ──────────────────────────────────────────────────────────────

class QuantizationError(VLARuntimeError):
    """Errors in quantization configuration or calibration."""


class PrecisionDegradationError(QuantizationError):
    """Task success rate dropped below acceptable threshold after quantization.

    Raised during sim validation when the quantized model underperforms by more
    than the allowed margin (typically 2 percentage points).
    """

    def __init__(
        self,
        baseline_rate: float,
        quantized_rate: float,
        threshold_drop: float,
        context: Optional[ErrorContext] = None,
    ) -> None:
        self.baseline_rate = baseline_rate
        self.quantized_rate = quantized_rate
        self.threshold_drop = threshold_drop
        actual_drop = baseline_rate - quantized_rate
        super().__init__(
            f"Task success rate dropped {actual_drop:.2%} "
            f"(baseline={baseline_rate:.2%}, quantized={quantized_rate:.2%}, "
            f"allowed_drop={threshold_drop:.2%})",
            context,
        )


# ── Kernel ────────────────────────────────────────────────────────────────────

class KernelError(VLARuntimeError):
    """Triton / TensorRT compilation or runtime errors."""

    def __init__(
        self,
        kernel_name: str,
        cause: Optional[Exception] = None,
        context: Optional[ErrorContext] = None,
    ) -> None:
        self.kernel_name = kernel_name
        self.cause = cause
        super().__init__(
            f"Kernel '{kernel_name}' error: {cause}",
            context,
        )


# ── Deployment ────────────────────────────────────────────────────────────────

class DeploymentError(VLARuntimeError):
    """Errors related to deployment topology and resource availability."""


class EdgeResourceError(DeploymentError):
    """Edge device has insufficient VRAM or compute for the requested operation."""

    def __init__(
        self,
        required_mb: float,
        available_mb: float,
        context: Optional[ErrorContext] = None,
    ) -> None:
        self.required_mb = required_mb
        self.available_mb = available_mb
        super().__init__(
            f"Edge resource insufficient: need {required_mb:.0f}MB, "
            f"available {available_mb:.0f}MB",
            context,
        )


class NetworkPartitionError(DeploymentError):
    """Cloud backend is unreachable; edge fallback triggered."""

    def __init__(
        self,
        timeout_s: float,
        context: Optional[ErrorContext] = None,
    ) -> None:
        self.timeout_s = timeout_s
        super().__init__(
            f"Cloud unreachable after {timeout_s:.1f}s — edge fallback active",
            context,
        )
