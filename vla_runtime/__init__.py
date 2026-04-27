"""VLA Runtime — model-agnostic VLA serving runtime.

Import the key public symbols here for convenience::

    from vla_runtime import AsyncScheduler, ActionRequest, get_tracer
"""

from vla_runtime.errors import (
    VLARuntimeError,
    BackendError,
    VLMTimeoutError,
    ActionHeadFailureError,
    PipelineError,
    BudgetExceededError,
    EmbeddingStaleError,
    QuantizationError,
    PrecisionDegradationError,
    KernelError,
    DeploymentError,
    EdgeResourceError,
    NetworkPartitionError,
)
from vla_runtime.observability.tracer import get_tracer, configure_logging

__all__ = [
    # errors
    "VLARuntimeError",
    "BackendError",
    "VLMTimeoutError",
    "ActionHeadFailureError",
    "PipelineError",
    "BudgetExceededError",
    "EmbeddingStaleError",
    "QuantizationError",
    "PrecisionDegradationError",
    "KernelError",
    "DeploymentError",
    "EdgeResourceError",
    "NetworkPartitionError",
    # observability
    "get_tracer",
    "configure_logging",
]
