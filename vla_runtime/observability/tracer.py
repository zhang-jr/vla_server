"""Structured tracer for VLA Runtime.

Phase 1: JSON log output via stdlib logging.
Phase 3: Will be swapped for OpenTelemetry — subclass and override _emit().

Usage::

    tracer = get_tracer("scheduler")

    with tracer.span("vlm_encode", request_id=req_id) as span:
        embedding = await backend.encode(image, text)
        span.set_attribute("embedding_dim", embedding.dim)
        # span automatically records duration on exit

    # One-shot event
    tracer.event("action_chunk_sent", request_id=req_id, chunk_size=8)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Generator, Optional

logger = logging.getLogger("vla_runtime")


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SpanRecord:
    trace_id: str
    span_id: str
    module: str
    name: str
    request_id: Optional[str]
    start_time_ns: int
    end_time_ns: Optional[int] = None
    duration_ms: Optional[float] = None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"           # ok | error | degraded
    error: Optional[str] = None


@dataclass
class EventRecord:
    trace_id: str
    module: str
    name: str
    request_id: Optional[str]
    timestamp_ns: int
    attributes: dict[str, Any] = field(default_factory=dict)


# ── Span context manager ───────────────────────────────────────────────────────

class ActiveSpan:
    """Returned by :meth:`Tracer.span`; use as a context manager."""

    def __init__(self, record: SpanRecord, tracer: "Tracer") -> None:
        self._record = record
        self._tracer = tracer

    # Public API

    def set_attribute(self, key: str, value: Any) -> None:
        self._record.attributes[key] = value

    def set_error(self, exc: Exception) -> None:
        self._record.status = "error"
        self._record.error = f"{type(exc).__name__}: {exc}"

    def set_degraded(self, reason: str) -> None:
        self._record.status = "degraded"
        self._record.attributes["degraded_reason"] = reason

    # Context manager

    def __enter__(self) -> "ActiveSpan":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        end_ns = time.perf_counter_ns()
        self._record.end_time_ns = end_ns
        self._record.duration_ms = (
            end_ns - self._record.start_time_ns
        ) / 1_000_000.0
        if exc_type is not None:
            self.set_error(exc_val)
        self._tracer._emit_span(self._record)
        return False  # do not suppress exceptions


# ── Tracer ────────────────────────────────────────────────────────────────────

class Tracer:
    """Per-module structured tracer.

    Args:
        module: The module name (e.g. ``"scheduler"``, ``"vlm.vllm"``).
        trace_id: Optional fixed trace_id; auto-generated if None.
    """

    def __init__(self, module: str, trace_id: Optional[str] = None) -> None:
        self.module = module
        self._trace_id = trace_id or str(uuid.uuid4())

    @property
    def trace_id(self) -> str:
        return self._trace_id

    def new_trace(self) -> "Tracer":
        """Return a new Tracer with a fresh trace_id (same module)."""
        return Tracer(self.module)

    def child(self, module: str) -> "Tracer":
        """Return a child Tracer that shares this trace_id."""
        return Tracer(module, trace_id=self._trace_id)

    # ── Span ──────────────────────────────────────────────────────────────────

    @contextmanager
    def span(
        self,
        name: str,
        request_id: Optional[str] = None,
        **attributes: Any,
    ) -> Generator[ActiveSpan, None, None]:
        """Context manager that records a timed span.

        Example::

            with tracer.span("vlm_encode", request_id=rid, model="openvla") as s:
                result = await backend.encode(...)
                s.set_attribute("tokens", result.token_count)
        """
        record = SpanRecord(
            trace_id=self._trace_id,
            span_id=str(uuid.uuid4())[:8],
            module=self.module,
            name=name,
            request_id=request_id,
            start_time_ns=time.perf_counter_ns(),
            attributes=dict(attributes),
        )
        active = ActiveSpan(record, self)
        try:
            yield active
        finally:
            # __exit__ handles emit; we just need the finally for cleanup
            pass

    # ── Event ─────────────────────────────────────────────────────────────────

    def event(
        self,
        name: str,
        request_id: Optional[str] = None,
        **attributes: Any,
    ) -> None:
        """Emit an instantaneous structured event (no duration)."""
        record = EventRecord(
            trace_id=self._trace_id,
            module=self.module,
            name=name,
            request_id=request_id,
            timestamp_ns=time.perf_counter_ns(),
            attributes=attributes,
        )
        self._emit_event(record)

    # ── Emit (override in Phase 3 OTel subclass) ──────────────────────────────

    def _emit_span(self, record: SpanRecord) -> None:
        payload = {
            "type": "span",
            "trace_id": record.trace_id,
            "span_id": record.span_id,
            "module": record.module,
            "name": record.name,
            "request_id": record.request_id,
            "duration_ms": round(record.duration_ms or 0.0, 3),
            "status": record.status,
            **record.attributes,
        }
        if record.error:
            payload["error"] = record.error
        logger.info(json.dumps(payload))

    def _emit_event(self, record: EventRecord) -> None:
        payload = {
            "type": "event",
            "trace_id": record.trace_id,
            "module": record.module,
            "name": record.name,
            "request_id": record.request_id,
            **record.attributes,
        }
        logger.info(json.dumps(payload))


# ── Module-level registry ─────────────────────────────────────────────────────

_registry: dict[str, Tracer] = {}


def get_tracer(module: str) -> Tracer:
    """Return a module-scoped Tracer (singleton per module name).

    The returned tracer generates a new trace_id per :meth:`Tracer.new_trace`
    call; for request-scoped tracing use :meth:`Tracer.child`.
    """
    if module not in _registry:
        _registry[module] = Tracer(module)
    return _registry[module]


def configure_logging(level: int = logging.INFO) -> None:
    """Set up stdlib logging for VLA Runtime structured output.

    Call once at application startup before any tracer use.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
