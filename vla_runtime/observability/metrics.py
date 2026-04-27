"""Runtime metrics collection for VLA Runtime.

Phase 1: In-process counters / gauges backed by simple Python dicts.
Phase 3: Will be wired to Prometheus via prometheus_client.

Usage::

    metrics = get_metrics()
    metrics.increment("action_chunks_sent", labels={"robot_id": "r1"})
    metrics.gauge("vlm_queue_depth", 3)
    metrics.histogram("vlm_latency_ms", 142.5)
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HistogramStats:
    count: int = 0
    total: float = 0.0
    min: float = float("inf")
    max: float = float("-inf")
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def observe(self, value: float) -> None:
        with self._lock:
            self.count += 1
            self.total += value
            self.min = min(self.min, value)
            self.max = max(self.max, value)

    @property
    def mean(self) -> Optional[float]:
        return self.total / self.count if self.count > 0 else None

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "count": self.count,
                "total_ms": round(self.total, 3),
                "mean_ms": round(self.mean, 3) if self.mean is not None else None,
                "min_ms": round(self.min, 3) if self.count > 0 else None,
                "max_ms": round(self.max, 3) if self.count > 0 else None,
            }


class Metrics:
    """Lightweight in-process metrics store."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, HistogramStats] = defaultdict(HistogramStats)

    def increment(self, name: str, delta: int = 1, labels: Optional[dict] = None) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] += delta

    def gauge(self, name: str, value: float, labels: Optional[dict] = None) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._gauges[key] = value

    def histogram(self, name: str, value: float, labels: Optional[dict] = None) -> None:
        key = self._key(name, labels)
        self._histograms[key].observe(value)

    def snapshot(self) -> dict:
        """Return a point-in-time snapshot of all metrics."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    k: v.snapshot() for k, v in self._histograms.items()
                },
            }

    @staticmethod
    def _key(name: str, labels: Optional[dict]) -> str:
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"


_default_metrics: Optional[Metrics] = None


def get_metrics() -> Metrics:
    """Return the process-wide Metrics singleton."""
    global _default_metrics
    if _default_metrics is None:
        _default_metrics = Metrics()
    return _default_metrics
