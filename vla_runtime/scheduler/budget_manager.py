"""Latency budget manager for the AsyncScheduler.

The core constraint from CLAUDE.md:

    inference_latency < action_execution_time

This is a hard constraint validated experimentally at NXP. The BudgetManager
tracks per-module latency and fires callbacks when budgets are exceeded.

Budget hierarchy:
    total_budget_ms
    ├── vlm_budget_ms          (Thread A: 1-5 Hz, heavy)
    ├── action_head_budget_ms  (Thread B: 5-20 Hz, medium)
    └── control_budget_ms      (Thread C: 20-100 Hz, real-time — NOT tracked here)

The control budget is enforced in C++ (control node). The Python scheduler
only tracks VLM + action head budgets.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

from vla_runtime.errors import BudgetExceededError, ErrorContext
from vla_runtime.observability.metrics import get_metrics
from vla_runtime.observability.tracer import get_tracer

_tracer = get_tracer("scheduler.budget_manager")
_metrics = get_metrics()


@dataclass
class BudgetConfig:
    """Latency budget configuration in milliseconds.

    Attributes:
        vlm_budget_ms: Maximum allowed VLM encode/generate latency.
        action_head_budget_ms: Maximum allowed action head predict latency.
        total_budget_ms: End-to-end budget (must be < action_execution_time).
        window_size: Rolling window size for P50/P95 tracking.
        degradation_threshold: Fraction of recent requests that may exceed
            budget before the on_budget_exceeded callback fires.
            Default 0.1 = 10% tolerance.
    """

    vlm_budget_ms: float = 200.0
    action_head_budget_ms: float = 50.0
    total_budget_ms: float = 300.0
    window_size: int = 100
    degradation_threshold: float = 0.1


@dataclass
class LatencySample:
    module: str
    latency_ms: float
    timestamp: float = field(default_factory=time.time)
    exceeded: bool = False


class BudgetManager:
    """Tracks per-module latency and fires degradation callbacks.

    Args:
        config: Budget configuration.
        on_budget_exceeded: Optional callback invoked when rolling budget
            violation rate exceeds config.degradation_threshold.
            Signature: (module, latency_ms, budget_ms) → None.
    """

    def __init__(
        self,
        config: BudgetConfig | None = None,
        on_budget_exceeded: Optional[Callable[[str, float, float], None]] = None,
    ) -> None:
        self._config = config or BudgetConfig()
        self._on_budget_exceeded = on_budget_exceeded
        self._samples: dict[str, deque[LatencySample]] = {
            "vlm": deque(maxlen=self._config.window_size),
            "action_head": deque(maxlen=self._config.window_size),
            "total": deque(maxlen=self._config.window_size),
        }

    # ── Record ────────────────────────────────────────────────────────────────

    def record(self, module: str, latency_ms: float, request_id: Optional[str] = None) -> None:
        """Record a latency measurement and check against budget.

        Args:
            module: One of ``"vlm"``, ``"action_head"``, ``"total"``.
            latency_ms: Measured latency.
            request_id: Optional for tracing correlation.

        Raises:
            BudgetExceededError: If the single measurement exceeds its hard limit
                AND the rolling violation rate is above degradation_threshold.
        """
        budget = self._budget_for(module)
        exceeded = latency_ms > budget

        sample = LatencySample(module=module, latency_ms=latency_ms, exceeded=exceeded)
        self._samples[module].append(sample)

        # Emit metrics
        _metrics.histogram(f"{module}_latency_ms", latency_ms, labels={"module": module})
        if exceeded:
            _metrics.increment(f"{module}_budget_exceeded")

        if exceeded:
            violation_rate = self._violation_rate(module)
            _tracer.event(
                "budget_exceeded",
                module=module,
                latency_ms=round(latency_ms, 2),
                budget_ms=budget,
                violation_rate=round(violation_rate, 3),
                request_id=request_id,
            )
            if violation_rate > self._config.degradation_threshold:
                if self._on_budget_exceeded is not None:
                    self._on_budget_exceeded(module, latency_ms, budget)

    def check_and_raise(
        self,
        module: str,
        latency_ms: float,
        context: Optional[ErrorContext] = None,
    ) -> None:
        """Record + raise BudgetExceededError on any single violation.

        Use in strict mode where any violation must trigger a degraded path.
        """
        self.record(module, latency_ms)
        budget = self._budget_for(module)
        if latency_ms > budget:
            raise BudgetExceededError(latency_ms, budget, context)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def percentile(self, module: str, p: float) -> Optional[float]:
        """Return the p-th percentile latency (e.g. p=95 for P95).

        Returns None if fewer than 2 samples are recorded.
        """
        samples = list(self._samples.get(module, []))
        if len(samples) < 2:
            return None
        values = sorted(s.latency_ms for s in samples)
        idx = int(len(values) * p / 100)
        return values[min(idx, len(values) - 1)]

    def violation_rate(self, module: str) -> float:
        return self._violation_rate(module)

    def snapshot(self) -> dict:
        result = {}
        for module in self._samples:
            budget = self._budget_for(module)
            p50 = self.percentile(module, 50)
            p95 = self.percentile(module, 95)
            result[module] = {
                "budget_ms": budget,
                "p50_ms": round(p50, 2) if p50 is not None else None,
                "p95_ms": round(p95, 2) if p95 is not None else None,
                "violation_rate": round(self._violation_rate(module), 4),
                "n_samples": len(self._samples[module]),
            }
        return result

    # ── Internal ──────────────────────────────────────────────────────────────

    def _budget_for(self, module: str) -> float:
        budgets = {
            "vlm": self._config.vlm_budget_ms,
            "action_head": self._config.action_head_budget_ms,
            "total": self._config.total_budget_ms,
        }
        return budgets.get(module, self._config.total_budget_ms)

    def _violation_rate(self, module: str) -> float:
        samples = self._samples.get(module, deque())
        if not samples:
            return 0.0
        return sum(1 for s in samples if s.exceeded) / len(samples)
