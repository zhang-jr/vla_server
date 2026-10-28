"""Timing loop, statistics and requirement verdict for the latency benchmark.

Framework-agnostic on purpose: the same harness runs under the LeRobot, OpenVLA,
OpenVLA-OFT and openpi (JAX) environments, so it depends on numpy and the
stdlib only. GPU memory comes from NVML (``nvidia-ml-py``), which sees every
framework's allocations; torch's own peak counter is added when torch is loaded.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np

Verdict = Literal["pass", "fail_latency", "fail_fps", "fail_both", "error"]


# ── Model protocol ────────────────────────────────────────────────────────────


@dataclass
class Observation:
    """One raw robot observation, as a camera / arm driver would hand it over.

    Attributes:
        images: Camera name → uint8 array [H, W, 3].
        state: Proprioceptive state, float32 [D].
        instruction: Language instruction.
        extra: Model-specific pre-built inputs (e.g. an openpi example dict).
    """

    images: dict[str, np.ndarray]
    state: np.ndarray
    instruction: str
    extra: dict[str, Any] = field(default_factory=dict)


class LatencyModel(Protocol):
    """What the harness needs from a model adapter."""

    name: str

    def load(self) -> None:
        """Download / build / move to device. Not timed."""

    def make_observation(self, rng: np.random.Generator, image_hw: tuple[int, int]) -> Observation:
        """A valid observation for this model; images at the raw camera resolution."""

    def infer(self, obs: Observation, timer: StageTimer) -> np.ndarray:
        """Full inference: raw observation → actions [T, D] on the host."""

    def describe(self) -> dict[str, Any]:
        """Checkpoint, precision, cameras, … recorded with the result."""


# ── Stage timer ───────────────────────────────────────────────────────────────


class StageTimer:
    """Per-call stage durations; ``sync`` runs before each stage closes.

    Without the sync, asynchronous GPU work would be billed to whichever later
    stage first blocks on it. The total of a call is timed independently by
    the harness, so stages may nest or leave gaps (``describe`` says which).

    Args:
        sync: Device barrier (``torch.cuda.synchronize``); no-op when None.
    """

    def __init__(self, sync: Callable[[], None] | None = None) -> None:
        self._sync = sync
        self.stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        yield
        if self._sync is not None:
            self._sync()
        self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter() - start) * 1e3

    def record(self, name: str, ms: float) -> None:
        """A duration measured inside the model (e.g. openpi's own ``infer_ms``)."""
        self.stages[name] = self.stages.get(name, 0.0) + ms


# ── GPU memory ────────────────────────────────────────────────────────────────


def _gpu_index() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return int(visible) if visible.isdigit() else 0


class GpuMemoryMonitor:
    """Samples device memory in use (NVML) on a background thread.

    ``baseline_gb`` is taken at construction, before the model loads, so
    ``peak_delta_gb`` is what this process added even on a shared GPU.

    Args:
        interval_s: Sampling period.
    """

    def __init__(self, interval_s: float = 0.01) -> None:
        self._interval = interval_s
        self._handle: Any = None
        self._nvml: Any = None
        self._peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline_gb: float | None = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(_gpu_index())
            self.baseline_gb = self._used() / 2**30
        except Exception as exc:  # NVML absent (CPU host) or no permission: report, don't fail
            self.unavailable_reason = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self._handle is not None

    def _used(self) -> int:
        return int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._peak = max(self._peak, self._used())
            self._stop.wait(self._interval)

    def start(self) -> None:
        if not self.available:
            return
        self._peak = self._used()
        self._thread = threading.Thread(target=self._loop, name="gpu-mem", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float | str | None]:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
        if not self.available:
            return {"nvml": getattr(self, "unavailable_reason", "unavailable")}
        peak_gb = self._peak / 2**30
        out: dict[str, float | str | None] = {
            "device_peak_gb": round(peak_gb, 2),
            "baseline_gb": round(self.baseline_gb or 0.0, 2),
            "peak_delta_gb": round(peak_gb - (self.baseline_gb or 0.0), 2),
        }
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            out["torch_max_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        return out

    def device_info(self) -> dict[str, Any]:
        if not self.available:
            return {}
        nvml = self._nvml
        name = nvml.nvmlDeviceGetName(self._handle)
        info: dict[str, Any] = {
            "gpu": name.decode() if isinstance(name, bytes) else name,
            "gpu_index": _gpu_index(),
            "gpu_memory_total_gb": round(nvml.nvmlDeviceGetMemoryInfo(self._handle).total / 2**30, 1),
        }
        driver = nvml.nvmlSystemGetDriverVersion()
        info["driver"] = driver.decode() if isinstance(driver, bytes) else driver
        try:
            info["power_limit_w"] = nvml.nvmlDeviceGetEnforcedPowerLimit(self._handle) / 1000
            info["max_sm_clock_mhz"] = nvml.nvmlDeviceGetMaxClockInfo(self._handle, nvml.NVML_CLOCK_SM)
        except nvml.NVMLError:
            pass
        return info


# ── Statistics ────────────────────────────────────────────────────────────────


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (numpy's default); None for no data."""
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _r(value: float | None, digits: int = 1) -> float | None:
    return None if value is None else round(value, digits)


def latency_stats(values: list[float]) -> dict[str, float | None]:
    return {
        "p50": _r(percentile(values, 50)),
        "p95": _r(percentile(values, 95)),
        "p99": _r(percentile(values, 99)),
        "mean": _r(float(np.mean(values)) if values else None),
        "std": _r(float(np.std(values)) if values else None),
        "max": _r(max(values) if values else None),
    }


@dataclass(frozen=True)
class Requirement:
    """The acceptance requirement: latency at ``percentile`` and sustained FPS.

    Attributes:
        max_latency_ms: Upper bound on end-to-end latency.
        percentile: Which latency percentile must meet it (95 → P95).
        min_fps: Lower bound on sustained completed inferences per second.
    """

    max_latency_ms: float = 500.0
    percentile: float = 95.0
    min_fps: float = 3.0

    def judge(self, latency_ms: float | None, fps: float | None) -> Verdict:
        if latency_ms is None or fps is None:
            return "error"
        latency_ok = latency_ms <= self.max_latency_ms
        fps_ok = fps >= self.min_fps
        if latency_ok and fps_ok:
            return "pass"
        if not latency_ok and not fps_ok:
            return "fail_both"
        return "fail_fps" if latency_ok else "fail_latency"


@dataclass(frozen=True)
class MeasureConfig:
    """How long to measure.

    The measured phase runs until both ``iterations`` calls and ``duration_s``
    seconds are reached, so the percentiles have enough samples and the FPS
    covers a sustained window (thermals, allocator, GC included).
    """

    warmup: int = 10
    iterations: int = 100
    duration_s: float = 60.0


def measure(
    model: LatencyModel,
    obs: Observation,
    config: MeasureConfig,
    sync: Callable[[], None] | None = None,
    monitor: GpuMemoryMonitor | None = None,
) -> dict[str, Any]:
    """Warm up, then run back-to-back inferences (one robot, closed loop).

    Returns:
        Raw samples and derived statistics; see ``docs/benchmark_methodology.md``.
    """
    warmup_ms: list[float] = []
    for _ in range(config.warmup):
        start = time.perf_counter()
        model.infer(obs, StageTimer(sync))
        warmup_ms.append((time.perf_counter() - start) * 1e3)

    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if monitor is not None:
        monitor.start()

    totals: list[float] = []
    stages: dict[str, list[float]] = {}
    action_shape: tuple[int, ...] | None = None
    phase_start = time.perf_counter()
    while True:
        timer = StageTimer(sync)
        start = time.perf_counter()
        actions = model.infer(obs, timer)
        totals.append((time.perf_counter() - start) * 1e3)
        for name, ms in timer.stages.items():
            stages.setdefault(name, []).append(ms)
        if action_shape is None:
            action_shape = tuple(int(d) for d in np.asarray(actions).shape)
        elapsed = time.perf_counter() - phase_start
        if len(totals) >= config.iterations and elapsed >= config.duration_s:
            break

    memory = monitor.stop() if monitor is not None else {}
    fps = len(totals) / elapsed
    return {
        "warmup_ms": [round(v, 2) for v in warmup_ms],
        "cold_start_ms": _r(warmup_ms[0]) if warmup_ms else None,
        "count": len(totals),
        "elapsed_s": round(elapsed, 2),
        "fps": round(fps, 2),
        "latency_ms": latency_stats(totals),
        "stages_ms": {name: latency_stats(v) for name, v in stages.items()},
        "action_shape": list(action_shape or ()),
        "memory": memory,
        "samples_ms": [round(v, 2) for v in totals],
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

_VERDICT_TEXT = {
    "pass": "✅ 达标",
    "fail_latency": "❌ 延迟超标",
    "fail_fps": "❌ 吞吐不足",
    "fail_both": "❌ 延迟与吞吐均不达标",
    "error": "⚠️ 运行失败",
}


def _cell(value: Any) -> str:
    return "-" if value is None else str(value)


def markdown_table(results: list[dict[str, Any]], requirement: Requirement) -> str:
    """One row per model result, ready to paste into BENCHMARKS.md."""
    q = f"P{requirement.percentile:g}"
    lines = [
        f"要求：延迟 {q} ≤ {requirement.max_latency_ms:g} ms，持续吞吐 ≥ {requirement.min_fps:g} 推理/s"
        "（单机器人，原始相机帧 → host 上的动作）",
        "",
        "| 模型 | checkpoint | GPU | 延迟 P50 (ms) | P95 | P99 | max | 吞吐 (推理/s) "
        "| 分段 P50 (ms) | 动作块 | 显存增量 (GB) | 首帧冷启动 (ms) | 结论 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for res in results:
        if res.get("status") != "ok":
            lines.append(
                f"| {res.get('model', '?')} | {_cell(res.get('describe', {}).get('checkpoint'))} "
                f"| - | - | - | - | - | - | - | - | - | - | {_VERDICT_TEXT['error']}: "
                f"{_cell(res.get('error'))[:80]} |"
            )
            continue
        m = res["measurement"]
        lat = m["latency_ms"]
        stages = ", ".join(f"{k} {v['p50']}" for k, v in m["stages_ms"].items()) or "-"
        shape = "×".join(str(d) for d in m["action_shape"]) or "-"
        lines.append(
            f"| {res['model']} | {_cell(res['describe'].get('checkpoint'))} "
            f"| {_cell(res['environment'].get('gpu'))} "
            f"| {_cell(lat['p50'])} | {_cell(lat['p95'])} | {_cell(lat['p99'])} | {_cell(lat['max'])} "
            f"| {m['fps']} | {stages} | {shape} | {_cell(m['memory'].get('peak_delta_gb'))} "
            f"| {_cell(m['cold_start_ms'])} | {_VERDICT_TEXT[res['verdict']]} |"
        )
    return "\n".join(lines) + "\n"
