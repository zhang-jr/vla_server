"""Per-episode records and run summaries for the closed-loop benchmarks.

Metric definitions follow BENCHMARKS.md:

- action jerk RMS: RMS of the third difference of the executed arm trajectory
  (cumulative sum of delta commands; the commands themselves for absolute
  control). Gripper channel excluded.
- chunk boundary discontinuity: mean L2 step between consecutive executed
  actions where a new chunk takes effect, divided by the median step
  elsewhere. 1.0 means boundaries are as smooth as the rest of the trajectory.
- underrun rate: fraction of control steps sent while the chunk buffer was
  empty (the hold action went out instead of a planned one).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

import torch


@dataclass
class EpisodeRecord:
    """Everything measured in one episode.

    Attributes:
        task: Task name.
        episode: Episode index within the task.
        seed: Reset seed (identical across modes for the same episode).
        robot_id: Robot that ran it (realtime fleet runs).
        success: Task completed before the step limit.
        steps: Control steps executed.
        inference_ms: Wall-clock latency per inference, observation → chunk.
        effective_latency_ms: Latency applied to the simulated clock
            (async_sim; equals inference_ms unless injected or scaled).
        vlm_ms / head_ms: Per-module latency where the mode can separate them.
        blocked_ms: Wall time the robot would stand idle waiting for inference.
        underrun_steps: Steps sent with the hold action (empty buffer).
        degraded: Inferences that completed without a new chunk.
        overrun_steps: Realtime control ticks that missed their deadline.
        tick_lag_ms: How late each missed tick was.
        sim_step_ms: Wall time of each env.step (realtime diagnostics).
        wall_s: Episode wall-clock time.
        jerk_rms / boundary_jump / boundary_ratio: Smoothness (see module doc).
    """

    task: str
    episode: int
    seed: int
    robot_id: str = "robot_0"
    success: bool = False
    steps: int = 0
    inference_ms: list[float] = field(default_factory=list)
    effective_latency_ms: list[float] = field(default_factory=list)
    vlm_ms: list[float] = field(default_factory=list)
    head_ms: list[float] = field(default_factory=list)
    blocked_ms: float = 0.0
    underrun_steps: int = 0
    degraded: int = 0
    overrun_steps: int = 0
    tick_lag_ms: list[float] = field(default_factory=list)
    sim_step_ms: list[float] = field(default_factory=list)
    wall_s: float = 0.0
    jerk_rms: float | None = None
    boundary_jump: float | None = None
    boundary_ratio: float | None = None

    def finalize(
        self,
        actions: list[torch.Tensor],
        boundaries: list[int],
        action_is_delta: bool,
        gripper_index: int = -1,
    ) -> None:
        """Compute smoothness from the executed actions; the actions are not kept."""
        if not actions:
            return
        executed = torch.stack([a.float().cpu() for a in actions])
        self.jerk_rms = jerk_rms(executed, action_is_delta, gripper_index)
        self.boundary_jump, self.boundary_ratio = boundary_discontinuity(
            executed, boundaries, gripper_index
        )


def _arm(actions: torch.Tensor, gripper_index: int) -> torch.Tensor:
    keep = [i for i in range(actions.shape[1]) if i != gripper_index % actions.shape[1]]
    return actions[:, keep]


def jerk_rms(actions: torch.Tensor, action_is_delta: bool, gripper_index: int = -1) -> float | None:
    """RMS of the third difference of the executed arm trajectory."""
    arm = _arm(actions, gripper_index)
    trajectory = torch.cumsum(arm, dim=0) if action_is_delta else arm
    if trajectory.shape[0] < 4:
        return None
    jerk = torch.diff(trajectory, n=3, dim=0)
    return float(jerk.pow(2).sum(dim=1).mean().sqrt())


def boundary_discontinuity(
    actions: torch.Tensor,
    boundaries: list[int],
    gripper_index: int = -1,
) -> tuple[float | None, float | None]:
    """(mean L2 step at chunk boundaries, that divided by the median step elsewhere)."""
    arm = _arm(actions, gripper_index)
    if arm.shape[0] < 2:
        return None, None
    steps = torch.linalg.vector_norm(torch.diff(arm, dim=0), dim=1)  # steps[t-1] = |a_t - a_{t-1}|
    at_boundary = {b for b in boundaries if 1 <= b < arm.shape[0]}
    jumps = [float(steps[b - 1]) for b in sorted(at_boundary)]
    others = [float(steps[t - 1]) for t in range(1, arm.shape[0]) if t not in at_boundary]
    if not jumps:
        return None, None
    jump = statistics.fmean(jumps)
    baseline = statistics.median(others) if others else 0.0
    ratio = jump / baseline if baseline > 1e-9 else None
    return jump, ratio


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile; None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval — honest error bars for small episode counts."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return statistics.fmean(present) if present else None


def _round(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(value, digits)


def summarize(
    records: list[EpisodeRecord],
    wall_s: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate episode records into the numbers a BENCHMARKS.md row needs.

    Args:
        records: All episodes of one run.
        wall_s: Wall-clock duration of the whole run (for throughput).
        extra: Run metadata merged into the summary (mode, blend policy, ...).
    """
    n = len(records)
    successes = sum(r.success for r in records)
    lo, hi = wilson_interval(successes, n)
    inference = [v for r in records for v in r.inference_ms]
    vlm = [v for r in records for v in r.vlm_ms]
    head = [v for r in records for v in r.head_ms]
    lag = [v for r in records for v in r.tick_lag_ms]
    sim_step = [v for r in records for v in r.sim_step_ms]
    total_steps = sum(r.steps for r in records)
    busy_s = sum(r.wall_s for r in records)

    per_task: dict[str, list[bool]] = {}
    for r in records:
        per_task.setdefault(r.task, []).append(r.success)

    summary: dict[str, Any] = {
        **(extra or {}),
        "episodes": n,
        "success_rate": _round(successes / n if n else 0.0, 4),
        "success_ci95": [round(lo, 4), round(hi, 4)],
        "per_task_success": {t: round(sum(v) / len(v), 4) for t, v in per_task.items()},
        "inference_count": len(inference),
        "latency_p50_ms": _round(percentile(inference, 50)),
        "latency_p95_ms": _round(percentile(inference, 95)),
        "latency_mean_ms": _round(statistics.fmean(inference) if inference else None),
        "vlm_p50_ms": _round(percentile(vlm, 50)),
        "head_p50_ms": _round(percentile(head, 50)),
        "throughput_inferences_per_s": _round(len(inference) / wall_s if wall_s > 0 else None),
        "steps_total": total_steps,
        "control_hz_achieved": _round(total_steps / busy_s if busy_s > 0 else None),
        "blocked_fraction": _round(
            sum(r.blocked_ms for r in records) / 1000.0 / busy_s if busy_s > 0 else None, 4
        ),
        "underrun_rate": _round(
            sum(r.underrun_steps for r in records) / total_steps if total_steps else None, 4
        ),
        "degraded_inferences": sum(r.degraded for r in records),
        "overrun_rate": _round(
            sum(r.overrun_steps for r in records) / total_steps if total_steps else None, 4
        ),
        "tick_lag_p95_ms": _round(percentile(lag, 95)),
        "sim_step_p95_ms": _round(percentile(sim_step, 95)),
        "jerk_rms": _round(_mean([r.jerk_rms for r in records]), 5),
        "boundary_jump": _round(_mean([r.boundary_jump for r in records]), 5),
        "boundary_ratio": _round(_mean([r.boundary_ratio for r in records]), 3),
        "wall_s": _round(wall_s),
    }
    if torch.cuda.is_available():
        summary["gpu_peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    return summary


def records_to_json(records: list[EpisodeRecord]) -> list[dict[str, Any]]:
    return [asdict(r) for r in records]


# Columns of the markdown table written next to summary.json.
_TABLE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("run", "run"),
    ("success_rate", "成功率"),
    ("success_ci95", "95% CI"),
    ("latency_p50_ms", "延迟 P50 (ms)"),
    ("latency_p95_ms", "延迟 P95 (ms)"),
    ("vlm_p50_ms", "VLM P50"),
    ("head_p50_ms", "Head P50"),
    ("throughput_inferences_per_s", "推理/s"),
    ("blocked_fraction", "阻塞占比"),
    ("underrun_rate", "欠载率"),
    ("overrun_rate", "tick 超时率"),
    ("jerk_rms", "jerk RMS"),
    ("boundary_ratio", "边界不连续度"),
    ("gpu_peak_mem_gb", "显存峰值 (GB)"),
)


def markdown_table(summaries: list[dict[str, Any]]) -> str:
    """Summaries as a markdown table, ready to paste into BENCHMARKS.md."""
    header = "| " + " | ".join(title for _, title in _TABLE_COLUMNS) + " |"
    rule = "|" + "|".join("---" for _ in _TABLE_COLUMNS) + "|"
    rows = []
    for s in summaries:
        cells = []
        for key, _ in _TABLE_COLUMNS:
            value = s.get(key)
            if value is None:
                cells.append("-")
            elif key == "success_ci95":
                cells.append(f"[{value[0]:.2f}, {value[1]:.2f}]")
            elif key in ("success_rate", "blocked_fraction", "underrun_rate", "overrun_rate"):
                cells.append(f"{value:.1%}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *rows])
