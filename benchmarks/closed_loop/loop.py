"""ROS-free core of the closed-loop benchmark: client, step timing, hold guard, stats.

Runs on the robot host's system Python 3.8 (ROS2 Foxy) with numpy 1.17, so:
stdlib + numpy only, no 3.9+ syntax at runtime. The ROS node (``arx_x5``)
plugs in as a ``FrameSource`` / ``CommandSink``; tests plug in fakes.

All timestamps are wall-clock seconds (``time.time()``): camera stamps,
client and server share the host clock (the container shares the kernel).
"""

from __future__ import annotations

import socket
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from benchmarks.closed_loop import protocol
from benchmarks.latency.harness import Requirement, latency_stats, percentile

DEFAULT_INSTRUCTION = "pick up the red block and put it in the bowl"
Mode = Literal["dry-run", "hold"]
MODES = ("dry-run", "hold")


# ── Camera frames ─────────────────────────────────────────────────────────────


def decode_image(encoding: str, height: int, width: int, step: int, data: bytes) -> np.ndarray:
    """``sensor_msgs/Image`` fields → RGB uint8 [H, W, 3] (copy-free when possible).

    Raises:
        ValueError: An encoding other than rgb8 / bgr8 (enable only the colour stream).
    """
    if encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"unsupported image encoding {encoding!r} (want rgb8 / bgr8)")
    rows = np.frombuffer(data, dtype=np.uint8).reshape(height, step)
    image = rows[:, : width * 3].reshape(height, width, 3)
    return image[..., ::-1] if encoding == "bgr8" else image


@dataclass
class Snapshot:
    """The newest inputs at one instant.

    Attributes:
        images: Camera name → RGB uint8 [H, W, 3].
        stamps: Camera name → frame timestamp (driver header stamp).
        state: Arm joint positions incl. gripper, or None without arm status.
        state_stamp: Receive time of that status, or None.
    """

    images: dict[str, np.ndarray]
    stamps: dict[str, float]
    state: np.ndarray | None
    state_stamp: float | None


class FrameSource:
    """Newest camera frames + arm state; ``None`` until every camera has a frame."""

    def snapshot(self) -> Snapshot | None:
        raise NotImplementedError


class CommandSink:
    """Publishes a joint target (6 joints + gripper) with ``mode=5``."""

    def publish_joint_target(self, target: np.ndarray) -> None:
        raise NotImplementedError


# ── Policy client ─────────────────────────────────────────────────────────────


class PolicyClient:
    """Blocking client of ``policy_server`` (one request in flight).

    Args:
        address: ``host:port`` of the server.
        timeout_s: Socket timeout per request.
    """

    def __init__(self, address: str, timeout_s: float = 30.0) -> None:
        host, _, port = address.rpartition(":")
        self.sock = protocol.tune(socket.create_connection((host, int(port)), timeout=timeout_s))

    def hello(self) -> dict[str, Any]:
        protocol.send(self.sock, {"op": "hello"})
        header, _ = protocol.recv(self.sock)
        if header.get("error"):
            raise RuntimeError(f"policy server: {header['error']}")
        return header

    def infer(
        self, seq: int, images: dict[str, np.ndarray], state: np.ndarray, instruction: str,
    ) -> tuple[np.ndarray, dict[str, Any], float, float]:
        """One request. Returns (actions, server header, t_send, t_reply)."""
        arrays = {"image/" + name: img for name, img in images.items()}
        arrays["state"] = np.asarray(state, dtype=np.float32)
        message = protocol.encode({"op": "infer", "seq": seq, "instruction": instruction}, arrays)
        t_send = time.time()
        self.sock.sendall(message)
        header, out = protocol.recv(self.sock)
        t_reply = time.time()
        if header.get("error"):
            raise RuntimeError(f"policy server: {header['error']}")
        return out["actions"], header, t_send, t_reply

    def close(self) -> None:
        try:
            protocol.send(self.sock, {"op": "bye"})
        except OSError:
            pass
        self.sock.close()


# ── Safety ────────────────────────────────────────────────────────────────────


class HoldAbortError(RuntimeError):
    """Hold mode stopped publishing: the arm left its pose or its status went stale."""


class HoldGuard:
    """Checks the arm still sits at the held pose before each command.

    Args:
        target: Held joint positions (6 joints + gripper), captured at start.
        max_deviation_rad: Abort when any of the 6 arm joints is further away.
        max_status_age_s: Abort when the newest arm status is older than this.
    """

    def __init__(self, target: np.ndarray, max_deviation_rad: float, max_status_age_s: float) -> None:
        self.target = np.asarray(target, dtype=np.float64).copy()
        self.max_deviation_rad = max_deviation_rad
        self.max_status_age_s = max_status_age_s

    def check(self, state: np.ndarray | None, state_stamp: float | None, now: float) -> float:
        """Returns the max joint deviation (rad); raises ``HoldAbortError`` when unsafe."""
        if state is None or state_stamp is None:
            raise HoldAbortError("no arm status")
        age = now - state_stamp
        if age > self.max_status_age_s:
            raise HoldAbortError(f"arm status is {age * 1e3:.0f} ms old (> {self.max_status_age_s * 1e3:.0f} ms)")
        deviation = float(np.max(np.abs(np.asarray(state[:6], dtype=np.float64) - self.target[:6])))
        if deviation > self.max_deviation_rad:
            raise HoldAbortError(
                f"joint deviation {deviation:.3f} rad > {self.max_deviation_rad:.3f} rad from the held pose"
            )
        return deviation


# ── Loop ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LoopConfig:
    """How the loop runs; measured phase ends when both floors are reached."""

    mode: Mode = "dry-run"
    instruction: str = DEFAULT_INSTRUCTION
    warmup: int = 5
    iterations: int = 100
    duration_s: float = 60.0
    frame_timeout_s: float = 5.0


@dataclass
class LoopResult:
    records: list[dict[str, Any]] = field(default_factory=list)
    warmup_records: list[dict[str, Any]] = field(default_factory=list)
    elapsed_s: float = 0.0
    action_shape: list[int] = field(default_factory=list)
    aborted: str | None = None


def _wait_snapshot(source: FrameSource, timeout_s: float, sleep: Callable[[float], None]) -> Snapshot:
    deadline = time.time() + timeout_s
    while True:
        snap = source.snapshot()
        if snap is not None:
            return snap
        if time.time() > deadline:
            raise TimeoutError(f"no frame from every camera within {timeout_s:g} s")
        sleep(0.005)


def run_step(
    seq: int,
    source: FrameSource,
    client: PolicyClient,
    config: LoopConfig,
    sink: CommandSink | None,
    guard: HoldGuard | None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any], np.ndarray]:
    """One closed-loop step: newest frames → server → (hold) command. Returns (record, actions)."""
    t_start = time.time()
    snap = _wait_snapshot(source, config.frame_timeout_s, sleep)
    state = snap.state if snap.state is not None else np.zeros(7, dtype=np.float32)
    t_obs = time.time()
    actions, server, t_send, t_reply = client.infer(seq, snap.images, state, config.instruction)
    record: dict[str, Any] = {
        "seq": seq,
        "t_start": t_start,
        "t_obs": t_obs,
        "camera_stamps": dict(snap.stamps),
        "state_stamp": snap.state_stamp,
        "t_send": t_send,
        "t_reply": t_reply,
        "server": {k: server[k] for k in ("t_recv", "t_infer_start", "t_infer_end", "t_reply", "stages") if k in server},
        "cmd_sent": False,
    }
    if config.mode == "hold":
        assert sink is not None and guard is not None
        record["hold_deviation_rad"] = guard.check(snap.state, snap.state_stamp, time.time())
        sink.publish_joint_target(guard.target)
        record["cmd_sent"] = True
    record["t_cmd"] = time.time()
    return record, actions


def run_loop(
    source: FrameSource,
    client: PolicyClient,
    config: LoopConfig,
    sink: CommandSink | None = None,
    guard: HoldGuard | None = None,
    sleep: Callable[[float], None] = time.sleep,
    should_stop: Callable[[], bool] = lambda: False,
) -> LoopResult:
    """Warm up, then back-to-back steps until ``iterations`` and ``duration_s`` are both met.

    A ``HoldAbortError`` or ``should_stop()`` ends the run early and is reported in
    ``aborted``; the records so far are kept.
    """
    if config.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {config.mode!r}")
    if config.mode == "hold" and (sink is None or guard is None):
        raise ValueError("hold mode needs a command sink and a hold guard")
    result = LoopResult()
    seq = 0
    try:
        for _ in range(config.warmup):
            record, actions = run_step(seq, source, client, config, sink, guard, sleep)
            result.warmup_records.append(record)
            seq += 1
        phase_start = time.time()
        while True:
            if should_stop():
                result.aborted = "stopped by operator"
                break
            record, actions = run_step(seq, source, client, config, sink, guard, sleep)
            seq += 1
            result.records.append(record)
            if not result.action_shape:
                result.action_shape = [int(d) for d in np.asarray(actions).shape]
            result.elapsed_s = time.time() - phase_start
            if len(result.records) >= config.iterations and result.elapsed_s >= config.duration_s:
                break
    except HoldAbortError as exc:
        result.aborted = f"hold guard: {exc}"
    return result


# ── Statistics ────────────────────────────────────────────────────────────────

# name → (description, function of a record returning seconds or None)
SEGMENTS: dict[str, tuple[str, Callable[[dict[str, Any]], float | None]]] = {
    "frame_age": ("oldest camera stamp → snapshot taken",
                  lambda r: r["t_start"] - min(r["camera_stamps"].values())),
    "client_obs": ("snapshot → request sent (decode + pack)", lambda r: r["t_send"] - r["t_start"]),
    "transport": ("round trip minus server time (TCP + (de)serialize)",
                  lambda r: (r["t_reply"] - r["t_send"]) - (r["server"]["t_reply"] - r["server"]["t_recv"])),
    "server_unpack": ("request received → observation built",
                      lambda r: r["server"]["t_infer_start"] - r["server"]["t_recv"]),
    "server_infer": ("model inference (script-1 timed call)",
                     lambda r: r["server"]["t_infer_end"] - r["server"]["t_infer_start"]),
    "command": ("reply → command published (guard + publish)", lambda r: r["t_cmd"] - r["t_reply"]),
    "loop": ("snapshot → command published", lambda r: r["t_cmd"] - r["t_start"]),
    "e2e": ("oldest camera stamp → command published",
            lambda r: r["t_cmd"] - min(r["camera_stamps"].values())),
}


def segment_ms(records: list[dict[str, Any]], name: str) -> list[float]:
    fn = SEGMENTS[name][1]
    return [fn(r) * 1e3 for r in records]


def summarize(result: LoopResult, requirement: Requirement) -> dict[str, Any]:
    """Per-segment statistics, FPS and the requirement verdict on end-to-end latency."""
    records = result.records
    count = len(records)
    fps = count / result.elapsed_s if result.elapsed_s > 0 else None
    e2e = segment_ms(records, "e2e")
    requirement_latency = percentile(e2e, requirement.percentile)
    verdict = requirement.judge(requirement_latency, fps)
    if result.aborted is not None and count < 2:
        verdict = "error"
    server_stages: dict[str, list[float]] = {}
    for r in records:
        for name, ms in r["server"].get("stages", {}).items():
            server_stages.setdefault(name, []).append(ms)
    return {
        "count": count,
        "elapsed_s": round(result.elapsed_s, 2),
        "fps": None if fps is None else round(fps, 2),
        "segments_ms": {name: latency_stats(segment_ms(records, name)) for name in SEGMENTS},
        "server_stages_ms": {name: latency_stats(v) for name, v in server_stages.items()},
        "action_shape": result.action_shape,
        "commands_sent": sum(1 for r in records if r["cmd_sent"]),
        "requirement_latency_ms": None if requirement_latency is None else round(requirement_latency, 1),
        "verdict": verdict,
        "aborted": result.aborted,
    }


_VERDICT_TEXT = {
    "pass": "✅ 达标",
    "fail_latency": "❌ 延迟超标",
    "fail_fps": "❌ 吞吐不足",
    "fail_both": "❌ 延迟与吞吐均不达标",
    "error": "⚠️ 运行失败",
}


def markdown_row(model: str, mode: Mode, cameras: list[str], summary: dict[str, Any]) -> str:
    seg = summary["segments_ms"]
    parts = ", ".join(
        f"{name} {seg[name]['p50']}"
        for name in ("frame_age", "client_obs", "transport", "server_infer", "command")
    )
    shape = "×".join(str(d) for d in summary["action_shape"]) or "-"
    note = f"（中止：{summary['aborted']}）" if summary["aborted"] else ""
    return (
        f"| {model} | {mode} | {'+'.join(cameras)} | {seg['e2e']['p50']} | {seg['e2e']['p95']} "
        f"| {seg['e2e']['p99']} | {seg['e2e']['max']} | {summary['fps']} | {parts} | {shape} "
        f"| {summary['commands_sent']} | {_VERDICT_TEXT[summary['verdict']]}{note} |"
    )


def markdown_table(rows: list[str], requirement: Requirement) -> str:
    q = f"P{requirement.percentile:g}"
    return "\n".join([
        f"要求：端到端延迟 {q} ≤ {requirement.max_latency_ms:g} ms，持续吞吐 ≥ {requirement.min_fps:g} 推理/s"
        "（相机帧时间戳 → /arm_cmd 发布）",
        "",
        "| 模型 | 模式 | 相机 | 端到端 P50 (ms) | P95 | P99 | max | 吞吐 (推理/s) "
        "| 分段 P50 (ms) | 动作块 | 已发指令 | 结论 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
        *rows,
    ]) + "\n"
