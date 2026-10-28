"""ARX X5 closed-loop benchmark — robot-host side (ROS2 Foxy, system Python 3.8).

Subscribes to the camera colour topics (``realsense2_camera``) and the arm
status (``/arm_status``, ``arx5_arm_msg/RobotStatus``), sends the newest frames
+ joint state to ``policy_server`` and, in hold mode, publishes the arm's
starting pose to ``/arm_cmd`` (``RobotCmd``, ``mode=5`` joint position control)
after every inference. Model actions are logged, never sent to the arm.

Modes:
    dry-run  no command is published; arm status is optional (zeros without it).
    hold     needs ``--operator-present``, a live ``/arm_status`` and an
             ``X5Controller`` subscribed to ``/arm_cmd``. Stops publishing as
             soon as the arm leaves the held pose by more than
             ``--max-deviation`` rad or its status goes stale; the controller
             then keeps the last (held) target.

Example (robot host; ``source benchmarks/closed_loop/host_env.sh`` first)::

    python3 -m benchmarks.closed_loop.arx_x5 --server 172.17.0.2:8765 \\
        --camera wrist=/cam/wrist/color/image_raw --camera third=/cam/third/color/image_raw \\
        --mode dry-run --duration 60

Outputs, under ``--out/<timestamp>-<mode>/``: meta.json, steps.jsonl (every
step's timestamps), result.json (statistics, verdict, server description) and
summary.md (one row for BENCHMARKS.md).
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.closed_loop.loop import (
    DEFAULT_INSTRUCTION,
    MODES,
    CommandSink,
    FrameSource,
    HoldGuard,
    LoopConfig,
    PolicyClient,
    Snapshot,
    decode_image,
    markdown_row,
    markdown_table,
    run_loop,
    summarize,
)
from benchmarks.latency.harness import Requirement

ROOT = Path(__file__).resolve().parents[2]
JOINT_TARGET_MODE = 5  # InterfacesThread::state POSITION_CONTROL


def _say(message: str) -> None:
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def _stamp(header: Any) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


class RosBridge(FrameSource, CommandSink):
    """Latest camera frames / arm status from ROS2 callbacks; publishes joint targets.

    Args:
        node: The rclpy node (spun on a background thread by the caller).
        cameras: Client camera name → ``sensor_msgs/Image`` topic.
        status_topic: ``RobotStatus`` topic of the arm controller.
        cmd_topic: ``RobotCmd`` topic; a publisher is created only when ``publish``.
        publish: False in dry-run, so no command can leave this process.
        gripper_scale: ``RobotCmd.gripper`` = status gripper × this. ``X5Controller``'s
            own follower mode does the same (``setCatch(joint_pos[6] * 5)``): the
            status reports the gripper in a different unit than the command takes.
    """

    def __init__(self, node: Any, cameras: dict[str, str], status_topic: str, cmd_topic: str,
                 publish: bool, gripper_scale: float = 5.0) -> None:
        from arx5_arm_msg.msg import RobotCmd, RobotStatus
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image

        self.node = node
        self.cameras = dict(cameras)
        self.cmd_topic = cmd_topic
        self.gripper_scale = gripper_scale
        self._robot_cmd = RobotCmd
        self._lock = threading.Lock()
        self._frames: dict[str, Any] = {}
        self._status: Any | None = None
        self._status_recv: float | None = None
        self.counts: dict[str, int] = {name: 0 for name in cameras}
        self.counts["arm_status"] = 0
        for name, topic in cameras.items():
            node.create_subscription(Image, topic, self._on_image(name), qos_profile_sensor_data)
        node.create_subscription(RobotStatus, status_topic, self._on_status, 10)
        self._pub = node.create_publisher(RobotCmd, cmd_topic, 10) if publish else None

    def _on_image(self, name: str) -> Any:
        def callback(msg: Any) -> None:
            with self._lock:
                self._frames[name] = msg
                self.counts[name] += 1
        return callback

    def _on_status(self, msg: Any) -> None:
        now = time.time()
        with self._lock:
            self._status = msg
            self._status_recv = now
            self.counts["arm_status"] += 1

    def snapshot(self) -> Snapshot | None:
        with self._lock:
            if len(self._frames) < len(self.cameras):
                return None
            frames = dict(self._frames)
            status, status_recv = self._status, self._status_recv
        images = {
            name: decode_image(m.encoding, m.height, m.width, m.step, bytes(m.data))
            for name, m in frames.items()
        }
        state = None if status is None else np.asarray(status.joint_pos, dtype=np.float64)
        return Snapshot(
            images=images,
            stamps={name: _stamp(m.header) for name, m in frames.items()},
            state=state,
            state_stamp=status_recv,
        )

    def latest_status(self) -> Any | None:
        with self._lock:
            return self._status

    def frame_info(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {name: {"topic": self.cameras[name], "encoding": m.encoding,
                           "height": m.height, "width": m.width}
                    for name, m in self._frames.items()}

    def cmd_subscribers(self) -> int:
        return int(self.node.count_subscribers(self.cmd_topic))

    def publish_joint_target(self, target: np.ndarray) -> None:
        if self._pub is None:
            raise RuntimeError("dry-run bridge has no command publisher")
        msg = self._robot_cmd()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.end_pos = [0.0] * 6
        msg.joint_pos = [float(v) for v in target[:6]]
        msg.gripper = float(target[6]) * self.gripper_scale
        msg.mode = JOINT_TARGET_MODE
        self._pub.publish(msg)


def _wait(predicate: Any, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def measure_rates(bridge: RosBridge, seconds: float) -> dict[str, float]:
    before = dict(bridge.counts)
    time.sleep(seconds)
    return {k: round((bridge.counts[k] - before[k]) / seconds, 1) for k in before}


def capture_hold_target(bridge: RosBridge, seconds: float) -> np.ndarray:
    """Mean joint position over ``seconds``; refuses when the arm is moving."""
    samples: list[np.ndarray] = []
    deadline = time.time() + seconds
    while time.time() < deadline:
        status = bridge.latest_status()
        if status is not None:
            samples.append(np.asarray(status.joint_pos, dtype=np.float64))
        time.sleep(0.01)
    if len(samples) < 5:
        raise SystemExit(f"hold: only {len(samples)} arm status samples in {seconds:g} s")
    stack = np.stack(samples)
    spread = float(np.max(stack[:, :6].max(axis=0) - stack[:, :6].min(axis=0)))
    if spread > 0.01:
        raise SystemExit(f"hold: arm is moving (joint spread {spread:.3f} rad over {seconds:g} s); "
                         "let it settle first")
    return stack.mean(axis=0)


def _parse_cameras(pairs: list[str]) -> dict[str, str]:
    cameras: dict[str, str] = {}
    for pair in pairs:
        name, sep, topic = pair.partition("=")
        if not sep or not name or not topic:
            raise SystemExit(f"--camera {pair!r} is not NAME=TOPIC")
        cameras[name] = topic
    if not cameras:
        raise SystemExit("at least one --camera NAME=TOPIC is required")
    return cameras


def _clock_warnings(records: list[dict[str, Any]]) -> list[str]:
    if not records:
        return []
    ages = [(r["t_start"] - min(r["camera_stamps"].values())) * 1e3 for r in records]
    median = float(np.median(ages))
    if median < -5.0 or median > 1000.0:
        return [f"median frame age {median:.0f} ms: camera stamps are probably not on the host "
                "wall clock, so frame_age / e2e are not meaningful"]
    return []


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--server", required=True, help="policy server host:port")
    parser.add_argument("--camera", action="append", default=[], metavar="NAME=TOPIC",
                        help="camera colour topic (repeatable)")
    parser.add_argument("--status-topic", default="/arm_status")
    parser.add_argument("--cmd-topic", default="/arm_cmd")
    parser.add_argument("--mode", choices=MODES, default="dry-run")
    parser.add_argument("--operator-present", action="store_true",
                        help="required for hold: someone is at the arm with the e-stop")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--max-deviation", type=float, default=0.05,
                        help="hold: abort when a joint is this far (rad) from the held pose")
    parser.add_argument("--gripper-scale", type=float, default=5.0,
                        help="hold: RobotCmd.gripper = status gripper x this (X5Controller follower uses 5)")
    parser.add_argument("--max-status-age", type=float, default=0.2,
                        help="hold: abort when the newest arm status is older (s)")
    parser.add_argument("--hold-capture", type=float, default=1.0,
                        help="hold: seconds of arm status averaged into the held pose")
    parser.add_argument("--status-wait", type=float, default=2.0,
                        help="dry-run: seconds to wait for arm status before using zeros")
    parser.add_argument("--max-latency-ms", type=float, default=500.0)
    parser.add_argument("--latency-percentile", type=float, default=95.0)
    parser.add_argument("--min-fps", type=float, default=3.0)
    parser.add_argument("--out", default=str(ROOT / "benchmarks" / "results" / "closed_loop"))
    args = parser.parse_args(argv)
    if args.mode == "hold" and not args.operator_present:
        parser.error("hold mode publishes to the arm: pass --operator-present once someone is at the e-stop")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cameras = _parse_cameras(args.camera)
    requirement = Requirement(args.max_latency_ms, args.latency_percentile, args.min_fps)
    config = LoopConfig(args.mode, args.instruction, args.warmup, args.iterations, args.duration)
    run_dir = Path(args.out) / (time.strftime("%Y%m%d-%H%M%S") + "-" + args.mode)
    run_dir.mkdir(parents=True, exist_ok=True)  # fail before touching the robot, not after the run

    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node

    rclpy.init()
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    node = Node("vla_closed_loop_bench")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, name="ros-spin", daemon=True)
    hold = args.mode == "hold"
    bridge = RosBridge(node, cameras, args.status_topic, args.cmd_topic, publish=hold,
                       gripper_scale=args.gripper_scale)
    spinner.start()
    client: PolicyClient | None = None
    try:
        if not _wait(lambda: bridge.snapshot() is not None, 10.0):
            raise SystemExit(f"no frames within 10 s on {cameras} (are the cameras launched?)")
        has_status = _wait(lambda: bridge.latest_status() is not None,
                           args.status_wait if not hold else 5.0)
        if hold and not has_status:
            raise SystemExit(f"hold: no {args.status_topic} within 5 s (is X5Controller running?)")
        rates = measure_rates(bridge, 2.0)
        _say(f"inputs: {bridge.frame_info()}; rates (Hz) {rates}; arm status {'yes' if has_status else 'no'}")

        guard = None
        if hold:
            if bridge.cmd_subscribers() < 1:
                raise SystemExit(f"hold: nothing subscribes to {args.cmd_topic} (is X5Controller running?)")
            target = capture_hold_target(bridge, args.hold_capture)
            guard = HoldGuard(target, args.max_deviation, args.max_status_age)
            _say(f"hold target (rad, gripper last): {np.round(target, 4).tolist()}")

        client = PolicyClient(args.server)
        hello = client.hello()
        _say(f"policy server: {hello['model']} cameras {hello['model_cameras']} state {hello['state_dim']}")
        _say(f"running {args.mode}: warmup {config.warmup}, then >={config.iterations} steps "
             f"and >={config.duration_s:g} s (Ctrl-C stops early) ...")
        result = run_loop(bridge, client, config, sink=bridge if hold else None, guard=guard,
                          should_stop=stop.is_set)
        rates_after = measure_rates(bridge, 1.0)
    finally:
        if client is not None:
            client.close()
        executor.shutdown(timeout_sec=1.0)
        node.destroy_node()
        rclpy.shutdown()

    summary = summarize(result, requirement)
    warnings = _clock_warnings(result.records)
    meta = {"argv": sys.argv, "args": vars(args), "requirement": vars(requirement)}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(run_dir / "steps.jsonl", "w", encoding="utf-8") as fh:
        for phase, records in (("warmup", result.warmup_records), ("measure", result.records)):
            for record in records:
                fh.write(json.dumps({"phase": phase, **record}) + "\n")
    out = {
        "mode": args.mode,
        "server": hello,
        "inputs": {"cameras": bridge.frame_info(), "rates_hz_before": rates, "rates_hz_after": rates_after,
                   "arm_status": has_status},
        "hold_target": None if guard is None else guard.target.tolist(),
        "summary": summary,
        "warnings": warnings,
    }
    (run_dir / "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    row = markdown_row(hello["model"], args.mode, sorted(cameras), summary)
    table = markdown_table([row], requirement)
    (run_dir / "summary.md").write_text(table, encoding="utf-8")
    seg = summary["segments_ms"]
    _say(f"e2e P50 {seg['e2e']['p50']} ms, P95 {seg['e2e']['p95']} ms, {summary['fps']} inferences/s "
         f"over {summary['count']} steps -> {summary['verdict']}"
         + (f" (aborted: {summary['aborted']})" if summary["aborted"] else ""))
    for warning in warnings:
        _say("WARNING: " + warning)
    _say(f"results: {run_dir}")
    return 0 if summary["verdict"] != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
