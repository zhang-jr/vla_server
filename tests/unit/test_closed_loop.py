"""CPU tests for the closed-loop benchmark (benchmarks/closed_loop).

ROS-free: a fake frame source / command sink stand in for the ARX X5 node and
the mock model serves over a real local TCP socket. The ROS node itself
(``arx_x5``) is exercised on the robot host.
"""

from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest

from benchmarks.closed_loop import protocol
from benchmarks.closed_loop.loop import (
    HoldAbortError,
    HoldGuard,
    LoopConfig,
    PolicyClient,
    Snapshot,
    decode_image,
    markdown_row,
    run_loop,
    summarize,
)
from benchmarks.closed_loop.policy_server import ObservationAdapter, PolicyServer
from benchmarks.latency.harness import Observation, Requirement
from benchmarks.latency.models import make_model


def test_protocol_round_trips_arrays_over_a_socket():
    a, b = socket.socketpair()
    with a, b:
        image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        state = np.array([0.1, -0.2], dtype=np.float32)
        protocol.send(a, {"op": "infer", "seq": 3}, {"image/wrist": image, "state": state})
        protocol.send(a, {"op": "bye"})
        header, arrays = protocol.recv(b)
        assert header == {"op": "infer", "seq": 3}
        assert np.array_equal(arrays["image/wrist"], image) and arrays["state"].dtype == np.float32
        assert protocol.recv(b) == ({"op": "bye"}, {})
        a.close()
        with pytest.raises(protocol.ProtocolError, match="closed"):
            protocol.recv(b)


def test_decode_image_handles_bgr_and_row_padding():
    rgb = np.random.default_rng(0).integers(0, 255, size=(2, 3, 3), dtype=np.uint8)
    padded = np.zeros((2, 12), dtype=np.uint8)
    padded[:, :9] = rgb.reshape(2, 9)
    assert np.array_equal(decode_image("rgb8", 2, 3, 12, padded.tobytes()), rgb)
    bgr = rgb[..., ::-1].copy()
    assert np.array_equal(decode_image("bgr8", 2, 3, 9, bgr.tobytes()), rgb)
    with pytest.raises(ValueError, match="encoding"):
        decode_image("16UC1", 2, 3, 6, b"\0" * 12)


def _template(slots: list[str], state_dim: int) -> Observation:
    return Observation(
        images={s: np.zeros((2, 2, 3), np.uint8) for s in slots},
        state=np.zeros(state_dim, np.float32), instruction="x",
    )


def test_observation_adapter_pins_drops_and_cycles_cameras():
    template = _template(["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"], 32)
    adapter = ObservationAdapter(template, {"base_0_rgb": "third", "right_wrist_0_rgb": "-"})
    images = {"third": np.full((2, 2, 3), 1, np.uint8), "wrist": np.full((2, 2, 3), 2, np.uint8)}
    obs = adapter.build(images, np.arange(7, dtype=np.float64), "go")
    assert adapter.mapping == {"base_0_rgb": "third", "left_wrist_0_rgb": "third", "right_wrist_0_rgb": None}
    assert set(obs.images) == {"base_0_rgb", "left_wrist_0_rgb"}
    assert obs.state.shape == (32,) and obs.state.dtype == np.float32
    assert obs.state[:7].tolist() == list(range(7)) and not obs.state[7:].any()
    assert obs.instruction == "go"
    short = ObservationAdapter(_template(["primary"], 2), {}).build(images, np.arange(7.0), "go")
    assert short.state.tolist() == [0.0, 1.0] and set(short.images) == {"primary"}
    with pytest.raises(ValueError, match="matches 0 model cameras"):
        ObservationAdapter(template, {"head": "third"})
    lerobot = _template(["observation.images.base_0_rgb", "observation.images.left_wrist_0_rgb"], 8)
    short = ObservationAdapter(lerobot, {"base_0_rgb": "third", "left_wrist_0_rgb": "wrist"})
    assert set(short.build(images, np.zeros(7), "go").images) == set(lerobot.images)
    assert short.mapping == {"observation.images.base_0_rgb": "third",
                             "observation.images.left_wrist_0_rgb": "wrist"}
    with pytest.raises(ValueError, match="client sends only"):
        ObservationAdapter(template, {"base_0_rgb": "head"}).build(images, np.zeros(7), "go")


class FakeSource:
    """Two cameras whose frames were stamped ``age_s`` ago; a settable arm state."""

    def __init__(self, age_s: float = 0.02) -> None:
        self.age_s = age_s
        self.state = np.zeros(7)

    def snapshot(self) -> Snapshot:
        now = time.time()
        frame = np.zeros((48, 64, 3), np.uint8)
        return Snapshot({"wrist": frame, "third": frame}, {"wrist": now - self.age_s, "third": now - self.age_s},
                        self.state.copy(), now)


class FakeSink:
    def __init__(self) -> None:
        self.targets: list[np.ndarray] = []

    def publish_joint_target(self, target: np.ndarray) -> None:
        self.targets.append(np.asarray(target).copy())


@pytest.fixture()
def mock_server():
    model = make_model("mock", {"model_ms": "5", "preprocess_ms": "1", "chunk": "10"})
    template = model.make_observation(np.random.default_rng(0), (48, 64))
    server = PolicyServer(("127.0.0.1", 0), model, ObservationAdapter(template, {}),
                          {"model": "mock", "model_cameras": list(template.images), "state_dim": 7})
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_dry_run_times_every_segment_and_sends_no_command(mock_server):
    client = PolicyClient(mock_server)
    try:
        assert client.hello()["model"] == "mock"
        result = run_loop(FakeSource(age_s=0.02), client,
                          LoopConfig("dry-run", warmup=2, iterations=8, duration_s=0.0))
    finally:
        client.close()
    assert len(result.warmup_records) == 2 and len(result.records) == 8
    assert result.action_shape == [10, 7] and result.aborted is None
    summary = summarize(result, Requirement())
    seg = summary["segments_ms"]
    assert seg["server_infer"]["p50"] >= 6.0
    assert seg["frame_age"]["p50"] >= 20.0
    assert seg["e2e"]["p50"] >= seg["loop"]["p50"] + 19.0
    assert all(seg[name]["p50"] is not None for name in seg)
    assert summary["commands_sent"] == 0 and summary["verdict"] == "pass"
    assert "| mock | dry-run | third+wrist |" in markdown_row("mock", "dry-run", ["third", "wrist"], summary)


def test_hold_publishes_the_held_pose_and_aborts_when_the_arm_moves(mock_server):
    source, sink = FakeSource(), FakeSink()
    target = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.5])
    source.state = target.copy()
    guard = HoldGuard(target, max_deviation_rad=0.05, max_status_age_s=1.0)
    original = sink.publish_joint_target

    def publish_then_drift(t: np.ndarray) -> None:
        original(t)
        if len(sink.targets) == 4:
            source.state[2] += 0.2  # someone pushes the arm

    sink.publish_joint_target = publish_then_drift
    client = PolicyClient(mock_server)
    try:
        result = run_loop(source, client, LoopConfig("hold", warmup=1, iterations=50, duration_s=0.0),
                          sink=sink, guard=guard)
    finally:
        client.close()
    assert len(sink.targets) == 4 and all(np.array_equal(t, target) for t in sink.targets)
    assert result.aborted is not None and "deviation" in result.aborted
    assert len(result.records) == 3
    assert summarize(result, Requirement())["commands_sent"] == 3


def test_hold_guard_rejects_stale_or_missing_status():
    guard = HoldGuard(np.zeros(7), 0.05, 0.2)
    assert guard.check(np.full(7, 0.01), 10.0, 10.1) == pytest.approx(0.01)
    with pytest.raises(HoldAbortError, match="old"):
        guard.check(np.zeros(7), 10.0, 10.5)
    with pytest.raises(HoldAbortError, match="no arm status"):
        guard.check(None, None, 10.0)


def test_hold_mode_requires_sink_and_guard():
    with pytest.raises(ValueError, match="hold mode needs"):
        run_loop(FakeSource(), None, LoopConfig("hold"))  # type: ignore[arg-type]
