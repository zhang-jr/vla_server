"""Policy server for the closed-loop benchmark — container side.

Loads one script-1 model adapter (``benchmarks.latency.models``) in its own env
group's interpreter, warms it up, then answers ``infer`` requests from the
robot client over TCP (``protocol``). Each reply carries the server-side
timestamps so the client can split the round trip into transport and model.

Camera mapping: the model's camera slots (its ``make_observation`` keys) are
filled from the client's named cameras. ``--camera-map SLOT=CLIENT`` pins a
slot, ``SLOT=-`` leaves it out (LeRobot π0 / π0.5 / SmolVLA pad a missing
camera with an empty, masked image; compute is unchanged); unpinned slots take
the client cameras in sorted order, cycling. State is zero-padded / truncated
to the model's state size.

Example (inside the container)::

    python -m benchmarks.closed_loop.policy_server --model pi05 --port 8765 \\
        --opt pi05.repo=/models/pi05_base --opt pi05.tokenizer=/models/paligemma-3b-pt-224 \\
        --camera-map base_0_rgb=third --camera-map left_wrist_0_rgb=wrist \\
        --camera-map right_wrist_0_rgb=-
"""

from __future__ import annotations

import argparse
import logging
import os
import socketserver
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.closed_loop import protocol
from benchmarks.latency.harness import GpuMemoryMonitor, Observation, StageTimer
from benchmarks.latency.models import ENV_GROUPS, make_model
from benchmarks.latency.run import _say, environment_info, parse_model_options, python_for

ROOT = Path(__file__).resolve().parents[2]
DROP = "-"


class ObservationAdapter:
    """Builds a model ``Observation`` from the client's named cameras and arm state.

    Args:
        template: The model's own ``make_observation`` result (slots, state size, extras).
        camera_map: Model camera slot → client camera name, or ``"-"`` to leave the slot out.
    """

    def __init__(self, template: Observation, camera_map: dict[str, str]) -> None:
        self.template = template
        self.camera_map = {self._slot(key): camera for key, camera in camera_map.items()}
        self._resolved: dict[str, str | None] | None = None
        self._client_cameras: list[str] = []

    def _slot(self, key: str) -> str:
        """Full slot name for ``key``; ``base_0_rgb`` also matches ``observation.images.base_0_rgb``."""
        matches = [s for s in self.template.images if s == key or s.endswith("." + key)]
        if len(matches) != 1:
            raise ValueError(f"--camera-map {key!r} matches {len(matches)} model cameras; "
                             f"the model has {sorted(self.template.images)}")
        return matches[0]

    def resolve(self, client_cameras: list[str]) -> dict[str, str | None]:
        """Model slot → client camera (None = left out), for this set of client cameras."""
        cameras = sorted(client_cameras)
        if not cameras:
            raise ValueError("the request carries no camera image")
        resolved: dict[str, str | None] = {}
        free = iter(cameras * len(self.template.images))
        for slot in self.template.images:
            pinned = self.camera_map.get(slot)
            if pinned == DROP:
                resolved[slot] = None
            elif pinned is not None:
                if pinned not in cameras:
                    raise ValueError(f"--camera-map {slot}={pinned}: client sends only {cameras}")
                resolved[slot] = pinned
            else:
                resolved[slot] = next(free)
        if all(v is None for v in resolved.values()):
            raise ValueError("--camera-map leaves out every model camera")
        return resolved

    def build(self, images: dict[str, np.ndarray], state: np.ndarray, instruction: str) -> Observation:
        cameras = sorted(images)
        if self._resolved is None or cameras != self._client_cameras:
            self._resolved = self.resolve(cameras)
            self._client_cameras = cameras
        slots = {slot: images[cam] for slot, cam in self._resolved.items() if cam is not None}
        model_state = np.zeros_like(self.template.state)
        n = min(model_state.shape[0], np.asarray(state).shape[0])
        model_state[:n] = np.asarray(state, dtype=model_state.dtype)[:n]
        extra = dict(self.template.extra)
        if "example" in extra:  # openpi: images live inside the example dict
            example = dict(extra["example"])
            example.update(slots)
            extra["example"] = example
        return Observation(images=slots, state=model_state, instruction=instruction, extra=extra)

    @property
    def mapping(self) -> dict[str, str | None] | None:
        return self._resolved


class PolicyServer(socketserver.TCPServer):
    """One client at a time: a robot runs one request in flight."""

    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], model: Any, adapter: ObservationAdapter,
                 hello: dict[str, Any]) -> None:
        super().__init__(address, _Handler)
        self.model = model
        self.adapter = adapter
        self.hello = hello
        self.sync = getattr(model, "sync", None)

    def handle_infer(self, header: dict[str, Any], arrays: dict[str, np.ndarray], t_recv: float) -> bytes:
        images = {k[len("image/"):]: v for k, v in arrays.items() if k.startswith("image/")}
        obs = self.adapter.build(images, arrays.get("state", np.zeros(0, np.float32)),
                                 header.get("instruction", ""))
        timer = StageTimer(self.sync)
        t_infer_start = time.time()
        actions = np.asarray(self.model.infer(obs, timer), dtype=np.float32)
        t_infer_end = time.time()
        reply = {
            "op": "infer", "seq": header.get("seq"), "t_recv": t_recv,
            "t_infer_start": t_infer_start, "t_infer_end": t_infer_end,
            "stages": {k: round(v, 3) for k, v in timer.stages.items()},
        }
        reply["t_reply"] = time.time()
        return protocol.encode(reply, {"actions": actions})


class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server: PolicyServer = self.server  # type: ignore[assignment]
        sock = protocol.tune(self.request)
        _say(f"client connected: {self.client_address[0]}:{self.client_address[1]}")
        while True:
            try:
                header, arrays = protocol.recv(sock)
            except (protocol.ProtocolError, OSError):
                break
            t_recv = time.time()
            op = header.get("op")
            try:
                if op == "hello":
                    protocol.send(sock, {**server.hello, "op": "hello"})
                elif op == "infer":
                    sock.sendall(server.handle_infer(header, arrays, t_recv))
                elif op == "bye":
                    break
                else:
                    protocol.send(sock, {"op": op, "error": f"unknown op {op!r}"})
            except Exception as exc:  # reported to the client, which aborts its run
                _say(traceback.format_exc())
                protocol.send(sock, {"op": op, "seq": header.get("seq"),
                                     "error": f"{type(exc).__name__}: {exc}"})
        _say(f"client disconnected (camera mapping {server.adapter.mapping})")


def parse_camera_map(pairs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in pairs:
        slot, sep, camera = pair.partition("=")
        if not sep or not slot or not camera:
            raise SystemExit(f"--camera-map {pair!r} is not <model camera>=<client camera | ->")
        out[slot] = camera
    return out


def _reexec_in_env(model: str) -> None:
    """Restart under the model's env-group interpreter (``/opt/venvs/<group>``) if needed."""
    target = python_for(ENV_GROUPS[model])
    if os.path.realpath(target) == os.path.realpath(sys.executable):
        return
    _say(f"[{model}] re-running under {target}")
    os.chdir(ROOT)
    os.execv(target, [target, "-m", "benchmarks.closed_loop.policy_server", "--no-reexec", *sys.argv[1:]])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help=f"one of {', '.join(ENV_GROUPS)}")
    parser.add_argument("--opt", action="append", default=[], metavar="MODEL.KEY=VALUE",
                        help="model option (repeatable), as in benchmarks.latency.run")
    parser.add_argument("--camera-map", action="append", default=[], metavar="SLOT=CAMERA",
                        help="model camera slot → client camera name, or SLOT=- to leave it out")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--warmup", type=int, default=5, help="untimed inferences before serving")
    parser.add_argument("--image-hw", type=int, nargs=2, default=(480, 640), metavar=("H", "W"),
                        help="frame size of the warm-up observation (match the cameras)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-reexec", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    args = parse_args(argv)
    if args.model not in ENV_GROUPS:
        raise SystemExit(f"unknown model {args.model!r}; expected one of {', '.join(ENV_GROUPS)}")
    if not args.no_reexec and argv is None:
        _reexec_in_env(args.model)
    options = parse_model_options(args.opt, [args.model])[args.model]
    camera_map = parse_camera_map(args.camera_map)

    np.random.seed(args.seed)
    monitor = GpuMemoryMonitor()
    model = make_model(args.model, options)
    _say(f"[{args.model}] loading ...")
    model.load()
    template = model.make_observation(np.random.default_rng(args.seed), tuple(args.image_hw))
    adapter = ObservationAdapter(template, camera_map)
    sync = getattr(model, "sync", None)
    for i in range(args.warmup):
        start = time.perf_counter()
        model.infer(template, StageTimer(sync))
        _say(f"[{args.model}] warmup {i + 1}/{args.warmup}: {(time.perf_counter() - start) * 1e3:.1f} ms")

    hello = {
        "model": args.model,
        "options": options,
        "describe": model.describe(),
        "model_cameras": list(template.images),
        "camera_map": camera_map,
        "state_dim": int(template.state.shape[0]),
        "environment": environment_info(monitor),
    }
    with PolicyServer((args.host, args.port), model, adapter, hello) as server:
        _say(f"[{args.model}] serving on {args.host}:{args.port} "
             f"(model cameras {list(template.images)}, state {hello['state_dim']})")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            _say("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
