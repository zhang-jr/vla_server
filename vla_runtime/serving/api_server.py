"""Policy server: robots connect over WebSocket, requests run through the fleet.

The wire protocol is openpi's (ADR-016), so an unmodified openpi client
(``openpi_client.websocket_client_policy.WebsocketClientPolicy``) can drive it:

1. The client connects to ``ws://host:port/`` (optionally ``?robot_id=<id>``).
   The server registers one robot in the fleet for the connection and sends a
   metadata map first.
2. Each request is one msgpack map (``vla_runtime.serving.wire``): camera
   images as HWC uint8 arrays under the configured keys, the state vector, the
   prompt; for RTC policies optionally ``prev_actions`` ``[n, action_dim]``
   (the actions still queued on the robot, in the executed space) and
   ``inference_delay_steps``.
3. The response is a map::

       {"actions": float32 [chunk_size, action_dim],
        "server_timing": {"infer_ms": ..., "prev_total_ms": ...},
        "runtime": {"robot_id", "request_id", "degraded", "rtc", ...}}

   ``runtime`` carries the chunk's scalar metadata (``head_batch_size``,
   ``replan_threshold`` under admission control, ...). A coupled policy whose
   backbone timed out returns zero actions with ``runtime["degraded"]`` set;
   the robot keeps executing what it has queued.
4. If a request fails, the server sends the traceback as a string and closes
   the connection with 1011, as openpi's server does. A robot the fleet
   cannot take (``max_robots``, admission control) is closed with 1013 (try
   again later), a duplicate ``robot_id`` with 1008.

``GET /healthz`` answers 200 for load balancers and health checks.

Run it with ``python -m vla_runtime.serving.api_server`` (``--help``), or
embed ``PolicyServer`` around any ``Policy``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import http
import itertools
import time
import traceback
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlsplit

import numpy as np
import torch
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.frames import CloseCode
from websockets.http11 import Request, Response

from vla_runtime.action_head.base import ActionChunk, ActionHeadContext, RobotState
from vla_runtime.errors import BudgetExceededError, EdgeResourceError
from vla_runtime.observability.metrics import get_metrics
from vla_runtime.observability.tracer import configure_logging, get_tracer
from vla_runtime.policy import Policy
from vla_runtime.serving.fleet import FleetConfig, FleetScheduler
from vla_runtime.serving.wire import packb, unpackb
from vla_runtime.vlm.base import ActionRequest

_tracer = get_tracer("serving.api_server")
_metrics = get_metrics()

# openpi's LIBERO example client: robot camera name → request key.
OPENPI_LIBERO_IMAGE_KEYS = {
    "agentview": "observation/image",
    "wrist": "observation/wrist_image",
}


class WireFormatError(ValueError):
    """A request does not match the server's wire format.

    A client mistake, reported back to that client (traceback, close 1011);
    it never reaches the fleet.
    """


# ── Request / response conversion ─────────────────────────────────────────────

def image_to_tensor(value: Any, key: str) -> torch.Tensor:
    """HWC (or CHW) image, uint8 or float in [0, 1] → CHW float32 in [0, 1].

    Raises:
        WireFormatError: If the array is not a 3-D image with 1 or 3 channels.
    """
    array = np.asarray(value)
    if array.ndim != 3:
        raise WireFormatError(f"{key!r}: expected a 3-D image, got shape {array.shape}")
    if array.shape[-1] in (1, 3):
        channels_last = True
    elif array.shape[0] in (1, 3):
        channels_last = False
    else:
        raise WireFormatError(f"{key!r}: no 1- or 3-channel axis in shape {array.shape}")
    tensor = torch.from_numpy(array.astype(np.float32))  # a copy: wire arrays are read-only
    if channels_last:
        tensor = tensor.permute(2, 0, 1).contiguous()
    if array.dtype == np.uint8:
        tensor.div_(255.0)
    return tensor


def state_to_robot_state(value: Any, key: str) -> RobotState:
    """One state vector → RobotState.

    The vector becomes ``extra["policy_state"]``, which prefix-state models
    (π0.5, SmolVLA) read in the dataset's convention. Heads that read the flat
    state get it as ``joint_positions`` with zero velocities and gripper.

    Raises:
        WireFormatError: If the value is not a numeric vector.
    """
    try:
        vector = torch.from_numpy(np.array(value, dtype=np.float32).reshape(-1))
    except (TypeError, ValueError) as exc:
        raise WireFormatError(f"{key!r}: expected a numeric state vector ({exc})") from exc
    return RobotState(
        joint_positions=vector,
        joint_velocities=torch.zeros_like(vector),
        gripper_state=torch.zeros(1),
        extra={"policy_state": vector.clone()},
    )


@dataclass(frozen=True)
class WireFormat:
    """Where a request carries each input.

    Attributes:
        image_keys: Robot camera name (as the policy's camera map knows it) →
            request key. Defaults to openpi's LIBERO client keys.
        state_key: Request key of the state vector.
        prompt_key: Request key of the instruction.
    """

    image_keys: Mapping[str, str] = field(default_factory=lambda: dict(OPENPI_LIBERO_IMAGE_KEYS))
    state_key: str = "observation/state"
    prompt_key: str = "prompt"

    def to_request(self, obs: Mapping[str, Any], frame_id: int) -> ActionRequest:
        """A decoded request map → ActionRequest.

        Raises:
            WireFormatError: If a configured key is missing or malformed.
        """
        missing = [k for k in (*self.image_keys.values(), self.state_key, self.prompt_key) if k not in obs]
        if missing:
            raise WireFormatError(f"request lacks {missing}; it has {sorted(map(str, obs))}")
        prompt = obs[self.prompt_key]
        if not isinstance(prompt, str):
            raise WireFormatError(f"{self.prompt_key!r}: expected a string, got {type(prompt).__name__}")
        return ActionRequest(
            request_id=uuid.uuid4().hex[:12],
            images={camera: image_to_tensor(obs[key], key) for camera, key in self.image_keys.items()},
            instruction=prompt,
            robot_state=state_to_robot_state(obs[self.state_key], self.state_key),
            frame_id=frame_id,
        )

    @staticmethod
    def head_context(obs: Mapping[str, Any]) -> ActionHeadContext | None:
        """The optional RTC fields → ActionHeadContext, or None if absent.

        Raises:
            WireFormatError: If ``prev_actions`` is not a 2-D array or the
                delay is negative.
        """
        prev = obs.get("prev_actions")
        if prev is None:
            return None
        actions = np.array(prev, dtype=np.float32)
        if actions.ndim != 2:
            raise WireFormatError(f"'prev_actions': expected [n, action_dim], got {actions.shape}")
        delay = int(obs.get("inference_delay_steps", 0))
        if delay < 0:
            raise WireFormatError(f"'inference_delay_steps' must be >= 0, got {delay}")
        return ActionHeadContext(prev_actions=torch.from_numpy(actions), inference_delay_steps=delay)


def chunk_to_response(
    chunk: ActionChunk | None,
    action_dim: int,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """An ActionChunk (or none, after a coupled timeout) → the response map."""
    if chunk is None:
        actions = np.zeros((0, action_dim), dtype=np.float32)
    else:
        actions = chunk.actions.detach().float().cpu().numpy()
        for key, value in chunk.metadata.items():
            if isinstance(value, (bool, int, float, str)) and key not in runtime:
                runtime[key] = value
    return {"actions": actions, "runtime": runtime}


# ── Server ────────────────────────────────────────────────────────────────────

class PolicyServer:
    """One policy served to many robots over WebSocket, through a FleetScheduler.

    Each connection is one robot: registered on connect (a head spawned from
    the policy), deregistered on disconnect. Requests from all connections
    share the fleet, so heads batch across robots (ADR-022).

    Args:
        policy: The policy to serve.
        wire: Request layout; defaults to openpi's LIBERO keys.
        fleet_config: Fleet limits and budgets.
        host: Interface to bind.
        port: Port to bind; 0 picks a free one (see ``port``).
        metadata: Extra entries for the handshake map (e.g. ``action_mode``).
    """

    def __init__(
        self,
        policy: Policy,
        wire: WireFormat | None = None,
        *,
        fleet_config: FleetConfig | None = None,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.policy = policy
        self.fleet = FleetScheduler.from_policy(policy, fleet_config or FleetConfig())
        self.wire = wire or WireFormat()
        self._host = host
        self._port = port
        self._metadata = dict(metadata or {})
        self._ids = itertools.count()
        self._server: Server | None = None

    @property
    def port(self) -> int:
        """The bound port (useful with ``port=0``)."""
        if self._server is None:
            raise RuntimeError("server not started")
        return int(next(iter(self._server.sockets)).getsockname()[1])

    async def start(self) -> None:
        """Start the fleet, then listen."""
        await self.fleet.start()
        self._server = await serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=self._process_request,
        )
        _tracer.event("api_server_started", host=self._host, port=self.port, policy=self.policy.name)

    async def stop(self) -> None:
        """Close every connection, stop listening, then stop the fleet."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        await self.fleet.stop()
        _tracer.event("api_server_stopped", policy=self.policy.name)

    async def serve_forever(self) -> None:
        """Start and serve until cancelled."""
        await self.start()
        try:
            assert self._server is not None
            await self._server.serve_forever()
        finally:
            await self.stop()

    def metadata(self, robot_id: str) -> dict[str, Any]:
        """The handshake map sent when a robot connects."""
        head = self.policy.head_prototype
        return {
            "server": "vla-runtime",
            "policy": self.policy.name,
            "robot_id": robot_id,
            "chunk_size": head.chunk_size,
            "action_dim": head.action_dim,
            "supports_rtc": self.policy.capabilities.supports_rtc,
            "replan_threshold": self.fleet.replan_threshold,
            "image_keys": dict(self.wire.image_keys),
            "state_key": self.wire.state_key,
            "prompt_key": self.wire.prompt_key,
            **self._metadata,
        }

    # ── Connection handling ───────────────────────────────────────────────────

    def _process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        if urlsplit(request.path).path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None

    def _robot_id(self, connection: ServerConnection) -> str:
        assert connection.request is not None
        query = parse_qs(urlsplit(connection.request.path).query)
        ids = query.get("robot_id")
        return ids[0] if ids else f"robot-{next(self._ids)}"

    async def _handler(self, connection: ServerConnection) -> None:
        robot_id = self._robot_id(connection)
        try:
            await self.fleet.register_robot(robot_id)
        except (EdgeResourceError, BudgetExceededError) as exc:
            _tracer.event("api_robot_rejected", robot_id=robot_id, reason=type(exc).__name__)
            await connection.close(CloseCode.TRY_AGAIN_LATER, f"fleet cannot take {robot_id}")
            return
        except ValueError as exc:
            _tracer.event("api_robot_rejected", robot_id=robot_id, reason=str(exc))
            await connection.close(CloseCode.POLICY_VIOLATION, f"robot_id {robot_id!r} is taken")
            return
        _tracer.event("api_robot_connected", robot_id=robot_id, remote=str(connection.remote_address))
        try:
            await self._serve_robot(connection, robot_id)
        finally:
            await self.fleet.deregister_robot(robot_id)
            _tracer.event("api_robot_disconnected", robot_id=robot_id)

    async def _serve_robot(self, connection: ServerConnection, robot_id: str) -> None:
        frame = 0
        prev_total_ms: float | None = None
        try:
            await connection.send(packb(self.metadata(robot_id)))
            async for message in connection:
                t0 = time.perf_counter()
                response = await self._infer(robot_id, message, frame)
                timing = {"infer_ms": (time.perf_counter() - t0) * 1000.0}
                if prev_total_ms is not None:
                    timing["prev_total_ms"] = prev_total_ms
                response["server_timing"] = timing
                await connection.send(packb(response))
                prev_total_ms = (time.perf_counter() - t0) * 1000.0
                frame += 1
        except ConnectionClosed as exc:
            _tracer.event("api_connection_lost", robot_id=robot_id, code=exc.rcvd.code if exc.rcvd else None)
        except Exception as exc:
            _metrics.increment("api_request_failed")
            _tracer.event(
                "api_request_failed", robot_id=robot_id, frame=frame, error=f"{type(exc).__name__}: {exc}"
            )
            with contextlib.suppress(ConnectionClosed):
                await connection.send(traceback.format_exc())
                await connection.close(CloseCode.INTERNAL_ERROR, "Internal server error")

    async def _infer(self, robot_id: str, message: str | bytes, frame: int) -> dict[str, Any]:
        if isinstance(message, str):
            raise WireFormatError("requests must be binary msgpack frames, got text")
        obs = unpackb(message)
        if not isinstance(obs, Mapping):
            raise WireFormatError(f"a request must be a msgpack map, got {type(obs).__name__}")
        request = self.wire.to_request(obs, frame_id=frame)
        rtc = self.policy.capabilities.supports_rtc
        context = self.wire.head_context(obs) if rtc else None
        handle = await self.fleet.submit(robot_id, request, head_context=context)
        chunk: ActionChunk | None = None
        async for produced in self.fleet.stream(handle):
            chunk = produced
        _metrics.increment("api_requests")
        runtime: dict[str, Any] = {
            "robot_id": robot_id,
            "request_id": request.request_id,
            "degraded": handle.degraded,
            "rtc": context is not None,
        }
        return chunk_to_response(chunk, self.policy.head_prototype.action_dim, runtime)


# ── Command line ──────────────────────────────────────────────────────────────

def _pairs(values: Sequence[str], flag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in values:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"{flag} {item!r} is not key=value")
        out[key.strip()] = value.strip()
    return out


def build_policy(
    name: str, options: Mapping[str, str], wire: WireFormat
) -> tuple[Policy, Callable[[], None]]:
    """Build a named policy for serving → (policy, close).

    ``mock`` takes ``chunk_size``, ``action_dim``, ``state_len`` (the wire state
    vector's length), ``latency_ms``, ``jitter_ms``, ``embedding_dim``,
    ``n_steps``. The LeRobot models take the options of
    ``vla_runtime.models.registry``. All take the RTC options, ``head_hz``
    (default 10) and the LeRobot models ``vlm_budget_ms`` (default 2000).

    Raises:
        ValueError: Unknown policy or option, or a camera the wire format lacks.
    """
    from vla_runtime.models.registry import MODEL_NAMES, load_model, pop_rtc_options

    opts = dict(options)
    rtc, rtc_config = pop_rtc_options(opts)
    head_hz = float(opts.pop("head_hz", "10"))
    if name == "mock":
        from vla_runtime.models.mock import mock_policy

        state_len = int(opts.pop("state_len", "8"))
        # The head reads the flat RobotState this wire state becomes.
        state_dim = state_to_robot_state(np.zeros(state_len), wire.state_key).flat_dim
        policy = mock_policy(
            num_views=len(wire.image_keys),
            action_dim=int(opts.pop("action_dim", "7")),
            state_dim=state_dim,
            chunk_size=int(opts.pop("chunk_size", "16")),
            embedding_dim=int(opts.pop("embedding_dim", "256")),
            n_steps=int(opts.pop("n_steps", "10")),
            latency_ms=float(opts.pop("latency_ms", "0")),
            jitter_ms=float(opts.pop("jitter_ms", "0")),
            frequency_hz=head_hz,
            rtc=rtc,
            rtc_config=rtc_config,
        )
        if opts:
            raise ValueError(f"unknown --policy-opt for mock: {sorted(opts)}")
        return policy, lambda: None
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown policy {name!r}; expected mock or one of {list(MODEL_NAMES)}")

    from vla_runtime.models.prefix_kv import prefix_kv_policy

    vlm_budget_ms = float(opts.pop("vlm_budget_ms", "2000"))

    def check_cameras(cameras: Mapping[str, str]) -> None:
        missing = set(cameras) - set(wire.image_keys)
        if missing:
            raise ValueError(
                f"the model reads cameras {sorted(missing)} that no --image-key maps; "
                f"mapped: {sorted(wire.image_keys)}"
            )

    model, _info = load_model(name, opts, check_cameras=check_cameras)
    policy, backbone = prefix_kv_policy(
        model, vlm_budget_ms=vlm_budget_ms, frequency_hz=head_hz, rtc=rtc, rtc_config=rtc_config
    )
    return policy, backbone.close


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m vla_runtime.serving.api_server",
        description="Serve one VLA policy to many robots over WebSocket (openpi wire protocol).",
    )
    parser.add_argument("--policy", required=True, help="mock, smolvla or pi05")
    parser.add_argument("--policy-opt", action="append", default=[], metavar="KEY=VALUE",
                        help="model / RTC / head options (vla_runtime.models.registry)")
    parser.add_argument("--image-key", action="append", default=[], metavar="CAMERA=KEY",
                        help="robot camera name → request key (default: openpi LIBERO keys)")
    parser.add_argument("--state-key", default="observation/state")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-robots", type=int, default=FleetConfig.max_robots)
    parser.add_argument("--max-head-batch", type=int, default=FleetConfig.max_head_batch,
                        help="rows per batched head pass; 1 disables head batching")
    parser.add_argument("--vlm-budget-ms", type=float, default=2000.0,
                        help="fleet backbone timeout per request")
    parser.add_argument("--head-budget-ms", type=float, default=1000.0)
    parser.add_argument("--metadata", action="append", default=[], metavar="KEY=VALUE",
                        help="extra handshake entries, e.g. action_mode=qpos")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point (``vla-serve``)."""
    args = _parse_args(argv)
    configure_logging()
    image_keys = _pairs(args.image_key, "--image-key") or dict(OPENPI_LIBERO_IMAGE_KEYS)
    wire = WireFormat(image_keys=image_keys, state_key=args.state_key, prompt_key=args.prompt_key)
    policy, close = build_policy(args.policy, _pairs(args.policy_opt, "--policy-opt"), wire)
    server = PolicyServer(
        policy,
        wire,
        fleet_config=FleetConfig(
            max_robots=args.max_robots,
            max_head_batch=args.max_head_batch,
            vlm_budget_ms=args.vlm_budget_ms,
            action_head_budget_ms=args.head_budget_ms,
        ),
        host=args.host,
        port=args.port,
        metadata=_pairs(args.metadata, "--metadata"),
    )
    try:
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        _tracer.event("api_server_interrupted")
    finally:
        close()


if __name__ == "__main__":
    main()
