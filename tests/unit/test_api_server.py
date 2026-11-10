"""Unit tests for the WebSocket policy server (ADR-016) and its wire codec.

Coverage:
- msgpack-numpy codec: openpi's exact encoding, round trips, rejected dtypes
- Request conversion: HWC/CHW uint8/float images, state, prompt, RTC fields
- End to end over a real socket with the mock policy: handshake, inference,
  one robot per connection, RTC context,
  error reporting, admission (max_robots), health check
- openpi's own client against the server, when openpi-client is installed
"""

from __future__ import annotations

import asyncio
import http.client

import msgpack
import numpy as np
import pytest
import torch
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from vla_runtime.models.mock import mock_policy
from vla_runtime.serving.api_server import (
    OPENPI_LIBERO_IMAGE_KEYS,
    PolicyServer,
    WireFormat,
    WireFormatError,
    build_policy,
    image_to_tensor,
    state_to_robot_state,
)
from vla_runtime.serving.fleet import FleetConfig
from vla_runtime.serving.wire import packb, unpackb

ACTION_DIM = 7
CHUNK = 8
STATE_LEN = 8


def _obs(size: int = 32, **extra) -> dict:
    rng = np.random.default_rng(0)
    return {
        "observation/image": rng.integers(0, 256, (size, size, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 256, (size, size, 3), dtype=np.uint8),
        "observation/state": np.zeros(STATE_LEN, dtype=np.float32),
        "prompt": "pick up the bowl",
        **extra,
    }


def _policy(rtc: bool = False, latency_ms: float = 0.0):
    return mock_policy(
        num_views=2,
        action_dim=ACTION_DIM,
        state_dim=state_to_robot_state(np.zeros(STATE_LEN), "s").flat_dim,
        chunk_size=CHUNK,
        embedding_dim=32,
        n_steps=2,
        latency_ms=latency_ms,
        rtc=rtc,
    )


@pytest.fixture
async def server():
    servers: list[PolicyServer] = []

    async def start(**kwargs) -> PolicyServer:
        policy = kwargs.pop("policy", None) or _policy()
        srv = PolicyServer(policy, host="127.0.0.1", port=0, **kwargs)
        await srv.start()
        servers.append(srv)
        return srv

    yield start
    for srv in servers:
        await srv.stop()


def _url(srv: PolicyServer, query: str = "") -> str:
    return f"ws://127.0.0.1:{srv.port}/{query}"


# Clients below pass proxy=None: websockets honors HTTP(S)_PROXY by default, and
# a local test must not depend on (or route through) the machine's proxy.


# ── Codec ─────────────────────────────────────────────────────────────────────

class TestWire:
    def test_arrays_use_openpis_encoding(self):
        raw = msgpack.unpackb(packb({"a": np.arange(6, dtype=np.int16).reshape(2, 3)}))
        encoded = raw["a"]
        assert encoded[b"__ndarray__"] is True
        assert encoded[b"dtype"] == "<i2"
        assert list(encoded[b"shape"]) == [2, 3]
        assert encoded[b"data"] == np.arange(6, dtype=np.int16).tobytes()

    @pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64, np.int64, np.bool_])
    def test_array_round_trip(self, dtype):
        array = (np.arange(24) % 2).astype(dtype).reshape(2, 3, 4)
        back = unpackb(packb({"x": array, "nested": [array[0]]}))
        np.testing.assert_array_equal(back["x"], array)
        np.testing.assert_array_equal(back["nested"][0], array[0])
        assert back["x"].dtype == array.dtype

    def test_numpy_scalars_and_tensors(self):
        back = unpackb(packb({"s": np.float32(1.5), "t": torch.ones(2, 2)}))
        assert back["s"] == np.float32(1.5) and isinstance(back["s"], np.float32)
        np.testing.assert_array_equal(back["t"], np.ones((2, 2), dtype=np.float32))

    def test_non_contiguous_array_packs_its_values(self):
        array = np.arange(12, dtype=np.float32).reshape(3, 4).T
        np.testing.assert_array_equal(unpackb(packb(array)), array)

    def test_object_arrays_are_rejected(self):
        with pytest.raises(ValueError, match="unsupported dtype"):
            packb(np.array([object()], dtype=object))


# ── Request conversion ────────────────────────────────────────────────────────

class TestWireFormat:
    def test_hwc_uint8_becomes_chw_unit_float(self):
        image = np.full((4, 5, 3), 255, dtype=np.uint8)
        tensor = image_to_tensor(image, "img")
        assert tensor.shape == (3, 4, 5) and tensor.dtype == torch.float32
        assert torch.allclose(tensor, torch.ones_like(tensor))

    def test_chw_float_is_kept(self):
        image = np.full((3, 4, 5), 0.5, dtype=np.float32)
        assert torch.allclose(image_to_tensor(image, "img"), torch.full((3, 4, 5), 0.5))

    def test_rejects_non_images(self):
        with pytest.raises(WireFormatError, match="3-D"):
            image_to_tensor(np.zeros((4, 4)), "img")
        with pytest.raises(WireFormatError, match="channel"):
            image_to_tensor(np.zeros((4, 4, 4)), "img")

    def test_request_fields(self):
        request = WireFormat().to_request(_obs(), frame_id=3)
        assert list(request.images) == list(OPENPI_LIBERO_IMAGE_KEYS)
        assert request.instruction == "pick up the bowl"
        assert request.frame_id == 3
        assert torch.equal(request.robot_state.extra["policy_state"], torch.zeros(STATE_LEN))
        assert torch.equal(request.robot_state.joint_positions, torch.zeros(STATE_LEN))

    def test_missing_key_names_it(self):
        obs = _obs()
        del obs["observation/wrist_image"]
        with pytest.raises(WireFormatError, match="observation/wrist_image"):
            WireFormat().to_request(obs, frame_id=0)

    def test_prompt_must_be_text(self):
        with pytest.raises(WireFormatError, match="string"):
            WireFormat().to_request(_obs(prompt=3), frame_id=0)

    def test_rtc_fields(self):
        assert WireFormat.head_context(_obs()) is None
        context = WireFormat.head_context(
            _obs(prev_actions=np.ones((4, ACTION_DIM), dtype=np.float64), inference_delay_steps=2)
        )
        assert context.prev_actions.shape == (4, ACTION_DIM)
        assert context.prev_actions.dtype == torch.float32
        assert context.inference_delay_steps == 2
        with pytest.raises(WireFormatError, match="prev_actions"):
            WireFormat.head_context(_obs(prev_actions=np.ones(4)))


# ── End to end ────────────────────────────────────────────────────────────────

class TestPolicyServer:
    async def test_handshake_then_actions(self, server):
        srv = await server()
        async with connect(_url(srv, "?robot_id=arm_0"), max_size=None, proxy=None) as ws:
            meta = unpackb(await ws.recv())
            assert meta["robot_id"] == "arm_0"
            assert (meta["chunk_size"], meta["action_dim"]) == (CHUNK, ACTION_DIM)
            assert meta["policy"] == "mock"
            assert "arm_0" in srv.fleet.stats()["registered_robots"]

            for _ in range(2):
                await ws.send(packb(_obs()))
                response = unpackb(await ws.recv())
                assert response["actions"].shape == (CHUNK, ACTION_DIM)
                assert response["actions"].dtype == np.float32
                assert np.isfinite(response["actions"]).all()
                assert response["runtime"]["robot_id"] == "arm_0"
                assert "fallback" not in response["runtime"], response["runtime"].get("reason")
                assert response["runtime"]["head_batch_size"] == 1
                assert response["server_timing"]["infer_ms"] > 0
            assert "prev_total_ms" in response["server_timing"]

        for _ in range(50):  # deregistration runs as the handler unwinds
            if "arm_0" not in srv.fleet.stats()["registered_robots"]:
                break
            await asyncio.sleep(0.01)
        assert "arm_0" not in srv.fleet.stats()["registered_robots"]

    async def test_each_connection_is_its_own_robot(self, server):
        srv = await server(policy=_policy(latency_ms=10.0))
        sockets = [await connect(_url(srv), max_size=None, proxy=None) for _ in range(3)]
        try:
            ids = [unpackb(await ws.recv())["robot_id"] for ws in sockets]
            assert len(set(ids)) == 3
            assert sorted(srv.fleet.stats()["registered_robots"]) == sorted(ids)

            async def infer(ws) -> dict:
                await ws.send(packb(_obs()))
                return unpackb(await ws.recv())["runtime"]

            runtimes = await asyncio.gather(*(infer(ws) for ws in sockets))
            assert [r["robot_id"] for r in runtimes] == ids
            assert not any("fallback" in r for r in runtimes)
        finally:
            for ws in sockets:
                await ws.close()

    async def test_rtc_context_reaches_the_head(self, server):
        srv = await server(policy=_policy(rtc=True))
        async with connect(_url(srv), max_size=None, proxy=None) as ws:
            assert unpackb(await ws.recv())["supports_rtc"] is True
            prev = np.zeros((CHUNK, ACTION_DIM), dtype=np.float32)
            await ws.send(packb(_obs(prev_actions=prev, inference_delay_steps=2)))
            response = unpackb(await ws.recv())
            assert "fallback" not in response["runtime"], response["runtime"].get("reason")
            assert response["runtime"]["rtc"] is True
            assert response["runtime"]["rtc_delay_steps"] == 2

    async def test_rtc_fields_are_ignored_without_rtc(self, server):
        srv = await server()
        async with connect(_url(srv), max_size=None, proxy=None) as ws:
            await ws.recv()
            await ws.send(packb(_obs(prev_actions=np.zeros((CHUNK, ACTION_DIM)))))
            assert unpackb(await ws.recv())["runtime"]["rtc"] is False

    async def test_bad_request_gets_a_traceback_and_1011(self, server):
        srv = await server()
        async with connect(_url(srv), max_size=None, proxy=None) as ws:
            await ws.recv()
            bad = _obs()
            del bad["prompt"]
            await ws.send(packb(bad))
            reply = await ws.recv()
            assert isinstance(reply, str) and "WireFormatError" in reply
            with pytest.raises(ConnectionClosed) as closed:
                await ws.recv()
            assert closed.value.rcvd.code == 1011

    async def test_full_fleet_closes_with_1013(self, server):
        srv = await server(fleet_config=FleetConfig(max_robots=1))
        async with connect(_url(srv), max_size=None, proxy=None) as first:
            await first.recv()
            async with connect(_url(srv), max_size=None, proxy=None) as second:
                with pytest.raises(ConnectionClosed) as closed:
                    await second.recv()
                assert closed.value.rcvd.code == 1013

    async def test_duplicate_robot_id_closes_with_1008(self, server):
        srv = await server()
        async with connect(_url(srv, "?robot_id=a"), max_size=None, proxy=None) as first:
            await first.recv()
            async with connect(_url(srv, "?robot_id=a"), max_size=None, proxy=None) as second:
                with pytest.raises(ConnectionClosed) as closed:
                    await second.recv()
                assert closed.value.rcvd.code == 1008

    async def test_healthz(self, server):
        srv = await server()

        def get() -> int:
            conn = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=5)
            conn.request("GET", "/healthz")
            return conn.getresponse().status

        assert await asyncio.to_thread(get) == 200


class TestOpenpiClient:
    """openpi's own client, unmodified (skipped where openpi-client is not installed)."""

    async def test_openpi_client_drives_the_server(self, server, monkeypatch):
        policy_client = pytest.importorskip("openpi_client.websocket_client_policy")
        msgpack_numpy = pytest.importorskip("openpi_client.msgpack_numpy")
        for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
            monkeypatch.delenv(var, raising=False)
        sample = {"a": np.arange(6, dtype=np.float32).reshape(2, 3), "s": np.int64(3)}
        np.testing.assert_array_equal(msgpack_numpy.unpackb(packb(sample))["a"], sample["a"])
        np.testing.assert_array_equal(unpackb(msgpack_numpy.packb(sample))["a"], sample["a"])

        srv = await server()

        def run() -> tuple[dict, dict, str]:
            client = policy_client.WebsocketClientPolicy(host="127.0.0.1", port=srv.port)
            result = client.infer(_obs())
            try:
                client.infer({"prompt": "no images"})
            except RuntimeError as exc:
                return client.get_server_metadata(), result, str(exc)
            raise AssertionError("a bad request did not raise in the openpi client")

        meta, result, error = await asyncio.to_thread(run)
        assert (meta["chunk_size"], meta["action_dim"]) == (CHUNK, ACTION_DIM)
        assert result["actions"].shape == (CHUNK, ACTION_DIM)
        assert "fallback" not in result["runtime"]
        assert "WireFormatError" in error


class TestBuildPolicy:
    async def test_mock_matches_the_wire_format(self):
        wire = WireFormat()
        policy, close = build_policy("mock", {"state_len": "8", "chunk_size": "4"}, wire)
        request = wire.to_request(_obs(), frame_id=0)
        embedding = await policy.vlm_backend.encode(request.images, request.instruction)
        chunk = policy.head_prototype.predict(embedding, request.robot_state)
        assert chunk.actions.shape == (4, ACTION_DIM)
        close()

    def test_unknown_options_and_policies(self):
        with pytest.raises(ValueError, match="unknown --policy-opt"):
            build_policy("mock", {"nope": "1"}, WireFormat())
        with pytest.raises(ValueError, match="unknown policy"):
            build_policy("gr00t", {}, WireFormat())
