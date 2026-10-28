"""Wire format between the robot client and the policy server.

One message = ``>II`` (header length, payload length) + UTF-8 JSON header +
payload. The header's ``blobs`` list describes the numpy arrays packed back to
back in the payload (name, dtype, shape, offset). Stdlib + numpy only: the
client side runs on the robot host's system Python 3.8, which has neither pip
nor msgpack, and raw blobs avoid base64 for the camera frames.
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any

import numpy as np

_PREFIX = struct.Struct(">II")
MAX_HEADER_BYTES = 1 << 20
MAX_PAYLOAD_BYTES = 1 << 28


class ProtocolError(RuntimeError):
    """Malformed or truncated message, or the peer closed the connection."""


def encode(header: dict[str, Any], arrays: dict[str, np.ndarray] | None = None) -> bytes:
    """Serialize a header and named arrays into one message."""
    blobs = []
    chunks = []
    offset = 0
    for name, array in (arrays or {}).items():
        data = np.ascontiguousarray(array)
        raw = data.tobytes()
        blobs.append({
            "name": name, "dtype": data.dtype.str, "shape": list(data.shape),
            "offset": offset, "nbytes": len(raw),
        })
        chunks.append(raw)
        offset += len(raw)
    head = json.dumps({**header, "blobs": blobs}, separators=(",", ":")).encode("utf-8")
    return _PREFIX.pack(len(head), offset) + head + b"".join(chunks)


def decode(head: bytes, payload: bytes) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Inverse of ``encode`` for an already split header / payload."""
    header = json.loads(head.decode("utf-8"))
    arrays: dict[str, np.ndarray] = {}
    for blob in header.pop("blobs", []):
        start = int(blob["offset"])
        end = start + int(blob["nbytes"])
        if end > len(payload):
            raise ProtocolError(f"blob {blob['name']!r} runs past the payload")
        arrays[blob["name"]] = np.frombuffer(
            payload[start:end], dtype=np.dtype(blob["dtype"])
        ).reshape(blob["shape"])
    return header, arrays


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    buf = bytearray(size)
    view = memoryview(buf)
    got = 0
    while got < size:
        n = sock.recv_into(view[got:], size - got)
        if n == 0:
            raise ProtocolError("connection closed by peer")
        got += n
    return bytes(buf)


def send(sock: socket.socket, header: dict[str, Any], arrays: dict[str, np.ndarray] | None = None) -> None:
    sock.sendall(encode(header, arrays))


def recv(sock: socket.socket) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    head_len, payload_len = _PREFIX.unpack(_recv_exact(sock, _PREFIX.size))
    if head_len > MAX_HEADER_BYTES or payload_len > MAX_PAYLOAD_BYTES:
        raise ProtocolError(f"message too large: header {head_len} B, payload {payload_len} B")
    head = _recv_exact(sock, head_len)
    payload = _recv_exact(sock, payload_len) if payload_len else b""
    return decode(head, payload)


def tune(sock: socket.socket) -> socket.socket:
    """Disable Nagle: requests and replies are single writes that must leave now."""
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock
