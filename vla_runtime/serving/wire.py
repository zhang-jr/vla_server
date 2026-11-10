"""msgpack with numpy arrays, byte-compatible with openpi's ``msgpack_numpy``.

openpi's policy server and client (``openpi_client.msgpack_numpy``) pack every
``np.ndarray`` as a map ``{b"__ndarray__": True, b"data": raw bytes,
b"dtype": dtype.str, b"shape": shape}`` and every numpy scalar as
``{b"__npgeneric__": True, b"data": value, b"dtype": dtype.str}``. Using the
same encoding lets an unmodified openpi client talk to the policy server
(ADR-016). torch tensors are packed as their numpy equivalent.
"""

from __future__ import annotations

from typing import Any

import msgpack
import numpy as np
import torch

_UNSUPPORTED_KINDS = ("V", "O", "c")  # void, object, complex


def _pack(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in _UNSUPPORTED_KINDS:
        raise ValueError(f"unsupported dtype for msgpack: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack(obj: dict) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=tuple(obj[b"shape"])
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def packb(obj: Any) -> bytes:
    """Serialize ``obj`` (dicts, lists, scalars, str, numpy arrays, tensors).

    Raises:
        ValueError: For numpy arrays of void, object or complex dtype.
        TypeError: For objects msgpack cannot represent.
    """
    return msgpack.packb(obj, default=_pack)


def unpackb(data: bytes) -> Any:
    """Inverse of :func:`packb`; arrays come back read-only, backed by ``data``."""
    return msgpack.unpackb(data, object_hook=_unpack)
