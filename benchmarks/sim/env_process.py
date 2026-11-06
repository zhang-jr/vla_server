"""Out-of-process environments for the realtime fleet runner.

A deployed fleet's robots run on their own machines; only observations and
actions reach the server. The realtime runner used to step every robot's
simulator on a thread of the server process instead, and MuJoCo stepping plus
EGL rendering hold the GIL long enough to starve inference: with four LIBERO
envs at 20 Hz, one π0.5 inference went from 162 to 648 ms (BENCHMARKS.md, V3
microbenchmark). The benchmark was measuring its own harness.

:class:`ProcessEnv` runs one env in a spawned child process and forwards the
``BenchEnv`` calls over a pipe. The caller still drives it from an
:class:`~benchmarks.sim.runners.EnvWorker` thread, which blocks on the pipe
with the GIL released.

Images cross the pipe as uint8. Envs produce ``k / 255`` floats, so the round
trip is exact and the transfer is a quarter of the float size.
"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Any

import torch

from benchmarks.sim.envs import BenchEnv, Observation

# Child shutdown grace before it is terminated.
_JOIN_TIMEOUT_S = 10.0


class EnvProcessError(RuntimeError):
    """The env process failed a call, could not start, or died."""


def _pack(obs: Observation) -> Observation:
    images = {
        name: image.mul(255.0).round_().to(torch.uint8) for name, image in obs["images"].items()
    }
    return {**obs, "images": images}


def _unpack(obs: Observation) -> Observation:
    images = {name: image.float().div_(255.0) for name, image in obs["images"].items()}
    return {**obs, "images": images}


def _serve(conn: Connection, factory: Callable[[], BenchEnv]) -> None:
    """Child process: build the env, then answer (method, args) until close."""
    try:
        env = factory()
    except BaseException:  # noqa: BLE001 - reported to the parent, which raises
        conn.send(("err", traceback.format_exc()))
        return
    conn.send(("ok", env.spec))
    while True:
        try:
            method, args = conn.recv()
        except EOFError:  # parent went away
            env.close()
            return
        try:
            result = getattr(env, method)(*args)
            if method == "reset":
                result = _pack(result)
            elif method == "step":
                obs, reward, done, info = result
                result = (_pack(obs), reward, done, info)
            conn.send(("ok", result))
        except Exception:  # noqa: BLE001 - reported to the parent, which raises
            conn.send(("err", traceback.format_exc()))
        if method == "close":
            return


class ProcessEnv(BenchEnv):
    """A ``BenchEnv`` living in its own process.

    Args:
        factory: Picklable zero-arg env constructor (``env_factory`` returns
            one); called in the child.

    Raises:
        EnvProcessError: If the child fails to build the env.
    """

    def __init__(self, factory: Callable[[], BenchEnv]) -> None:
        # spawn, not fork: the parent holds CUDA and thread state a fork would copy.
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_serve, args=(child, factory), daemon=True)
        self._proc.start()
        child.close()
        self.spec = self._receive("start")

    def _receive(self, method: str) -> Any:
        try:
            status, payload = self._conn.recv()
        except EOFError as exc:
            raise EnvProcessError(
                f"env process (pid {self._proc.pid}) died during {method}; "
                f"exit code {self._proc.exitcode}"
            ) from exc
        if status == "err":
            raise EnvProcessError(f"env process failed in {method}:\n{payload}")
        return payload

    def _call(self, method: str, *args: Any) -> Any:
        self._conn.send((method, args))
        return self._receive(method)

    def task_names(self) -> list[str]:
        return self._call("task_names")

    def max_steps(self, task_name: str) -> int:
        return self._call("max_steps", task_name)

    def reset(self, task_name: str, episode_idx: int, seed: int) -> Observation:
        return _unpack(self._call("reset", task_name, episode_idx, seed))

    def step(self, action: torch.Tensor) -> tuple[Observation, float, bool, dict]:
        obs, reward, done, info = self._call("step", action)
        return _unpack(obs), reward, done, info

    def close(self) -> None:
        if self._proc.is_alive():
            try:
                self._call("close")
            finally:
                self._proc.join(_JOIN_TIMEOUT_S)
                if self._proc.is_alive():
                    self._proc.terminate()
                    self._proc.join()
        self._conn.close()
