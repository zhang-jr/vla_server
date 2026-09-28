"""The four execution modes the closed-loop benchmark compares.

    native     The model's own monolithic inference, blocking: when the action
               queue is empty the sim pauses, one chunk is computed, its first
               ``n_action_steps`` actions execute. The no-runtime baseline
               (what LeRobot's sim eval does).
    sync       Same blocking loop through the runtime's split components
               (VLMBackend.encode → ActionHeadRuntime.predict), no scheduler.
               Must match ``native`` in success rate; the latency difference
               is the split's overhead, and VLM / head times separate.
    async_sim  AsyncScheduler + ChunkBuffer on a simulated clock: an inference
               started at step ``s`` lands at ``s + ceil(latency * hz)``, the
               robot meanwhile executes the queued tail. Deterministic in the
               latency, which may be the measured one, scaled, or injected —
               the latter is the condition-staleness calibration of TODO 0.3.
    realtime   FleetScheduler with N robots, each env stepped by a wall-clock
               control loop at the env's control rate. What deployment feels:
               buffer underruns, missed ticks, fleet contention on one VLM.

Blocking modes pause the sim during inference, so their success rate ignores
latency; ``blocked_fraction`` reports how long a real robot would stand still.
Async modes use a client-side ChunkBuffer per robot: the buffer belongs to the
control loop, and the simulated clock has to decide *when* a chunk lands.

Every mode enumerates episodes exactly like TaskSuccessRateValidator (same
seed → same task order, same reset seeds), so results are comparable row by row.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import torch

from benchmarks.sim.envs import BenchEnv, EnvSpec, Observation
from benchmarks.sim.metrics import EpisodeRecord
from benchmarks.sim.policies import BenchPolicy
from vla_runtime.action_head.base import ActionChunk
from vla_runtime.errors import PipelineError
from vla_runtime.observability.tracer import get_tracer
from vla_runtime.scheduler.async_scheduler import AsyncScheduler, RequestHandle
from vla_runtime.scheduler.budget_manager import BudgetConfig
from vla_runtime.scheduler.chunk_buffer import ChunkBlendPolicy, ChunkBuffer
from vla_runtime.serving.fleet import FleetConfig, FleetScheduler
from vla_runtime.vlm.base import ActionRequest

_tracer = get_tracer("benchmarks.sim")

Mode = Literal["native", "sync", "async_sim", "realtime"]
MODES: tuple[Mode, ...] = ("native", "sync", "async_sim", "realtime")
T = TypeVar("T")


@dataclass
class RunConfig:
    """How to run one benchmark configuration.

    Attributes:
        mode: Execution mode (module docstring).
        episodes_per_task: Episodes per task.
        num_tasks: First N tasks of the env (None = all).
        seed: Episode enumeration seed (ADR-008 reproducibility).
        blend_policy: ChunkBuffer policy for the async modes (ADR-014).
        chunk_threshold: Async modes request the next chunk once the queued
            actions fall to this fraction of a chunk (LeRobot async's ``g``).
        max_trim_steps: ChunkBuffer latency-trim ceiling (None = runtime default).
        latency_ms: async_sim — apply this latency instead of the measured one.
        latency_scale: async_sim — multiply the measured latency (e.g. to
            emulate a slower edge device).
        vlm_budget_ms / head_budget_ms: Scheduler latency budgets.
        num_robots: realtime — robots sharing the fleet's backbone.
        max_steps: Override every task's step limit.
        warmup: Untimed inferences before the first episode.
    """

    mode: Mode
    episodes_per_task: int = 10
    num_tasks: int | None = None
    seed: int = 42
    blend_policy: ChunkBlendPolicy = "linear_overlap"
    chunk_threshold: float = 0.5
    max_trim_steps: int | None = None
    latency_ms: float | None = None
    latency_scale: float = 1.0
    vlm_budget_ms: float = 2000.0
    head_budget_ms: float = 1000.0
    num_robots: int = 1
    max_steps: int | None = None
    warmup: int = 3

    @property
    def label(self) -> str:
        parts: list[str] = [self.mode]
        if self.mode in ("async_sim", "realtime"):
            parts.append(self.blend_policy)
        if self.mode == "async_sim" and self.latency_ms is not None:
            parts.append(f"lat{self.latency_ms:g}ms")
        if self.mode == "async_sim" and self.latency_scale != 1.0:
            parts.append(f"x{self.latency_scale:g}")
        if self.mode == "realtime":
            parts.append(f"{self.num_robots}robot")
        return "/".join(parts)


# ── Env worker ────────────────────────────────────────────────────────────────

class EnvWorker:
    """One environment, built and stepped on one dedicated thread.

    Simulator and EGL rendering contexts are bound to their creating thread;
    keeping every call on that thread also keeps env.step off the event loop.
    """

    def __init__(self, pool: ThreadPoolExecutor, env: BenchEnv) -> None:
        self._pool = pool
        self.env = env

    @classmethod
    async def create(cls, factory: Callable[[], BenchEnv]) -> EnvWorker:
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sim-env")
        env = await asyncio.get_running_loop().run_in_executor(pool, factory)
        return cls(pool, env)

    async def call(self, fn: Callable[..., T], *args: Any) -> T:
        return await asyncio.get_running_loop().run_in_executor(self._pool, fn, *args)

    async def close(self) -> None:
        await self.call(self.env.close)
        self._pool.shutdown(wait=True)


def episode_plan(env: BenchEnv, config: RunConfig) -> list[tuple[str, int, int]]:
    """(task, episode_idx, reset_seed) — the same enumeration as the validator."""
    rng = random.Random(config.seed)
    tasks = env.task_names()
    if config.num_tasks is not None:
        tasks = tasks[: config.num_tasks]
    return [
        (task, ep, rng.randint(0, 2**31))
        for task in tasks
        for ep in range(config.episodes_per_task)
    ]


def _request(obs: Observation, frame_id: int) -> ActionRequest:
    return ActionRequest(
        request_id=uuid.uuid4().hex[:12],
        images=obs["images"],
        instruction=obs["instruction"],
        robot_state=obs["robot_state"],
        frame_id=frame_id,
    )


def _remaining(buffer: ChunkBuffer) -> int:
    current = buffer.current()
    return 0 if current is None else current.chunk_size


def _needs_chunk(buffer: ChunkBuffer, chunk_size: int, threshold: float) -> bool:
    return _remaining(buffer) <= threshold * chunk_size


def _new_buffer(spec: EnvSpec, config: RunConfig, robot_id: str) -> ChunkBuffer:
    kwargs: dict[str, Any] = {}
    if config.max_trim_steps is not None:
        kwargs["max_latency_trim_steps"] = config.max_trim_steps
    return ChunkBuffer(
        robot_id=robot_id,
        blend_policy=config.blend_policy,
        control_rate_hz=spec.control_hz,
        **kwargs,
    )


async def _drain(stream: Any) -> ActionChunk | None:
    last: ActionChunk | None = None
    async for chunk in stream:
        last = chunk
    return last


def _raise_if_failed(handle: RequestHandle) -> None:
    if handle.error is not None:
        raise handle.error


# ── Blocking modes: native / sync ─────────────────────────────────────────────

BlockingInfer = Callable[[Observation, EpisodeRecord], Awaitable[torch.Tensor]]


async def _blocking_episode(
    worker: EnvWorker,
    infer: BlockingInfer,
    n_action_steps: int,
    record: EpisodeRecord,
    max_steps: int,
) -> None:
    env = worker.env
    obs = await worker.call(env.reset, record.task, record.episode, record.seed)
    queue: deque[torch.Tensor] = deque()
    executed: list[torch.Tensor] = []
    boundaries: list[int] = []
    t_episode = time.perf_counter()

    for step in range(max_steps):
        if not queue:
            t0 = time.perf_counter()
            actions = await infer(obs, record)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            record.inference_ms.append(elapsed_ms)
            record.blocked_ms += elapsed_ms
            boundaries.append(step)
            queue.extend(actions[:n_action_steps].float().cpu())
        action = queue.popleft()
        obs, _, done, info = await worker.call(env.step, action)
        executed.append(action)
        record.steps = step + 1
        if done:
            record.success = bool(info.get("success", False))
            break

    record.wall_s = time.perf_counter() - t_episode
    record.finalize(executed, boundaries, env.spec.action_is_delta, env.spec.gripper_index)


def _native_infer(bench: BenchPolicy) -> BlockingInfer:
    if bench.native is None:
        raise ValueError(f"policy {bench.name!r} has no native reference path")
    native = bench.native

    async def infer(obs: Observation, record: EpisodeRecord) -> torch.Tensor:
        return await native(obs)

    return infer


def _sync_infer(bench: BenchPolicy) -> BlockingInfer:
    policy = bench.policy
    head = policy.head_for_robot()
    to_backbone = policy.capabilities.state_to_backbone

    async def infer(obs: Observation, record: EpisodeRecord) -> torch.Tensor:
        state = obs["robot_state"]
        t0 = time.perf_counter()
        embedding = await policy.vlm_backend.encode(
            obs["images"], obs["instruction"], state if to_backbone else None
        )
        t1 = time.perf_counter()
        chunk = await asyncio.to_thread(head.predict, embedding, state)
        t2 = time.perf_counter()
        record.vlm_ms.append((t1 - t0) * 1000.0)
        record.head_ms.append((t2 - t1) * 1000.0)
        return chunk.actions

    return infer


# ── async_sim ─────────────────────────────────────────────────────────────────

@dataclass
class _InFlight:
    chunk: ActionChunk
    ready_step: int
    latency_ms: float


async def _async_sim_episode(
    worker: EnvWorker,
    bench: BenchPolicy,
    config: RunConfig,
    record: EpisodeRecord,
    max_steps: int,
) -> None:
    env = worker.env
    spec = env.spec
    scheduler = AsyncScheduler.from_policy(
        bench.policy,
        robot_id=record.robot_id,
        budget_config=BudgetConfig(
            vlm_budget_ms=config.vlm_budget_ms,
            action_head_budget_ms=config.head_budget_ms,
            total_budget_ms=config.vlm_budget_ms + config.head_budget_ms,
        ),
        blend_policy=config.blend_policy,
        control_rate_hz=spec.control_hz,
    )
    chunk_size = bench.policy.head_prototype.chunk_size
    buffer = _new_buffer(spec, config, record.robot_id)
    frame = 0

    async def infer(obs: Observation) -> tuple[ActionChunk | None, float, str | None]:
        nonlocal frame
        frame += 1
        t0 = time.perf_counter()
        handle = await scheduler.submit(_request(obs, frame))
        chunk = await _drain(scheduler.stream(handle))
        _raise_if_failed(handle)
        return chunk, (time.perf_counter() - t0) * 1000.0, handle.degraded

    await scheduler.start()
    try:
        obs = await worker.call(env.reset, record.task, record.episode, record.seed)
        t_episode = time.perf_counter()

        # First chunk: nothing to execute yet, the robot waits for it.
        first, elapsed_ms, _ = await infer(obs)
        if first is None:
            raise PipelineError("first inference of the episode produced no chunk")
        record.inference_ms.append(elapsed_ms)
        record.effective_latency_ms.append(elapsed_ms)
        record.blocked_ms += elapsed_ms
        buffer.push(first)

        executed: list[torch.Tensor] = []
        boundaries = [0]
        in_flight: _InFlight | None = None
        last_action: torch.Tensor | None = None

        for step in range(max_steps):
            if in_flight is None and _needs_chunk(buffer, chunk_size, config.chunk_threshold):
                chunk, elapsed_ms, degraded = await infer(obs)
                record.inference_ms.append(elapsed_ms)
                if chunk is None:
                    record.degraded += 1
                    _tracer.event("sim_inference_degraded", reason=degraded, step=step)
                else:
                    latency = (
                        config.latency_ms
                        if config.latency_ms is not None
                        else elapsed_ms * config.latency_scale
                    )
                    record.effective_latency_ms.append(latency)
                    delay = math.ceil(latency / 1000.0 * spec.control_hz)
                    in_flight = _InFlight(chunk, step + delay, latency)

            if in_flight is not None and step >= in_flight.ready_step:
                buffer.update_latency_trim(in_flight.latency_ms)
                buffer.push(in_flight.chunk)
                boundaries.append(step)
                in_flight = None

            action = buffer.next_action()
            if action is None:
                record.underrun_steps += 1
                action = env.hold_action(last_action)
            action = action.float().cpu()
            obs, _, done, info = await worker.call(env.step, action)
            executed.append(action)
            last_action = action
            record.steps = step + 1
            if done:
                record.success = bool(info.get("success", False))
                break

        record.wall_s = time.perf_counter() - t_episode
        record.finalize(executed, boundaries, spec.action_is_delta, spec.gripper_index)
    finally:
        await scheduler.stop()


# ── realtime (fleet) ──────────────────────────────────────────────────────────

async def _realtime_episode(
    worker: EnvWorker,
    fleet: FleetScheduler,
    bench: BenchPolicy,
    config: RunConfig,
    record: EpisodeRecord,
    max_steps: int,
) -> None:
    env = worker.env
    spec = env.spec
    robot_id = record.robot_id
    chunk_size = bench.policy.head_prototype.chunk_size
    buffer = _new_buffer(spec, config, robot_id)
    loop = asyncio.get_running_loop()
    period = 1.0 / spec.control_hz
    frame = 0

    async def infer(obs: Observation) -> tuple[ActionChunk | None, float, str | None]:
        nonlocal frame
        frame += 1
        t0 = time.perf_counter()
        handle = await fleet.submit(robot_id, _request(obs, frame))
        chunk = await _drain(fleet.stream(handle))
        _raise_if_failed(handle)
        return chunk, (time.perf_counter() - t0) * 1000.0, handle.degraded

    # A fresh registration per episode: no condition survives from the last one.
    await fleet.register_robot(robot_id)
    in_flight: asyncio.Task | None = None
    try:
        obs = await worker.call(env.reset, record.task, record.episode, record.seed)
        t_episode = time.perf_counter()

        first, elapsed_ms, _ = await infer(obs)
        if first is None:
            raise PipelineError("first inference of the episode produced no chunk")
        record.inference_ms.append(elapsed_ms)
        record.blocked_ms += elapsed_ms
        buffer.push(first)

        executed: list[torch.Tensor] = []
        boundaries = [0]
        last_action: torch.Tensor | None = None
        next_tick = loop.time()

        for step in range(max_steps):
            if in_flight is not None and in_flight.done():
                chunk, elapsed_ms, degraded = in_flight.result()
                in_flight = None
                record.inference_ms.append(elapsed_ms)
                if chunk is None:
                    record.degraded += 1
                    _tracer.event("realtime_inference_degraded", reason=degraded, step=step)
                else:
                    buffer.update_latency_trim(elapsed_ms)
                    buffer.push(chunk)
                    boundaries.append(step)
            if in_flight is None and _needs_chunk(buffer, chunk_size, config.chunk_threshold):
                in_flight = asyncio.create_task(infer(obs))

            action = buffer.next_action()
            if action is None:
                record.underrun_steps += 1
                action = env.hold_action(last_action)
            action = action.float().cpu()
            t_step = time.perf_counter()
            obs, _, done, info = await worker.call(env.step, action)
            record.sim_step_ms.append((time.perf_counter() - t_step) * 1000.0)
            executed.append(action)
            last_action = action
            record.steps = step + 1
            if done:
                record.success = bool(info.get("success", False))
                break

            next_tick += period
            slack = next_tick - loop.time()
            if slack > 0:
                await asyncio.sleep(slack)
            else:
                record.overrun_steps += 1
                record.tick_lag_ms.append(-slack * 1000.0)
                next_tick = loop.time()  # no catch-up burst after a missed tick

        record.wall_s = time.perf_counter() - t_episode
        record.finalize(executed, boundaries, spec.action_is_delta, spec.gripper_index)
    finally:
        if in_flight is not None:
            await asyncio.gather(in_flight, return_exceptions=True)
        await fleet.deregister_robot(robot_id)


# ── Entry ─────────────────────────────────────────────────────────────────────

async def _warmup(bench: BenchPolicy, worker: EnvWorker, plan: list[tuple[str, int, int]], n: int) -> None:
    """Untimed inferences so JIT / cuDNN autotune / allocator growth stay out of the numbers."""
    if n <= 0 or not plan:
        return
    task, episode, seed = plan[0]
    obs = await worker.call(worker.env.reset, task, episode, seed)
    scratch = EpisodeRecord(task=task, episode=episode, seed=seed)
    sync = _sync_infer(bench)
    for _ in range(n):
        await sync(obs, scratch)
        if bench.native is not None:
            await bench.native(obs)


def _reset_gpu_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


async def run_benchmark(
    bench: BenchPolicy,
    env_factory: Callable[[], BenchEnv],
    config: RunConfig,
    worker: EnvWorker | None = None,
    on_episode: Callable[[EpisodeRecord], None] | None = None,
) -> tuple[list[EpisodeRecord], float]:
    """Run every planned episode in one mode.

    Args:
        bench: The policy.
        env_factory: Builds an env (called once per extra realtime robot).
        config: Mode and knobs.
        worker: Reuse an existing env worker for robot 0 (keeps the sim warm
            across modes); created and closed here when None.
        on_episode: Progress callback after each episode.

    Returns:
        (episode records, run wall-clock seconds).
    """
    own_worker = worker is None
    if worker is None:
        worker = await EnvWorker.create(env_factory)
    plan = episode_plan(worker.env, config)
    records: list[EpisodeRecord] = []

    def max_steps_for(env: BenchEnv, task: str) -> int:
        return config.max_steps or env.max_steps(task)

    extra_workers: list[EnvWorker] = []
    try:
        await _warmup(bench, worker, plan, config.warmup)
        _reset_gpu_peak()
        t_run = time.perf_counter()

        if config.mode in ("native", "sync", "async_sim"):
            infer: BlockingInfer | None = None
            if config.mode == "native":
                infer = _native_infer(bench)
            elif config.mode == "sync":
                infer = _sync_infer(bench)
            for task, episode, seed in plan:
                torch.manual_seed(seed)
                record = EpisodeRecord(task=task, episode=episode, seed=seed)
                limit = await worker.call(max_steps_for, worker.env, task)
                if infer is not None:
                    await _blocking_episode(worker, infer, bench.n_action_steps, record, limit)
                else:
                    await _async_sim_episode(worker, bench, config, record, limit)
                records.append(record)
                if on_episode:
                    on_episode(record)
        else:
            extra_workers = [
                await EnvWorker.create(env_factory) for _ in range(config.num_robots - 1)
            ]
            workers = [worker, *extra_workers]
            fleet = FleetScheduler.from_policy(
                bench.policy,
                FleetConfig(
                    max_robots=config.num_robots,
                    vlm_budget_ms=config.vlm_budget_ms,
                    action_head_budget_ms=config.head_budget_ms,
                ),
            )
            await fleet.start()

            async def robot(i: int) -> None:
                for task, episode, seed in plan[i :: config.num_robots]:
                    record = EpisodeRecord(
                        task=task, episode=episode, seed=seed, robot_id=f"robot_{i}"
                    )
                    limit = await workers[i].call(max_steps_for, workers[i].env, task)
                    await _realtime_episode(workers[i], fleet, bench, config, record, limit)
                    records.append(record)
                    if on_episode:
                        on_episode(record)

            try:
                await asyncio.gather(*(robot(i) for i in range(config.num_robots)))
            finally:
                await fleet.stop()

        return records, time.perf_counter() - t_run
    finally:
        for extra in extra_workers:
            await extra.close()
        if own_worker:
            await worker.close()
