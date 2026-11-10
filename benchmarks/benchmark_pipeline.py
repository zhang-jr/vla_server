"""End-to-end pipeline benchmarks — pytest-benchmark suite.

Measures the full inference latency of:
  1. AsyncScheduler (single robot, mock VLM + action head)
  2. FleetScheduler (multi-robot, shared mock VLM)
  3. Per-head latency (FlowMatching / Diffusion / DiscreteToken with mock VLM)

These benchmarks use MockVLMBackend so they run on CPU without a real model.
They validate the scheduler and buffer overhead, not kernel speed.
Kernel speed is measured separately in benchmark_kernels.py.

Phase 2 targets (from TODO.md):
    - End-to-end P50 ≤ 500ms (full VLM, not mock)
    - ≥ 3 FPS throughput per robot

Run::

    pytest benchmarks/benchmark_pipeline.py \
        --benchmark-only \
        --benchmark-json=benchmarks/results/pipeline.json \
        -v
"""

from __future__ import annotations

import asyncio
import time

import pytest
import torch

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_request(robot_id: str = "robot_0") -> "ActionRequest":
    from vla_runtime.action_head.base import RobotState
    from vla_runtime.vlm.base import ActionRequest
    import uuid
    return ActionRequest(
        images={"base": torch.zeros(3, 224, 224)},
        instruction="pick up the apple",
        robot_state=RobotState(
            joint_positions=torch.zeros(7),
            joint_velocities=torch.zeros(7),
            gripper_state=torch.zeros(1),
        ),
        request_id=str(uuid.uuid4()),
    )


def _capabilities() -> "PolicyCapabilities":
    """pi0-like policy; the mock backbone emits a [1, 4096] prefix-token condition."""
    from vla_runtime.capabilities import PolicyCapabilities
    return PolicyCapabilities(
        state_placement="suffix",
        condition_kind="prefix_tokens",
        head_structure="iterative",
        max_condition_staleness_ms=500.0,
    )


class _FastMockVLMBackend:
    """Mock VLM that returns immediately (measures scheduler overhead, not VLM)."""

    latency_budget_ms: float = 200.0
    kv_cache_enabled: bool = False

    async def encode(self, images, text, state=None):
        from vla_runtime.vlm.base import Embedding
        return Embedding(data=torch.zeros(1, 4096), model_id="mock")

    async def generate(self, images, text, state=None):
        raise NotImplementedError

    async def warmup(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass


class _LatencyMockVLMBackend(_FastMockVLMBackend):
    """Mock VLM with configurable latency (simulates real model timing)."""

    def __init__(self, latency_ms: float = 100.0) -> None:
        self._latency_s = latency_ms / 1000.0
        self.latency_budget_ms = latency_ms * 1.5

    async def encode(self, images, text, state=None):
        await asyncio.sleep(self._latency_s)
        return await super().encode(images, text, state)


class _MockFlowMatchingHead:
    """Synchronous mock flow matching action head."""

    head_type = "flow_matching"
    frequency_hz: float = 10.0
    chunk_size: int = 16
    action_dim: int = 7

    def predict(self, embedding, robot_state):
        from vla_runtime.action_head.base import ActionChunk
        return ActionChunk(
            actions=torch.zeros(self.chunk_size, self.action_dim, dtype=torch.float16),
            head_type=self.head_type,
        )


# ── AsyncScheduler benchmarks ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def _async_scheduler_round_trip(vlm_latency_ms: float) -> float:
    """Return wall-clock time for one request through AsyncScheduler."""
    from vla_runtime.scheduler.async_scheduler import AsyncScheduler
    from vla_runtime.scheduler.budget_manager import BudgetConfig

    backend = _LatencyMockVLMBackend(vlm_latency_ms)
    head = _MockFlowMatchingHead()
    scheduler = AsyncScheduler(
        vlm_backend=backend,
        action_head=head,
        capabilities=_capabilities(),
        budget_config=BudgetConfig(
            vlm_budget_ms=vlm_latency_ms * 2,
            action_head_budget_ms=50.0,
        ),
    )
    await scheduler.start()

    t0 = time.perf_counter()
    handle = await scheduler.submit(_make_request())
    async for _ in scheduler.stream(handle):
        pass
    elapsed = (time.perf_counter() - t0) * 1000.0

    await scheduler.stop()
    return elapsed


def test_benchmark_scheduler_0ms_vlm(benchmark):
    """Scheduler overhead with instant VLM (measures pure scheduler cost)."""

    def run():
        return asyncio.run(_async_scheduler_round_trip(0.1))

    elapsed = benchmark(run)
    assert elapsed < 500.0, f"Scheduler overhead unexpectedly high: {elapsed:.1f}ms"


def test_benchmark_scheduler_100ms_vlm(benchmark):
    """Full round-trip with 100ms VLM latency — target total ≤ 200ms."""

    def run():
        return asyncio.run(_async_scheduler_round_trip(100.0))

    elapsed = benchmark(run)
    # Budget: 100ms VLM + action head overhead, should be well under 500ms
    assert elapsed < 500.0


# ── Fleet scheduler benchmarks ────────────────────────────────────────────────

@pytest.mark.asyncio
async def _fleet_multi_robot(num_robots: int, requests_per_robot: int) -> float:
    """Measure fleet throughput with N robots submitting K requests each."""
    from vla_runtime.serving.fleet import FleetConfig, FleetScheduler

    vlm = _FastMockVLMBackend()
    fleet = FleetScheduler(
        vlm_backend=vlm,
        capabilities=_capabilities(),
        config=FleetConfig(max_robots=num_robots, vlm_budget_ms=50.0),
    )
    await fleet.start()

    # Register robots
    for i in range(num_robots):
        await fleet.register_robot(f"robot_{i}", _MockFlowMatchingHead(), priority=1)

    t0 = time.perf_counter()

    # Submit all requests
    tasks = []
    for i in range(num_robots):
        for _ in range(requests_per_robot):
            handle = await fleet.submit(f"robot_{i}", _make_request(f"robot_{i}"))
            tasks.append(handle)

    # Drain all streams
    async def drain(h):
        async for _ in fleet.stream(h):
            pass

    await asyncio.gather(*[drain(h) for h in tasks])

    elapsed = (time.perf_counter() - t0) * 1000.0
    await fleet.stop()
    return elapsed


def test_benchmark_fleet_1_robot(benchmark):
    """Fleet with 1 robot — baseline for multi-robot overhead."""

    def run():
        return asyncio.run(_fleet_multi_robot(num_robots=1, requests_per_robot=5))

    elapsed = benchmark(run)
    assert elapsed < 2000.0


def test_benchmark_fleet_4_robots(benchmark):
    """Fleet with 4 robots — shared VLM should amortise well."""

    def run():
        return asyncio.run(_fleet_multi_robot(num_robots=4, requests_per_robot=3))

    elapsed = benchmark(run)
    assert elapsed < 5000.0


def test_benchmark_fleet_8_robots(benchmark):
    """Fleet with 8 robots — max pool size."""

    def run():
        return asyncio.run(_fleet_multi_robot(num_robots=8, requests_per_robot=2))

    elapsed = benchmark(run)
    assert elapsed < 10000.0


# ── KV cache benchmarks ───────────────────────────────────────────────────────

def test_benchmark_kv_alloc_free(benchmark):
    """KV cache allocate+free cycle — must be O(1) amortised."""
    from vla_runtime.vlm.kv_cache import KVCacheConfig, KVCacheManager

    # Use CPU to avoid GPU memory issues on CI
    cfg = KVCacheConfig(
        num_layers=4,
        num_heads=4,
        head_dim=32,
        max_context_tokens=64,
        pool_size=8,
        device="cpu",
        dtype=torch.float32,
    )
    manager = KVCacheManager(cfg)

    def run():
        slot = manager.allocate("req-bench")
        manager.free("req-bench")
        return slot

    result = benchmark(run)
    assert result is not None
    manager.shutdown()


def test_benchmark_kv_write_tokens(benchmark):
    """KV cache write_kv for 10 new tokens per call."""
    from vla_runtime.vlm.kv_cache import KVCacheConfig, KVCacheManager

    cfg = KVCacheConfig(
        num_layers=4,
        num_heads=4,
        head_dim=32,
        max_context_tokens=128,
        pool_size=4,
        device="cpu",
        dtype=torch.float32,
    )
    manager = KVCacheManager(cfg)
    slot = manager.allocate("req-write")

    new_k = torch.zeros(10, 4, 32)
    new_v = torch.zeros(10, 4, 32)

    def run():
        slot.seq_len = 0  # reset without full free/alloc cycle
        for layer in range(4):
            slot.write_kv(layer, new_k, new_v)

    benchmark(run)
    manager.free("req-write")
    manager.shutdown()


# ── Throughput summary ────────────────────────────────────────────────────────

def test_throughput_single_robot():
    """Sanity check: single robot should sustain ≥ 3 FPS with fast mock VLM.

    Not a benchmark fixture — just asserts the FPS target is achievable
    before full kernel + model integration.
    """
    import asyncio

    async def measure():
        from vla_runtime.scheduler.async_scheduler import AsyncScheduler
        from vla_runtime.scheduler.budget_manager import BudgetConfig

        backend = _FastMockVLMBackend()
        head = _MockFlowMatchingHead()
        scheduler = AsyncScheduler(
            vlm_backend=backend,
            action_head=head,
            capabilities=_capabilities(),
            budget_config=BudgetConfig(vlm_budget_ms=200.0, action_head_budget_ms=50.0),
        )
        await scheduler.start()

        N = 20
        t0 = time.perf_counter()
        for _ in range(N):
            handle = await scheduler.submit(_make_request())
            async for _ in scheduler.stream(handle):
                pass
        elapsed_s = time.perf_counter() - t0

        await scheduler.stop()
        return N / elapsed_s

    fps = asyncio.run(measure())
    # With mock VLM the scheduler overhead should allow well above 3 FPS
    assert fps >= 3.0, f"Throughput {fps:.1f} FPS below target 3 FPS"
