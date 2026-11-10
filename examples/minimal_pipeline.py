"""Minimal end-to-end example: the runtime pipeline on a mock policy (no GPU).

Demonstrates:
1. Declaring a policy's capabilities (ADR-018) and bundling it as a Policy
2. AsyncScheduler submission and streaming
3. The structured trace each inference emits

The backbone is a mock that returns a pooled condition after 50 ms, and the
head is the reference FlowMatchingHead (an MLP velocity net). Real models
(π0.5, SmolVLA) plug in through the prefix-KV adapters in ``benchmarks/sim``;
see ``python -m benchmarks.sim.run --help``.

Usage:
    python examples/minimal_pipeline.py
"""

from __future__ import annotations

import asyncio
import logging
import uuid

import torch

from vla_runtime import configure_logging, get_tracer
from vla_runtime.action_head.base import RobotState
from vla_runtime.action_head.flow_matching import FlowMatchingHead
from vla_runtime.capabilities import PolicyCapabilities
from vla_runtime.policy import Policy
from vla_runtime.scheduler.async_scheduler import AsyncScheduler
from vla_runtime.vlm.base import ActionRequest, Embedding

configure_logging(level=logging.INFO)
_tracer = get_tracer("example.minimal_pipeline")

DOF = 7
EMBEDDING_DIM = 512


class MockBackbone:
    """Stand-in backbone: a pooled condition after a simulated 50 ms encode."""

    model_id = "mock:backbone"
    latency_budget_ms = 200.0
    kv_cache_enabled = False

    async def encode(self, images, text: str, state=None) -> Embedding:
        await asyncio.sleep(0.05)
        return Embedding(data=torch.randn(EMBEDDING_DIM, dtype=torch.float16), model_id=self.model_id)

    async def generate(self, images, text, state=None):
        raise NotImplementedError("continuous heads use encode()")

    async def warmup(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass


async def main() -> None:
    # State enters the backbone prefix (SmolVLA / π0.5 style), so the policy
    # runs coupled: every head call gets a fresh backbone encode.
    capabilities = PolicyCapabilities(
        state_placement="prefix",
        condition_kind="pooled",
        head_structure="iterative",
    )
    head = FlowMatchingHead(
        action_dim=DOF,
        chunk_size=16,
        embedding_dim=EMBEDDING_DIM,
        state_dim=2 * DOF + 1,
        n_steps=10,
        frequency_hz=15.0,
    )
    policy = Policy("mock-flow", capabilities, MockBackbone(), head)
    scheduler = AsyncScheduler.from_policy(policy, robot_id="arm_0")
    _tracer.event("example_policy", frequency_mode=capabilities.effective_frequency_mode)

    await scheduler.start()
    for cycle in range(3):
        request = ActionRequest(
            request_id=uuid.uuid4().hex[:8],
            images={"base": torch.rand(3, 224, 224)},
            instruction="pick up the red cube and place it in the bin",
            robot_state=RobotState(
                joint_positions=torch.zeros(DOF),
                joint_velocities=torch.zeros(DOF),
                gripper_state=torch.zeros(1),
            ),
            frame_id=cycle,
        )
        handle = await scheduler.submit(request)
        async for chunk in scheduler.stream(handle):
            _tracer.event(
                "example_chunk",
                request_id=request.request_id,
                cycle=cycle,
                shape=list(chunk.actions.shape),
                head=chunk.head_type,
            )

    _tracer.event("example_stats", **scheduler.stats())
    await scheduler.stop()


if __name__ == "__main__":
    asyncio.run(main())
