"""A CPU-only mock policy: plumbing checks without a GPU or model weights.

``MockBackbone`` sleeps for a configurable latency and emits a pooled vector;
``mock_policy`` pairs it with the reference ``FlowMatchingHead`` (random
weights). The actions are meaningless; the shapes, timing and scheduling are
real. Used by the sim harness (``--policy mock``) and the policy server
(``--policy mock``).
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Mapping

import torch

from vla_runtime.action_head.base import RobotState
from vla_runtime.action_head.flow_matching import FlowMatchingHead
from vla_runtime.action_head.rtc import RTCConfig
from vla_runtime.capabilities import PolicyCapabilities
from vla_runtime.policy import Policy
from vla_runtime.vlm.base import Embedding, TokenStream


class MockBackbone:
    """VLMBackend stand-in: sleeps for a configurable latency, emits a pooled vector.

    The vector is a fixed projection of per-view image means, so it changes with
    the observation the way a real condition would.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        latency_ms: float = 0.0,
        jitter_ms: float = 0.0,
        num_views: int = 2,
        seed: int = 0,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self._projection = torch.randn(num_views, embedding_dim, generator=generator)
        self._latency_ms = latency_ms
        self._jitter_ms = jitter_ms
        self._rng = random.Random(seed)

    @property
    def model_id(self) -> str:
        return "mock-backbone"

    @property
    def latency_budget_ms(self) -> float:
        return max(1.0, 2.0 * (self._latency_ms + self._jitter_ms))

    @property
    def kv_cache_enabled(self) -> bool:
        return False

    async def encode(
        self,
        images: Mapping[str, torch.Tensor],
        text: str,
        state: RobotState | None = None,
    ) -> Embedding:
        delay_ms = self._latency_ms + self._rng.uniform(0.0, self._jitter_ms)
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)
        means = torch.stack([img.float().mean() for img in images.values()])
        views = min(means.shape[0], self._projection.shape[0])
        pooled = means[:views] @ self._projection[:views]
        return Embedding(data=pooled.to(torch.float16), kind="pooled", model_id=self.model_id)

    async def generate(
        self,
        images: Mapping[str, torch.Tensor],
        text: str,
        state: RobotState | None = None,
    ) -> TokenStream:
        raise NotImplementedError("mock backbone has no token output")

    async def warmup(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass


def mock_policy(
    *,
    num_views: int,
    action_dim: int,
    state_dim: int,
    chunk_size: int = 16,
    embedding_dim: int = 256,
    n_steps: int = 10,
    latency_ms: float = 0.0,
    jitter_ms: float = 0.0,
    frequency_hz: float = 10.0,
    rtc: bool = False,
    rtc_config: RTCConfig | None = None,
    max_condition_staleness_ms: float | None = None,
) -> Policy:
    """Mock backbone + reference flow-matching head as a validated Policy.

    Args:
        num_views: Camera views the backbone reads.
        action_dim: Width of each action.
        state_dim: Flat robot-state width the head reads (``RobotState.flat_dim``).
        chunk_size: Actions per chunk.
        embedding_dim: Width of the pooled condition.
        n_steps: Euler steps of the reference head.
        latency_ms: Simulated backbone latency.
        jitter_ms: Uniform extra latency on top of ``latency_ms``.
        frequency_hz: The head's nominal rate.
        rtc: Declare ``supports_rtc``.
        rtc_config: Guidance parameters.
        max_condition_staleness_ms: Calibrated staleness; None keeps the
            policy coupled (ADR-018).
    """
    backbone = MockBackbone(
        embedding_dim=embedding_dim,
        latency_ms=latency_ms,
        jitter_ms=jitter_ms,
        num_views=num_views,
    )
    head = FlowMatchingHead(
        action_dim=action_dim,
        chunk_size=chunk_size,
        embedding_dim=embedding_dim,
        state_dim=state_dim,
        n_steps=n_steps,
        frequency_hz=frequency_hz,
        rtc_config=rtc_config,
    )
    # pi0-like topology; uncalibrated staleness keeps it coupled (ADR-018).
    caps = PolicyCapabilities(
        state_placement="suffix",
        condition_kind="pooled",
        head_structure="iterative",
        max_condition_staleness_ms=max_condition_staleness_ms,
        supports_rtc=rtc,
    )
    name = "mock+rtc" if rtc else "mock"
    return Policy(name=name, capabilities=caps, vlm_backend=backbone, head_prototype=head)
