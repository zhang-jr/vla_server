"""Policies the closed-loop benchmarks can evaluate.

A ``BenchPolicy`` bundles a validated ``vla_runtime.policy.Policy`` (backbone +
head prototype + capability declaration — what the runtime modes drive) with
an optional ``native`` reference: the model's own monolithic inference, which
the ``native`` mode runs as the no-runtime baseline.

Registry:
    mock     CPU-only: latency-injecting mock backbone + the built-in
             FlowMatchingHead with random weights. Plumbing checks only.
    smolvla  LeRobot SmolVLA split at the prefix-KV boundary (GPU).
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from benchmarks.sim.envs import EnvSpec, Observation
from vla_runtime.action_head.base import RobotState
from vla_runtime.action_head.flow_matching import FlowMatchingHead
from vla_runtime.capabilities import PolicyCapabilities
from vla_runtime.policy import Policy
from vla_runtime.vlm.base import Embedding, TokenStream

NativeInfer = Callable[[Observation], Awaitable[torch.Tensor]]


@dataclass
class BenchPolicy:
    """A policy ready for benchmarking.

    Attributes:
        name: Report label.
        policy: Backbone + head prototype + capabilities (runtime modes).
        n_action_steps: Actions executed per chunk in the blocking modes
            (LeRobot's ``n_action_steps``).
        native: Reference monolithic inference, observation → [T, action_dim].
            None when the model has no separate reference path.
        info: Metadata recorded in every result (checkpoint, dtype, ...).
        close: Releases model memory at the end of the run.
    """

    name: str
    policy: Policy
    n_action_steps: int
    native: NativeInfer | None = None
    info: dict[str, Any] = field(default_factory=dict)
    close: Callable[[], None] = lambda: None


# ── Mock ──────────────────────────────────────────────────────────────────────

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


def _build_mock(spec: EnvSpec, options: dict[str, str]) -> BenchPolicy:
    opts = dict(options)
    embedding_dim = int(opts.pop("embedding_dim", "256"))
    chunk_size = int(opts.pop("chunk_size", "16"))
    n_action_steps = int(opts.pop("n_action_steps", str(chunk_size)))
    staleness_ms = opts.pop("staleness_ms", None)
    backbone = MockBackbone(
        embedding_dim=embedding_dim,
        latency_ms=float(opts.pop("latency_ms", "0")),
        jitter_ms=float(opts.pop("jitter_ms", "0")),
        num_views=len(spec.camera_names),
    )
    head = FlowMatchingHead(
        action_dim=spec.action_dim,
        chunk_size=chunk_size,
        embedding_dim=embedding_dim,
        state_dim=spec.state_dim,
        n_steps=int(opts.pop("n_steps", "10")),
    )
    if opts:
        raise ValueError(f"unknown --policy-opt for mock: {sorted(opts)}")
    # pi0-like topology; uncalibrated staleness keeps it coupled (ADR-018).
    caps = PolicyCapabilities(
        state_placement="suffix",
        condition_kind="pooled",
        head_structure="iterative",
        max_condition_staleness_ms=float(staleness_ms) if staleness_ms else None,
    )
    policy = Policy(name="mock", capabilities=caps, vlm_backend=backbone, head_prototype=head)
    reference_head = head.spawn()

    async def native(obs: Observation) -> torch.Tensor:
        embedding = await backbone.encode(obs["images"], obs["instruction"])
        chunk = await asyncio.to_thread(reference_head.predict, embedding, obs["robot_state"])
        return chunk.actions

    return BenchPolicy(
        name="mock",
        policy=policy,
        n_action_steps=n_action_steps,
        native=native,
        info={"embedding_dim": embedding_dim, "chunk_size": chunk_size},
    )


def _build_smolvla(spec: EnvSpec, options: dict[str, str]) -> BenchPolicy:
    from benchmarks.sim.lerobot_smolvla import build_smolvla

    return build_smolvla(spec, options)


_REGISTRY: dict[str, Callable[[EnvSpec, dict[str, str]], BenchPolicy]] = {
    "mock": _build_mock,
    "smolvla": _build_smolvla,
}


def make_policy(name: str, spec: EnvSpec, options: dict[str, str]) -> BenchPolicy:
    """Build a registered policy against an environment spec.

    Args:
        name: Registry key (``mock``, ``smolvla``).
        spec: The environment the policy will act in.
        options: ``--policy-opt key=value`` pairs.

    Raises:
        ValueError: Unknown policy or option.
    """
    if name not in _REGISTRY:
        raise ValueError(f"unknown policy {name!r}; expected one of {sorted(_REGISTRY)}")
    return _REGISTRY[name](spec, options)
