"""Policies the closed-loop benchmarks can evaluate.

A ``BenchPolicy`` bundles a validated ``vla_runtime.policy.Policy`` (backbone +
head prototype + capability declaration — what the runtime modes drive) with
an optional ``native`` reference: the model's own monolithic inference, which
the ``native`` mode runs as the no-runtime baseline.

Registry:
    mock     CPU-only: latency-injecting mock backbone (vla_runtime.models.mock) +
             the reference FlowMatchingHead with random weights. Plumbing checks only.
    smolvla  LeRobot SmolVLA split at the prefix-KV boundary (GPU).
    pi05     LeRobot π0.5 split at the PaliGemma prefix-KV boundary (GPU).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from benchmarks.sim.envs import EnvSpec, Observation
from vla_runtime.action_head.flow_matching import FlowMatchingHead
from vla_runtime.action_head.rtc import RTCConfig
from vla_runtime.capabilities import PolicyCapabilities
from vla_runtime.models.mock import MockBackbone
from vla_runtime.policy import Policy

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

def _build_mock(spec: EnvSpec, options: dict[str, str]) -> BenchPolicy:
    opts = dict(options)
    embedding_dim = int(opts.pop("embedding_dim", "256"))
    chunk_size = int(opts.pop("chunk_size", "16"))
    n_action_steps = int(opts.pop("n_action_steps", str(chunk_size)))
    staleness_ms = opts.pop("staleness_ms", None)
    rtc = opts.pop("rtc", "0").lower() not in ("0", "false", "no", "")
    rtc_fields: dict[str, Any] = {}
    if "rtc_beta" in opts:
        rtc_fields["max_guidance_weight"] = float(opts.pop("rtc_beta"))
    if "rtc_schedule" in opts:
        rtc_fields["prefix_schedule"] = opts.pop("rtc_schedule")
    if "rtc_jacobian" in opts:
        rtc_fields["jacobian"] = opts.pop("rtc_jacobian")
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
        rtc_config=RTCConfig(**rtc_fields),
    )
    if opts:
        raise ValueError(f"unknown --policy-opt for mock: {sorted(opts)}")
    # pi0-like topology; uncalibrated staleness keeps it coupled (ADR-018).
    caps = PolicyCapabilities(
        state_placement="suffix",
        condition_kind="pooled",
        head_structure="iterative",
        max_condition_staleness_ms=float(staleness_ms) if staleness_ms else None,
        supports_rtc=rtc,
    )
    name = "mock+rtc" if rtc else "mock"
    policy = Policy(name=name, capabilities=caps, vlm_backend=backbone, head_prototype=head)
    reference_head = head.spawn()

    async def native(obs: Observation) -> torch.Tensor:
        embedding = await backbone.encode(obs["images"], obs["instruction"])
        chunk = await asyncio.to_thread(reference_head.predict, embedding, obs["robot_state"])
        return chunk.actions

    return BenchPolicy(
        name=name,
        policy=policy,
        n_action_steps=n_action_steps,
        native=native,
        info={"embedding_dim": embedding_dim, "chunk_size": chunk_size, "rtc": rtc},
    )


def _build_smolvla(spec: EnvSpec, options: dict[str, str]) -> BenchPolicy:
    from benchmarks.sim.lerobot_smolvla import build_smolvla

    return build_smolvla(spec, options)


def _build_pi05(spec: EnvSpec, options: dict[str, str]) -> BenchPolicy:
    from benchmarks.sim.lerobot_pi05 import build_pi05

    return build_pi05(spec, options)


_REGISTRY: dict[str, Callable[[EnvSpec, dict[str, str]], BenchPolicy]] = {
    "mock": _build_mock,
    "smolvla": _build_smolvla,
    "pi05": _build_pi05,
}


def make_policy(name: str, spec: EnvSpec, options: dict[str, str]) -> BenchPolicy:
    """Build a registered policy against an environment spec.

    Args:
        name: Registry key (``mock``, ``smolvla``, ``pi05``).
        spec: The environment the policy will act in.
        options: ``--policy-opt key=value`` pairs.

    Raises:
        ValueError: Unknown policy or option.
    """
    if name not in _REGISTRY:
        raise ValueError(f"unknown policy {name!r}; expected one of {sorted(_REGISTRY)}")
    return _REGISTRY[name](spec, options)
