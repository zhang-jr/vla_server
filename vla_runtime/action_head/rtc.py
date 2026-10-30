"""Real-Time Chunking guidance for iterative action heads (ADR-015).

RTC (Black et al., arXiv:2506.07339) treats the next chunk as an inpainting
problem: the actions the robot will execute while the new chunk is being
computed are already fixed by the previous chunk, so the denoising loop is
guided to reproduce them and to continue smoothly from them. It is an
inference-time method — no retraining — and works for any flow matching head
whose velocity field can be differentiated with respect to its input.

Convention used throughout this module (the paper's): ``tau`` runs from 0
(pure noise) to 1 (data), the integrator steps ``x <- x + dt * v`` with
``dt > 0``, and the one-step estimate of the clean chunk is
``x1_hat = x + (1 - tau) * v``. Heads that integrate the other way
(SmolVLA / pi0: time 1 -> 0) pass ``tau = 1 - time`` and a sign-flipped
velocity; see ``benchmarks/sim/lerobot_smolvla.py``.

Per denoising step::

    g   = (dx1_hat / dx)^T [ W * (Y - x1_hat) ]           (one VJP)
    v'  = v + min(beta, (1 - tau) / (tau * r_tau^2)) * g
    r_tau^2 = (1 - tau)^2 / (tau^2 + (1 - tau)^2)

With ``jacobian="identity"`` the VJP is replaced by ``g = W * (Y - x1_hat)``,
i.e. ``dx1_hat / dx`` approximated by the identity. That is what LeRobot's
``RTCProcessor`` computes (it evaluates ``v`` before enabling grad on ``x``), and
it saves the backward pass through the velocity network on every step.

``Y`` is the previous chunk's not-yet-executed tail, right-padded to the
chunk length. ``W`` is the prefix weight: 1 for the ``d`` steps that execute
during inference (frozen), a decaying soft weight over the rest of the
overlap, 0 past it. The schedules match LeRobot's ``RTCProcessor``
(``policies/rtc/modeling_rtc.py``), which ``benchmarks/benchmark_sim.py``
uses as the numerical reference.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field

from vla_runtime.action_head.base import ActionHeadContext

PrefixSchedule = Literal["exp", "linear", "ones", "zeros"]
GuidanceJacobian = Literal["vjp", "identity"]


class RTCConfig(BaseModel):
    """RTC guidance parameters.

    Attributes:
        max_guidance_weight: ``beta``, the clip on the guidance weight. The
            paper uses 5; LeRobot defaults to 10. Larger values follow the
            previous chunk more tightly at the cost of sample quality.
        prefix_schedule: How the weight decays over the soft region of the
            overlap. ``exp`` is the paper's choice, ``linear`` LeRobot's
            default; ``ones`` / ``zeros`` are ablations (hard overlap / only
            the frozen steps).
        jacobian: ``vjp`` back-propagates the error through the velocity
            network (the paper); ``identity`` skips it (LeRobot), one forward
            per step instead of a forward plus a backward.
    """

    model_config = ConfigDict(frozen=True)

    max_guidance_weight: float = Field(default=5.0, gt=0)
    prefix_schedule: PrefixSchedule = "exp"
    jacobian: GuidanceJacobian = "vjp"


@dataclass(frozen=True)
class RTCGuidance:
    """Guidance inputs for one chunk, in the head's own action space.

    Attributes:
        target: ``Y`` — [horizon, action_dim], previous tail zero-padded.
        weights: ``W`` — [horizon], prefix weight per step.
        delay_steps: Frozen steps actually applied (``d`` after clamping).
        overlap_steps: Steps covered by the previous tail (soft region end).
        max_guidance_weight: ``beta``.
        jacobian: How the correction is computed (RTCConfig.jacobian).
    """

    target: torch.Tensor
    weights: torch.Tensor
    delay_steps: int
    overlap_steps: int
    max_guidance_weight: float
    jacobian: GuidanceJacobian = "vjp"


def prefix_weights(
    delay_steps: int,
    overlap_steps: int,
    horizon: int,
    schedule: PrefixSchedule = "exp",
) -> torch.Tensor:
    """Per-step prefix weight ``W`` of length ``horizon``.

    Steps ``[0, d)`` weigh 1, ``[d, overlap)`` decay by ``schedule``, and
    ``[overlap, horizon)`` weigh 0, with ``d = min(delay_steps, overlap_steps)``.
    The soft weights use ``c_i = (overlap - i) / (overlap - d + 1)``; ``exp``
    maps them through ``c * (e^c - 1) / (e - 1)``.

    Args:
        delay_steps: Steps executed while the chunk is computed.
        overlap_steps: Steps the previous chunk still covers.
        horizon: Chunk length.
        schedule: Soft-region shape.

    Returns:
        Float32 tensor [horizon].
    """
    overlap = max(0, min(overlap_steps, horizon))
    delay = max(0, min(delay_steps, overlap))
    weights = torch.zeros(horizon, dtype=torch.float32)
    weights[:delay] = 1.0
    if schedule == "zeros" or overlap <= delay:
        return weights
    if schedule == "ones":
        weights[delay:overlap] = 1.0
        return weights
    steps = torch.arange(delay, overlap, dtype=torch.float32)
    c = (overlap - steps) / (overlap - delay + 1)
    if schedule == "exp":
        c = c * torch.expm1(c) / (math.e - 1.0)
    weights[delay:overlap] = c
    return weights


def guidance_weight(tau: float, max_guidance_weight: float) -> float:
    """``min(beta, (1 - tau) / (tau * r_tau^2))``; ``beta`` at ``tau = 0``."""
    if tau <= 0.0:
        return max_guidance_weight
    if tau >= 1.0:
        return 0.0
    one_minus = 1.0 - tau
    inv_r2 = (one_minus * one_minus + tau * tau) / (one_minus * one_minus)
    return min(max_guidance_weight, one_minus / tau * inv_r2)


def prepare_guidance(
    context: ActionHeadContext | None,
    horizon: int,
    action_dim: int,
    config: RTCConfig,
) -> RTCGuidance | None:
    """Turn a scheduler-side context into guidance inputs, or None for plain sampling.

    No context, no previous actions, or an empty tail all mean there is
    nothing to stay consistent with (episode start, drained buffer), so the
    head samples unguided. The tail is truncated to ``horizon`` and zero-padded
    below it; the target stays in the executed-action space, so heads with an
    internal normalisation convert it themselves.

    Args:
        context: What the scheduler knows about the previous chunk.
        horizon: Chunk length of the head.
        action_dim: Action width of the head.
        config: Guidance parameters.

    Raises:
        ValueError: If the previous actions are not [n, action_dim] or the
            delay is negative.
    """
    if context is None or context.prev_actions is None:
        return None
    prev = context.prev_actions
    if prev.dim() != 2 or prev.shape[1] != action_dim:
        raise ValueError(
            f"RTC prev_actions must be [n, {action_dim}], got {tuple(prev.shape)}"
        )
    if context.inference_delay_steps < 0:
        raise ValueError(
            f"RTC inference_delay_steps must be >= 0, got {context.inference_delay_steps}"
        )
    overlap = min(prev.shape[0], horizon)
    if overlap == 0:
        return None
    target = torch.zeros(horizon, action_dim, dtype=torch.float32)
    target[:overlap] = prev[:overlap].detach().to(torch.float32).cpu()
    delay = min(context.inference_delay_steps, overlap)
    return RTCGuidance(
        target=target,
        weights=prefix_weights(delay, overlap, horizon, config.prefix_schedule),
        delay_steps=delay,
        overlap_steps=overlap,
        max_guidance_weight=config.max_guidance_weight,
        jacobian=config.jacobian,
    )


def guided_velocity(
    velocity_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    tau: float,
    target: torch.Tensor,
    weights: torch.Tensor,
    max_guidance_weight: float,
    jacobian: GuidanceJacobian = "vjp",
) -> torch.Tensor:
    """One RTC-guided velocity evaluation (paper convention, see module docstring).

    ``vjp`` costs one forward and one vector-Jacobian product through the
    velocity network, with gradients taken with respect to ``x`` only (the
    network's parameters need not, and should not, require grad). ``identity``
    costs one forward and needs no autograd at all.

    Args:
        velocity_fn: ``x -> v`` at the current ``tau``; must be differentiable.
        x: Current sample [..., horizon, action_dim].
        tau: Current time in [0, 1), 0 = noise.
        target: ``Y`` broadcastable to ``x``.
        weights: ``W`` [horizon], broadcast over the action dimension.
        max_guidance_weight: ``beta``.
        jacobian: ``vjp`` or ``identity``.

    Returns:
        Guided velocity, detached, same shape and dtype as ``velocity_fn(x)``.
    """
    weight = guidance_weight(tau, max_guidance_weight)
    if jacobian == "identity":
        with torch.no_grad():
            v = velocity_fn(x)
            x1_hat = x + (1.0 - tau) * v
            w = weights.to(device=x1_hat.device, dtype=x1_hat.dtype).unsqueeze(-1)
            y = target.to(device=x1_hat.device, dtype=x1_hat.dtype)
            return (v + weight * (y - x1_hat) * w).detach()
    with torch.enable_grad():
        x_in = x.detach().clone().requires_grad_(True)
        v = velocity_fn(x_in)
        x1_hat = x_in + (1.0 - tau) * v
        w = weights.to(device=x1_hat.device, dtype=x1_hat.dtype).unsqueeze(-1)
        y = target.to(device=x1_hat.device, dtype=x1_hat.dtype)
        error = ((y - x1_hat) * w).detach()
        (correction,) = torch.autograd.grad(x1_hat, x_in, grad_outputs=error)
    return (v + weight * correction).detach()
