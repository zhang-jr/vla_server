"""Shared builders for unit tests: robot state, requests, policy capabilities."""

from __future__ import annotations

import uuid
from typing import Optional

import torch

from vla_runtime.action_head.base import RobotState
from vla_runtime.capabilities import PolicyCapabilities
from vla_runtime.vlm.base import ActionRequest, ConditionKind


def make_robot_state(dof: int = 7) -> RobotState:
    """Flattens to ``2 * dof + 1`` (the gripper has its own channel)."""
    return RobotState(
        joint_positions=torch.zeros(dof),
        joint_velocities=torch.zeros(dof),
        gripper_state=torch.zeros(1),
    )


def make_request(
    request_id: Optional[str] = None,
    dof: int = 7,
    image_size: int = 64,
    cameras: tuple[str, ...] = ("base",),
) -> ActionRequest:
    return ActionRequest(
        request_id=request_id or uuid.uuid4().hex[:8],
        images={cam: torch.rand(3, image_size, image_size) for cam in cameras},
        instruction="pick up object",
        robot_state=make_robot_state(dof),
    )


def decoupled_caps(
    condition_kind: ConditionKind = "pooled",
    max_condition_staleness_ms: float = 60_000.0,
) -> PolicyCapabilities:
    """pi0-like: state enters the action expert, head is iterative, calibrated."""
    return PolicyCapabilities(
        state_placement="suffix",
        condition_kind=condition_kind,
        head_structure="iterative",
        max_condition_staleness_ms=max_condition_staleness_ms,
    )


def coupled_caps(condition_kind: ConditionKind = "pooled") -> PolicyCapabilities:
    """SmolVLA-like: state enters the backbone prefix."""
    return PolicyCapabilities(
        state_placement="prefix",
        condition_kind=condition_kind,
        head_structure="iterative",
    )
