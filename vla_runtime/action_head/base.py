"""Action Head Runtime Protocol and shared types.

ActionHeadRuntime is intentionally synchronous — frequency guarantees are
provided by the caller (AsyncScheduler), not by the action head itself.
Each robot instance gets its own ActionHeadRuntime; they are NOT shared.

Data flow:
    Embedding + RobotState → ActionHeadRuntime.predict() → ActionChunk

Three head paradigms (from CLAUDE.md):
    1. discrete_token  — OpenVLA, RT-2: FAST tokenizer + OFT parallel decode
    2. flow_matching   — Pi0, SmolVLA: Euler / RK4 integration
    3. diffusion       — Diffusion Policy: iterative denoising

Quantization constraint (ADR-003):
    flow_matching and diffusion MUST stay FP16.
    discrete_token may use W8A16 with sim validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import torch

ActionHeadType = Literal["discrete_token", "flow_matching", "diffusion"]


# ── Shared data types ─────────────────────────────────────────────────────────

@dataclass
class RobotState:
    """Current state of the robot, passed as a tensor channel (not tokenized).

    Attributes:
        joint_positions: [dof] joint angles in radians.
        joint_velocities: [dof] joint velocities in rad/s.
        gripper_state: Scalar or [n_fingers] gripper position (0=open, 1=closed).
        ee_pose: [7] end-effector pose (xyz + quaternion), optional.
        extra: Additional modality-specific state tensors.
    """

    joint_positions: torch.Tensor           # [dof]
    joint_velocities: torch.Tensor          # [dof]
    gripper_state: torch.Tensor             # [] or [n_fingers]
    ee_pose: torch.Tensor | None = None     # [7]
    extra: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def dof(self) -> int:
        return self.joint_positions.shape[0]

    @property
    def flat_dim(self) -> int:
        """Width of as_flat_tensor() — what a head's ``state_dim`` must equal.

        Note this is NOT ``2 * dof``: the gripper occupies its own channel, and
        ee_pose / extra add more. Deriving it from a RobotState instance is the
        only safe way to size a head's conditioning input.
        """
        width = (
            self.joint_positions.shape[0]
            + self.joint_velocities.shape[0]
            + self.gripper_state.numel()
        )
        if self.ee_pose is not None:
            width += self.ee_pose.numel()
        return width + sum(v.numel() for v in self.extra.values())

    def as_flat_tensor(self) -> torch.Tensor:
        """Concatenate all state tensors into a single flat vector."""
        parts = [self.joint_positions, self.joint_velocities, self.gripper_state.flatten()]
        if self.ee_pose is not None:
            parts.append(self.ee_pose)
        for v in self.extra.values():
            parts.append(v.flatten())
        return torch.cat(parts)


@dataclass
class ActionChunk:
    """A chunk of predicted actions covering multiple control steps.

    Attributes:
        actions: [chunk_size, action_dim] predicted action sequence.
        chunk_size: Number of time steps in this chunk.
        action_dim: Dimensionality of each action vector.
        head_type: Which action head paradigm produced this chunk.
        confidence: Optional per-step confidence scores [chunk_size].
        metadata: Head-specific extras (e.g. denoising steps used).
    """

    actions: torch.Tensor           # [chunk_size, action_dim]
    head_type: ActionHeadType
    confidence: torch.Tensor | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def chunk_size(self) -> int:
        return self.actions.shape[0]

    @property
    def action_dim(self) -> int:
        return self.actions.shape[1]

    def step(self, idx: int) -> torch.Tensor:
        """Return the action for a single control step."""
        return self.actions[idx]


# ── Shared conditioning checks ────────────────────────────────────────────────

def check_conditioning_dims(
    head_type: ActionHeadType,
    embedding_dim: int,
    state_dim: int,
    embedding: "vla_runtime.vlm.base.Embedding",  # noqa: F821
    robot_state: RobotState,
) -> None:
    """Validate that the conditioning inputs match how the head was configured.

    Without this the mismatch surfaces as a bare matmul shape error from inside
    the conditioning MLP, which says nothing about which input was wrong. The
    common trap is sizing ``state_dim`` as ``2 * dof`` and forgetting that the
    gripper occupies its own channel in RobotState.flat_dim.

    Args:
        head_type: Head paradigm, for the error message.
        embedding_dim: Embedding width the head was built for.
        state_dim: Flat state width the head was built for.
        embedding: The embedding about to be consumed.
        robot_state: The robot state about to be consumed.

    Raises:
        ValueError: If either width disagrees with the head's configuration.
            Callers wrap this in ActionHeadFailureError alongside their own
            failure handling.
    """
    actual_embedding_dim = embedding.data.shape[-1]
    if actual_embedding_dim != embedding_dim:
        raise ValueError(
            f"{head_type} head built for embedding_dim={embedding_dim} but got "
            f"{actual_embedding_dim}"
        )
    if robot_state.flat_dim != state_dim:
        raise ValueError(
            f"{head_type} head built for state_dim={state_dim} but RobotState "
            f"flattens to {robot_state.flat_dim} "
            f"(dof={robot_state.dof}, gripper={robot_state.gripper_state.numel()}, "
            f"ee_pose={'yes' if robot_state.ee_pose is not None else 'no'}, "
            f"extra={sum(v.numel() for v in robot_state.extra.values())}); "
            f"size state_dim from RobotState.flat_dim, not 2 * dof"
        )


# ── ActionHeadRuntime Protocol ────────────────────────────────────────────────

@runtime_checkable
class ActionHeadRuntime(Protocol):
    """Protocol that every action head implementation must satisfy.

    Implementations:
        - DiscreteTokenHead (discrete_token)
        - FlowMatchingHead  (flow_matching)
        - DiffusionHead     (diffusion)
        - TensorRTBackend   (wraps any of the above with TRT acceleration)

    The predict() method is synchronous. The AsyncScheduler manages timing.
    Each robot instance has its own ActionHeadRuntime — state isolation is
    guaranteed at the caller level, not within the head.
    """

    @property
    def head_type(self) -> ActionHeadType:
        """Which action generation paradigm this head implements."""
        ...

    @property
    def frequency_hz(self) -> float:
        """Target inference frequency in Hz (5-20 Hz typical).

        The AsyncScheduler uses this to schedule predict() calls.
        If a single predict() call takes longer than 1/frequency_hz seconds,
        BudgetExceededError should be raised by the scheduler.
        """
        ...

    @property
    def chunk_size(self) -> int:
        """Number of action steps predicted per call.

        Larger chunks amortise per-call overhead but increase latency before
        the robot receives updated actions. Typical: 8-16 steps.
        """
        ...

    @property
    def action_dim(self) -> int:
        """Dimensionality of each action vector (joints + gripper)."""
        ...

    def predict(
        self,
        embedding: "vla_runtime.vlm.base.Embedding",  # noqa: F821
        robot_state: RobotState,
    ) -> ActionChunk:
        """Predict the next action chunk from embedding + robot state.

        Args:
            embedding: Shared embedding from VLMBackend.encode().
            robot_state: Current robot state (NOT tokenized into the prompt).

        Returns:
            ActionChunk with actions tensor [chunk_size, action_dim].

        Raises:
            ActionHeadFailureError: On any unrecoverable prediction error.
        """
        ...

    def load_weights(self, checkpoint_path: str) -> None:
        """Load weights from a checkpoint file."""
        ...

    def to_device(self, device: torch.device) -> None:
        """Move weights to the specified device."""
        ...
