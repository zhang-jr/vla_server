"""Flow Matching action head runtime.

Implements Euler and RK4 integration for flow matching-based VLA models
(Pi0, SmolVLA style). Uses the Triton kernel in kernels/flow_matching_step.py
for the inner integration loop.

Quantization constraint (ADR-003):
    This head MUST run in FP16. Iterative denoising accumulates numerical
    error — lower precision causes task success rate degradation.

Architecture (simplified):
    noise z_0 ~ N(0, I)  →  v_θ network  →  z_1 (action chunk)

    The velocity network v_θ takes (z_t, t, embedding, robot_state) and
    outputs the velocity field. We integrate from t=0 to t=1 with n_steps
    Euler steps (or n_steps/4 RK4 steps for equivalent accuracy).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from kernels.flow_matching_step import flow_matching_euler_step, flow_matching_rk4_step
from vla_runtime.action_head.base import (
    ActionChunk,
    ActionHeadType,
    RobotState,
    SharedWeights,
    check_conditioning_dims,
)
from vla_runtime.errors import ActionHeadFailureError, ErrorContext
from vla_runtime.observability.tracer import get_tracer
from vla_runtime.vlm.base import Embedding

_tracer = get_tracer("action_head.flow_matching")


class VelocityMLP(nn.Module):
    """Simple velocity field network for development / unit testing.

    Production models replace this with the actual Pi0 / SmolVLA velocity net.
    Input: [embedding_dim + state_dim + action_dim + 1(time)]
    Output: [action_dim] velocity
    """

    def __init__(
        self,
        embedding_dim: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        in_dim = embedding_dim + state_dim + action_dim + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        z: torch.Tensor,        # [B, action_dim]
        t: float,
        embedding: torch.Tensor,    # [B, embedding_dim]
        state: torch.Tensor,        # [B, state_dim]
    ) -> torch.Tensor:
        t_tensor = torch.full((z.shape[0], 1), t, device=z.device, dtype=z.dtype)
        x = torch.cat([z, t_tensor, embedding, state], dim=-1)
        return self.net(x)


class FlowMatchingHead:
    """ActionHeadRuntime for flow matching-based VLA models.

    Args:
        action_dim: Dimensionality of the action vector.
        chunk_size: Number of action steps to predict per call.
        embedding_dim: Expected embedding dimension from VLMBackend.
        state_dim: Flat robot state dimension.
        n_steps: Number of ODE integration steps (Euler: more=accurate, slower).
        use_rk4: Use RK4 instead of Euler (4x cost, higher accuracy).
        frequency_hz: Target inference frequency (used by scheduler).
        device: Compute device.
    """

    # The conditioning MLP reads one pooled vector (ADR-019).
    accepted_condition_kinds: tuple[str, ...] = ("pooled",)

    def __init__(
        self,
        action_dim: int = 7,
        chunk_size: int = 16,
        embedding_dim: int = 4096,
        state_dim: int = 14,
        n_steps: int = 10,
        use_rk4: bool = False,
        frequency_hz: float = 10.0,
        device: torch.device | None = None,
    ) -> None:
        self._action_dim = action_dim
        self._chunk_size = chunk_size
        self._embedding_dim = embedding_dim
        self._state_dim = state_dim
        self._n_steps = n_steps
        self._use_rk4 = use_rk4
        self._frequency_hz = frequency_hz
        device = device or torch.device("cpu")

        self._weights = SharedWeights(
            VelocityMLP(
                embedding_dim=embedding_dim,
                state_dim=state_dim,
                action_dim=action_dim,
            ).to(dtype=torch.float16, device=device),
            device,
        )

    @property
    def _device(self) -> torch.device:
        return self._weights.device

    @property
    def _velocity_net(self) -> VelocityMLP:
        return self._weights.module

    # ── Weight sharing (ADR-012 ③) ────────────────────────────────────────────

    def spawn(self) -> "FlowMatchingHead":
        """New instance for another robot, sharing these weights."""
        return copy.copy(self)

    def shares_weights_with(self, other: object) -> bool:
        return getattr(other, "_weights", None) is self._weights

    # ── Protocol properties ───────────────────────────────────────────────────

    @property
    def head_type(self) -> ActionHeadType:
        return "flow_matching"

    @property
    def frequency_hz(self) -> float:
        return self._frequency_hz

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def action_dim(self) -> int:
        return self._action_dim

    # ── Core interface ────────────────────────────────────────────────────────

    def predict(self, embedding: Embedding, robot_state: RobotState) -> ActionChunk:
        """Predict action chunk via flow matching integration.

        Runs the ODE integration from z_0~N(0,I) to z_1 using the velocity
        network conditioned on the shared embedding and robot state.

        Args:
            embedding: VLM embedding [embedding_dim].
            robot_state: Current robot state.

        Returns:
            ActionChunk with actions [chunk_size, action_dim].
        """
        request_id = None  # caller may inject via thread-local in production
        ctx = ErrorContext(
            trace_id=_tracer.trace_id,
            module_name="action_head.flow_matching",
        )

        with _tracer.span("flow_matching_predict", n_steps=self._n_steps) as span:
            try:
                actions = self._integrate(embedding, robot_state)
                span.set_attribute("chunk_size", self._chunk_size)
                span.set_attribute("integration", "rk4" if self._use_rk4 else "euler")
                return ActionChunk(
                    actions=actions,
                    head_type="flow_matching",
                    metadata={
                        "n_steps": self._n_steps,
                        "integrator": "rk4" if self._use_rk4 else "euler",
                    },
                )
            except Exception as exc:
                span.set_error(exc)
                raise ActionHeadFailureError("flow_matching", exc, ctx) from exc

    # ── Internal ──────────────────────────────────────────────────────────────

    def _integrate(self, embedding: Embedding, robot_state: RobotState) -> torch.Tensor:
        """ODE integration: z_0 → z_1 producing [chunk_size, action_dim]."""
        check_conditioning_dims(
            "flow_matching",
            self._embedding_dim,
            self._state_dim,
            embedding,
            robot_state,
            self.accepted_condition_kinds,
        )
        B = self._chunk_size

        # Initial noise z_0 ~ N(0, I) in fp16
        z = torch.randn(B, self._action_dim, dtype=torch.float16, device=self._device)

        # Prepare conditioning
        emb = embedding.data.to(self._device, dtype=torch.float16)
        state = robot_state.as_flat_tensor().to(self._device, dtype=torch.float16)

        # Broadcast to batch dimension [B, D]
        emb_batch = emb.unsqueeze(0).expand(B, -1)
        state_batch = state.unsqueeze(0).expand(B, -1)

        dt = 1.0 / self._n_steps
        times = [i * dt for i in range(self._n_steps)]

        if self._use_rk4:
            for t in times:
                def vel_fn(z_in: torch.Tensor, t_in: float) -> torch.Tensor:
                    with torch.no_grad():
                        return self._velocity_net(z_in, t_in, emb_batch, state_batch)
                z = flow_matching_rk4_step(z, vel_fn, t, dt)
        else:
            for t in times:
                with torch.no_grad():
                    velocity = self._velocity_net(z, t, emb_batch, state_batch)
                z = flow_matching_euler_step(z, velocity, dt)

        return z  # [chunk_size, action_dim]

    def load_weights(self, checkpoint_path: str) -> None:
        """Load into the shared weights — applies to every spawned instance."""
        self._weights.load(checkpoint_path)
        _tracer.event("flow_matching_weights_loaded", path=checkpoint_path)

    def to_device(self, device: torch.device) -> None:
        """Move the shared weights — applies to every spawned instance."""
        self._weights.to(device)
