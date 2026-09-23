"""Diffusion Policy action head runtime.

Implements DDPM and DDIM denoising for diffusion-based VLA models.
Uses the Triton kernel in kernels/diffusion_step.py for denoising steps.

Quantization constraint (ADR-003):
    Iterative denoising accumulates fp16 rounding error. This head MUST
    stay FP16. Any attempt to configure INT4 / W4A16 raises PrecisionDegradationError
    during QuantizationConfig.validate_for_head_type().

Noise schedule:
    We use a cosine schedule for ᾱ_t (Nichol & Dhariwal 2021), which provides
    better signal-to-noise ratio at intermediate steps vs linear.

Architecture:
    x_T ~ N(0,I)  →  T denoising steps  →  x_0 (action chunk)
    At each step, a UNet-style noise predictor ε_θ(x_t, t, emb, state) is called.
"""

from __future__ import annotations

import copy
import math
from typing import Literal

import torch
import torch.nn as nn

from kernels.diffusion_step import ddim_step, ddpm_step
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

_tracer = get_tracer("action_head.diffusion")

SamplerType = Literal["ddpm", "ddim"]


# Floor for ᾱ. The cosine schedule evaluates to exactly 0 at t=1, and every
# sampler divides the residual by sqrt(ᾱ_t) to recover x_0 — at ᾱ_t = 0 that
# amplification is unbounded and overflows fp16 (max 65504) to inf, which the
# next step turns into NaN. Clamping ᾱ away from 0 is the same guard the
# improved-DDPM schedule gets from clipping β to 0.999 (Nichol & Dhariwal 2021).
_ALPHA_BAR_MIN = 1e-4


def _cosine_alpha_bar(t: float, s: float = 0.008) -> float:
    """Cosine noise schedule: ᾱ_t = cos²(π/2 * (t+s)/(1+s)), floored at _ALPHA_BAR_MIN."""
    alpha_bar = math.cos(math.pi / 2 * (t + s) / (1 + s)) ** 2
    return min(max(alpha_bar, _ALPHA_BAR_MIN), 1.0)


def _build_schedule(n_steps: int) -> tuple[list[float], list[float]]:
    """Build cosine alpha_bar schedule for n_steps denoising steps.

    Returns:
        alpha_bars: [n_steps+1] cumulative product, index 0 = pure signal (=1),
            index n_steps = pure noise (≈0).
        sigmas: [n_steps] stochastic noise scales for DDPM.
    """
    timesteps = [i / n_steps for i in range(n_steps + 1)]
    alpha_bars = [_cosine_alpha_bar(t) for t in timesteps]
    sigmas = [
        math.sqrt((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i]) * (1 - alpha_bars[i] / alpha_bars[i - 1]))
        for i in range(1, n_steps + 1)
    ]
    return alpha_bars, sigmas


class NoisePredictorMLP(nn.Module):
    """Simple noise predictor for development / unit testing.

    Production: replace with the actual Diffusion Policy UNet / Transformer.
    """

    def __init__(self, embedding_dim: int, state_dim: int, action_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        in_dim = action_dim + embedding_dim + state_dim + 1  # +1 for time t
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        x: torch.Tensor,       # [B, action_dim]
        t: float,
        embedding: torch.Tensor,    # [B, embedding_dim]
        state: torch.Tensor,        # [B, state_dim]
    ) -> torch.Tensor:
        t_vec = torch.full((x.shape[0], 1), t, device=x.device, dtype=x.dtype)
        inp = torch.cat([x, t_vec, embedding, state], dim=-1)
        return self.net(inp)


class DiffusionHead:
    """ActionHeadRuntime for diffusion-based VLA models.

    Args:
        action_dim: Dimensionality of the action vector.
        chunk_size: Number of action steps per prediction call.
        embedding_dim: Expected VLM embedding dimension.
        state_dim: Flat robot state dimension.
        n_steps: Number of denoising steps (fewer = faster, lower quality).
        sampler: ``"ddpm"`` (stochastic) or ``"ddim"`` (deterministic, faster).
        frequency_hz: Target inference frequency.
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
        n_steps: int = 20,
        sampler: SamplerType = "ddim",
        frequency_hz: float = 5.0,
        device: torch.device | None = None,
    ) -> None:
        self._action_dim = action_dim
        self._chunk_size = chunk_size
        self._embedding_dim = embedding_dim
        self._state_dim = state_dim
        self._n_steps = n_steps
        self._sampler = sampler
        self._frequency_hz = frequency_hz
        device = device or torch.device("cpu")

        self._weights = SharedWeights(
            NoisePredictorMLP(
                embedding_dim=embedding_dim,
                state_dim=state_dim,
                action_dim=action_dim,
            ).to(dtype=torch.float16, device=device),
            device,
        )

        self._alpha_bars, self._sigmas = _build_schedule(n_steps)

    @property
    def _device(self) -> torch.device:
        return self._weights.device

    @property
    def _noise_predictor(self) -> NoisePredictorMLP:
        return self._weights.module

    # ── Weight sharing (ADR-012 ③) ────────────────────────────────────────────

    def spawn(self) -> "DiffusionHead":
        """New instance for another robot, sharing these weights."""
        return copy.copy(self)

    def shares_weights_with(self, other: object) -> bool:
        return getattr(other, "_weights", None) is self._weights

    # ── Protocol properties ───────────────────────────────────────────────────

    @property
    def head_type(self) -> ActionHeadType:
        return "diffusion"

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
        """Predict action chunk via iterative diffusion denoising.

        Runs n_steps of DDPM or DDIM denoising conditioned on embedding + state.

        Args:
            embedding: VLM embedding [embedding_dim].
            robot_state: Current robot state.

        Returns:
            ActionChunk with actions [chunk_size, action_dim].
        """
        ctx = ErrorContext(
            trace_id=_tracer.trace_id,
            module_name="action_head.diffusion",
        )
        with _tracer.span(
            "diffusion_predict",
            n_steps=self._n_steps,
            sampler=self._sampler,
        ) as span:
            try:
                actions = self._denoise(embedding, robot_state)
                span.set_attribute("chunk_size", self._chunk_size)
                return ActionChunk(
                    actions=actions,
                    head_type="diffusion",
                    metadata={
                        "n_steps": self._n_steps,
                        "sampler": self._sampler,
                    },
                )
            except Exception as exc:
                span.set_error(exc)
                raise ActionHeadFailureError("diffusion", exc, ctx) from exc

    # ── Internal ──────────────────────────────────────────────────────────────

    def _denoise(self, embedding: Embedding, robot_state: RobotState) -> torch.Tensor:
        """Reverse diffusion: x_T → x_0."""
        check_conditioning_dims(
            "diffusion",
            self._embedding_dim,
            self._state_dim,
            embedding,
            robot_state,
            self.accepted_condition_kinds,
        )
        B = self._chunk_size
        x = torch.randn(B, self._action_dim, dtype=torch.float16, device=self._device)

        emb = embedding.data.to(self._device, dtype=torch.float16)
        state = robot_state.as_flat_tensor().to(self._device, dtype=torch.float16)
        emb_batch = emb.unsqueeze(0).expand(B, -1)
        state_batch = state.unsqueeze(0).expand(B, -1)

        # Reverse: T → 0
        for i in reversed(range(self._n_steps)):
            t_norm = (i + 1) / self._n_steps   # t in (0, 1]
            ab_t = self._alpha_bars[i + 1]
            ab_prev = self._alpha_bars[i]
            sigma = self._sigmas[i] if self._sampler == "ddpm" else 0.0

            with torch.no_grad():
                noise_pred = self._noise_predictor(x, t_norm, emb_batch, state_batch)

            if self._sampler == "ddpm":
                x = ddpm_step(x, noise_pred, ab_t, ab_prev, sigma)
            else:
                x = ddim_step(x, noise_pred, ab_t, ab_prev)

        return x  # [chunk_size, action_dim]

    def load_weights(self, checkpoint_path: str) -> None:
        """Load into the shared weights — applies to every spawned instance."""
        self._weights.load(checkpoint_path)
        _tracer.event("diffusion_weights_loaded", path=checkpoint_path)

    def to_device(self, device: torch.device) -> None:
        """Move the shared weights — applies to every spawned instance."""
        self._weights.to(device)
