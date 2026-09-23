"""Policy capability declaration (ADR-018).

A VLA policy is trained end to end, so facts that decide how the runtime may
schedule it — where robot state enters, what crosses the VLM → action head
boundary, whether the head can be re-run on a cached condition — belong to the
model, not to the runtime. Each model adapter declares them here once; the
schedulers read them instead of assuming.

Two kinds of decoupling are distinguished:

- Spatial: one VLM instance serves the fleet, conditions and head state stay
  per robot. Holds for every model with a VLM → head boundary.
- Temporal (frequency): the head re-runs between VLM refreshes with fresh
  robot state. Only pays off when the head itself consumes fresh input; for a
  prefix-state model, re-running the head on an unchanged condition only
  resamples noise.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vla_runtime.errors import ErrorContext, PolicyCapabilityError
from vla_runtime.vlm.base import ConditionKind, Embedding

StatePlacement = Literal["prefix", "suffix", "head_encoder", "none"]
HeadStructure = Literal["iterative", "single_pass", "monolithic"]
FrequencyMode = Literal["decoupled", "coupled"]


class PolicyCapabilities(BaseModel):
    """What a loaded policy supports, as fixed by its training.

    Attributes:
        state_placement: Where robot state is consumed. ``prefix``: inside the
            VLM prefix (SmolVLA, pi0.5) — state must be passed to encode().
            ``suffix``: as an action-expert token (pi0). ``head_encoder``: by
            a state encoder inside the head (GR00T). ``none``: not consumed.
        condition_kind: What the head reads from the VLM (see Embedding.kind).
        head_structure: ``iterative`` (flow matching / diffusion solver loop),
            ``single_pass`` (parallel decode, e.g. OpenVLA-OFT), or
            ``monolithic`` (backend only exposes a fused predict; the boundary
            cannot be split, e.g. a whole-policy engine).
        max_condition_staleness_ms: How old a reused condition may be before
            it stops being trustworthy. Calibrated per (model, task suite)
            from sim success rate; None means not yet calibrated.
        supports_rtc: Head supports real-time chunking conditioning (ADR-015).

    Raises:
        pydantic.ValidationError: On combinations that cannot be honoured.
    """

    model_config = ConfigDict(frozen=True)

    state_placement: StatePlacement
    condition_kind: ConditionKind
    head_structure: HeadStructure
    max_condition_staleness_ms: Optional[float] = Field(default=None, gt=0)
    supports_rtc: bool = False

    @property
    def frequency_mode(self) -> FrequencyMode:
        """Structural mode: can the head usefully re-run on a cached condition?"""
        fresh_input = self.state_placement in ("suffix", "head_encoder")
        if fresh_input and self.head_structure == "iterative":
            return "decoupled"
        return "coupled"

    @property
    def staleness_calibrated(self) -> bool:
        return self.max_condition_staleness_ms is not None

    @property
    def effective_frequency_mode(self) -> FrequencyMode:
        """Mode the scheduler actually runs.

        A structurally decoupled model with no calibrated staleness tolerance
        runs coupled: reusing a condition without knowing how long it stays
        valid is a guess, and the conservative path never needs one.
        """
        if self.frequency_mode == "decoupled" and self.staleness_calibrated:
            return "decoupled"
        return "coupled"

    @property
    def state_to_backbone(self) -> bool:
        """Whether the scheduler must hand robot state to VLMBackend.encode()."""
        return self.state_placement == "prefix"

    def check_condition(self, embedding: Embedding, context: ErrorContext) -> None:
        """Verify a backend produced the condition kind this policy declares.

        A mismatch means the backend adapter and the declaration disagree
        about the model; carrying on would hand the head a payload it cannot
        read, so it is surfaced where it originates.

        Raises:
            PolicyCapabilityError: If ``embedding.kind`` differs from
                ``condition_kind``.
        """
        if embedding.kind != self.condition_kind:
            raise PolicyCapabilityError(
                f"backend produced a {embedding.kind!r} condition but the policy "
                f"declares condition_kind={self.condition_kind!r}",
                context,
            )

    @model_validator(mode="after")
    def _check_consistency(self) -> "PolicyCapabilities":
        if self.supports_rtc and self.head_structure != "iterative":
            raise ValueError(
                "supports_rtc requires an iterative head: RTC conditions the "
                "denoising loop on already-executed actions"
            )
        if self.max_condition_staleness_ms is not None and self.frequency_mode == "coupled":
            raise ValueError(
                "max_condition_staleness_ms has no effect on a coupled-frequency "
                f"policy (state_placement={self.state_placement!r}, "
                f"head_structure={self.head_structure!r}); the head never "
                "re-runs on a cached condition"
            )
        return self
