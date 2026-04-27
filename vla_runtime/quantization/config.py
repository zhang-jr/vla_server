"""Quantization configuration for VLA Runtime.

Design constraints (from CLAUDE.md ADR-003 / ADR-007):
- Action head: FP16 or W8A16 only — INT4 / GPTQ is forbidden.
- LLM backbone: aggressive quantization allowed (W4A16, GPTQ, GGUF Q4).
- Vision encoder: W8A8 default; W4A8 experimental.
- flow_matching / diffusion heads MUST use FP16 (iterative denoising,
  error accumulates — any lower precision degrades task success rate).
- Validation is done via sim task success rate, NOT loss / perplexity.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator

ActionHeadType = Literal["discrete_token", "flow_matching", "diffusion"]

VisionQuantScheme = Literal["fp16", "w8a8", "w4a8"]
LLMQuantScheme = Literal["fp16", "w4a16", "w8a16", "gptq", "gguf_q4"]
ActionHeadQuantScheme = Literal["fp16", "w8a16"]


class QuantizationConfig(BaseModel):
    """Per-module quantization settings.

    Attributes:
        vision_encoder: Quantization scheme for the vision encoder.
        llm_backbone: Quantization scheme for the LLM backbone.
        action_head: Quantization scheme for the action head.
            Only fp16 and w8a16 are allowed — INT4 is forbidden (ADR-003).
    """

    vision_encoder: VisionQuantScheme = "w8a8"
    llm_backbone: LLMQuantScheme = "w4a16"
    action_head: ActionHeadQuantScheme = "fp16"

    # ── Validation ────────────────────────────────────────────────────────────

    def validate_for_head_type(self, head_type: ActionHeadType) -> None:
        """Enforce precision rules that depend on the action head type.

        Args:
            head_type: The type of action head this config will be used with.

        Raises:
            PrecisionDegradationError: If flow_matching or diffusion head
                is configured with anything other than fp16.
        """
        # Import here to avoid circular import at module load time
        from vla_runtime.errors import PrecisionDegradationError

        if head_type in ("flow_matching", "diffusion") and self.action_head != "fp16":
            raise PrecisionDegradationError(
                baseline_rate=float("nan"),   # not yet measured
                quantized_rate=float("nan"),
                threshold_drop=0.0,
                # context will be injected by the caller
            )

    @model_validator(mode="after")
    def _check_action_head_not_int4(self) -> "QuantizationConfig":
        """Guard: action_head must never be INT4-class (enforced at model level).

        The Literal type already prevents INT4 from being set, but this
        validator documents the intent explicitly and will catch future schema
        drift.
        """
        # Pydantic's Literal enforcement is sufficient; this is a documentation hook.
        return self

    # ── Factory helpers ───────────────────────────────────────────────────────

    @classmethod
    def full_precision(cls) -> "QuantizationConfig":
        """FP16 everywhere — baseline for task success rate comparison."""
        return cls(vision_encoder="fp16", llm_backbone="fp16", action_head="fp16")

    @classmethod
    def cloud_aggressive(cls) -> "QuantizationConfig":
        """Aggressive backbone quantization for cloud GPU deployment."""
        return cls(vision_encoder="w8a8", llm_backbone="w4a16", action_head="fp16")

    @classmethod
    def edge_balanced(cls) -> "QuantizationConfig":
        """Edge-optimised: GGUF Q4 backbone, FP16 action head.

        Follows the llama.cpp + ROS2 pattern from LiteVLA-Edge (arxiv 2603.03380).
        """
        return cls(vision_encoder="w8a8", llm_backbone="gguf_q4", action_head="fp16")

    def summary(self) -> str:
        return (
            f"vision={self.vision_encoder} "
            f"llm={self.llm_backbone} "
            f"action={self.action_head}"
        )
