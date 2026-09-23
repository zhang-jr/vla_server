"""Policy — the model adapter layer (ADR-018).

A VLA policy is one trained model split at the VLM → action head boundary.
``Policy`` holds the three things that must agree about that model — the
backbone backend, the head, and the capability declaration — and checks them
against each other once, at construction, instead of leaving each scheduler
to trust that they were wired consistently.

Schedulers are built from it (``AsyncScheduler.from_policy``,
``FleetScheduler.from_policy``), so a deployment names one policy rather than
passing a backend, a head and a declaration that could come from different
models.

What is checked statically here, and what at runtime:

- Here: the head can read the declared condition kind; the head paradigm is
  consistent with the declared head structure.
- At runtime (schedulers): each backend output's kind matches the
  declaration — a backend's output kind is not known until it runs.
"""

from __future__ import annotations

from dataclasses import dataclass

from vla_runtime.action_head.base import ActionHeadRuntime, WeightSharingHead
from vla_runtime.capabilities import PolicyCapabilities
from vla_runtime.vlm.base import VLMBackend

# Head paradigm → head structures it can have.
_STRUCTURES_BY_HEAD_TYPE: dict[str, tuple[str, ...]] = {
    "flow_matching": ("iterative",),
    "diffusion": ("iterative",),
    "discrete_token": ("single_pass", "monolithic"),
}


class PolicyConfigurationError(ValueError):
    """Backend, head and capability declaration disagree about the model.

    Raised at construction, before any request is served — a configuration
    mistake, not a runtime failure, hence a ValueError rather than a
    VLARuntimeError carrying a trace context.
    """


@dataclass(frozen=True)
class Policy:
    """A loaded VLA policy.

    Args:
        name: Identifier for traces and logs (e.g. "pi0-libero").
        capabilities: What the model supports, as fixed by its training.
        vlm_backend: The backbone. One instance serves every robot.
        head_prototype: A head instance that is never itself handed to a
            robot; per-robot heads are spawned from it and share its weights.

    Raises:
        PolicyConfigurationError: If the head cannot read the declared
            condition kind, or its paradigm contradicts the declared head
            structure.
    """

    name: str
    capabilities: PolicyCapabilities
    vlm_backend: VLMBackend
    head_prototype: ActionHeadRuntime

    def __post_init__(self) -> None:
        caps = self.capabilities
        head = self.head_prototype

        accepted = getattr(head, "accepted_condition_kinds", None)
        if accepted is not None and caps.condition_kind not in accepted:
            raise PolicyConfigurationError(
                f"policy {self.name!r} declares condition_kind={caps.condition_kind!r} "
                f"but {type(head).__name__} reads {list(accepted)}"
            )

        allowed = _STRUCTURES_BY_HEAD_TYPE.get(head.head_type)
        if allowed is not None and caps.head_structure not in allowed:
            raise PolicyConfigurationError(
                f"policy {self.name!r} declares head_structure={caps.head_structure!r} "
                f"but a {head.head_type} head is {' or '.join(allowed)}"
            )

    def head_for_robot(self) -> ActionHeadRuntime:
        """A new head instance for one robot, sharing the prototype's weights.

        Raises:
            PolicyConfigurationError: If the head cannot share weights; such a
                head needs one fully separate instance per robot, which the
                policy cannot create on its own.
        """
        if not isinstance(self.head_prototype, WeightSharingHead):
            raise PolicyConfigurationError(
                f"{type(self.head_prototype).__name__} cannot share weights (no spawn()); "
                "construct one instance per robot and register it explicitly"
            )
        return self.head_prototype.spawn()
