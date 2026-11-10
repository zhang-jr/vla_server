"""π0.5 (LeRobot) as a benchmark policy.

The adapter is ``vla_runtime.models.pi05``; options are listed in
``vla_runtime.models.registry`` and ``benchmarks.sim.lerobot_smolvla``.
"""

from __future__ import annotations

from benchmarks.sim.envs import EnvSpec
from benchmarks.sim.lerobot_smolvla import build_lerobot
from benchmarks.sim.policies import BenchPolicy


def build_pi05(spec: EnvSpec, options: dict[str, str]) -> BenchPolicy:
    """Load π0.5 and wrap it as a BenchPolicy."""
    return build_lerobot("pi05", spec, options)
