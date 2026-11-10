"""Fleet admission control — load-adaptive replanning and admission (ADR-023).

The fleet's latency is set by how many requests share each dispatch round:
with the VLM run once per robot and the head batched across robots
(ADR-022), a round where all N robots ask costs about

    round_ms(N) ≈ N × vlm_ms + head_ms(N)

(V3: 4 robots predicted 256 ms, measured P50 264 ms). Two knobs act on it:

- **Replan threshold** — a robot asks for its next chunk once its queued
  actions fall to ``threshold × chunk_size``. Lowering it means fewer
  requests per robot, so fewer robots in each round; it costs success at light
  load, where fresh chunks are free. The fleet holds no chunk buffer (each
  robot's client does), so the threshold is advice: every chunk carries
  ``replan_threshold`` and the client applies it to its next request.
- **Admission** — refuse a new robot whose arrival would push the predicted
  round past a hard limit, rather than slow every robot already served.

Decisions use the *predicted* round from the number of registered robots and
the measured stage costs, never the latency the fleet just observed: lowering
the threshold drops the observed latency at once, so feedback on it would
flip back and forth between levels. The prediction depends only on N and the
per-stage costs, which the threshold does not change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

AdmissionLevel = Literal["fresh", "relaxed"]


@dataclass
class AdmissionConfig:
    """Admission control settings.

    Attributes:
        latency_target_ms: Predicted round latency above which robots are told
            to replan less often (calibrated per policy from the fleet sweep,
            where the fixed-threshold success curves cross).
        fresh_threshold: Replan threshold while under target.
        relaxed_threshold: Replan threshold while over target.
        hysteresis: Return to ``fresh`` only once the prediction falls below
            ``latency_target_ms × (1 - hysteresis)``.
        admit_latency_ms: Refuse a registration whose predicted round would
            exceed this; None admits up to ``FleetConfig.max_robots``.
        ewma_alpha: Weight of each new stage-latency sample.
        skip_first: Samples dropped per stage and head batch size before
            estimating: the first call at a new batch size captures its CUDA
            Graph (V3: ~700 ms against ~155 ms warm), which would hold the
            estimate over target for several rounds.
    """

    latency_target_ms: float
    fresh_threshold: float = 0.9
    relaxed_threshold: float = 0.5
    hysteresis: float = 0.1
    admit_latency_ms: float | None = None
    ewma_alpha: float = 0.2
    skip_first: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.relaxed_threshold <= self.fresh_threshold <= 1.0:
            raise ValueError(
                "need 0 <= relaxed_threshold <= fresh_threshold <= 1, got "
                f"{self.relaxed_threshold} / {self.fresh_threshold}"
            )
        if not 0.0 <= self.hysteresis < 1.0:
            raise ValueError(f"hysteresis must be in [0, 1), got {self.hysteresis}")
        if not 0.0 < self.ewma_alpha <= 1.0:
            raise ValueError(f"ewma_alpha must be in (0, 1], got {self.ewma_alpha}")
        if self.skip_first < 0:
            raise ValueError(f"skip_first must be >= 0, got {self.skip_first}")


@dataclass
class LevelChange:
    """A replan-threshold switch, for the fleet's trace."""

    level: AdmissionLevel
    threshold: float
    num_robots: int
    predicted_round_ms: float


@dataclass
class AdmissionController:
    """Stage-cost estimates and the level they imply. Not thread-safe; the
    fleet calls it from its event loop only."""

    config: AdmissionConfig
    level: AdmissionLevel = field(default="fresh", init=False)
    vlm_ms: float | None = field(default=None, init=False)
    head_ms: dict[int, float] = field(default_factory=dict, init=False)
    _seen: dict[str | int, int] = field(default_factory=dict, init=False, repr=False)

    @property
    def threshold(self) -> float:
        if self.level == "fresh":
            return self.config.fresh_threshold
        return self.config.relaxed_threshold

    def record_vlm(self, ms: float) -> None:
        if self._warm("vlm"):
            self.vlm_ms = self._ewma(self.vlm_ms, ms)

    def record_head(self, batch_size: int, ms: float) -> None:
        if self._warm(batch_size):
            self.head_ms[batch_size] = self._ewma(self.head_ms.get(batch_size), ms)

    def head_estimate(self, batch_size: int) -> float | None:
        """Head cost at ``batch_size``: measured, else a line through the measured sizes.

        Under the relaxed level large batches are rare, so sizes beyond the
        measured ones are extrapolated; one measured size is taken as flat.
        """
        if batch_size in self.head_ms:
            return self.head_ms[batch_size]
        if not self.head_ms:
            return None
        if len(self.head_ms) == 1:
            return next(iter(self.head_ms.values()))
        xs = list(self.head_ms)
        ys = [self.head_ms[x] for x in xs]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        return max(0.0, my + slope * (batch_size - mx))

    def predict_round_ms(self, num_robots: int) -> float | None:
        """Round latency with every robot asking; None until both stages were measured."""
        if num_robots <= 0:
            return 0.0
        head = self.head_estimate(num_robots)
        if self.vlm_ms is None or head is None:
            return None
        return num_robots * self.vlm_ms + head

    def update(self, num_robots: int) -> LevelChange | None:
        """Re-evaluate the level for ``num_robots``; the change, if any."""
        predicted = self.predict_round_ms(num_robots)
        if predicted is None:
            return None
        target = self.config.latency_target_ms
        if self.level == "fresh" and predicted > target:
            self.level = "relaxed"
        elif self.level == "relaxed" and predicted <= target * (1.0 - self.config.hysteresis):
            self.level = "fresh"
        else:
            return None
        return LevelChange(self.level, self.threshold, num_robots, predicted)

    def admits(self, num_robots: int) -> tuple[bool, float | None]:
        """Whether a fleet of ``num_robots`` stays within ``admit_latency_ms``.

        Without measurements yet there is nothing to predict from, so the
        robot is admitted; ``max_robots`` still bounds the fleet.
        """
        limit = self.config.admit_latency_ms
        predicted = self.predict_round_ms(num_robots)
        if limit is None or predicted is None:
            return True, predicted
        return predicted <= limit, predicted

    def snapshot(self) -> dict:
        return {
            "level": self.level,
            "replan_threshold": self.threshold,
            "vlm_ms": None if self.vlm_ms is None else round(self.vlm_ms, 2),
            "head_ms": {b: round(ms, 2) for b, ms in sorted(self.head_ms.items())},
        }

    def _warm(self, key: str | int) -> bool:
        """Count a sample for ``key``; False while it is among the first ``skip_first``."""
        seen = self._seen.get(key, 0)
        self._seen[key] = seen + 1
        return seen >= self.config.skip_first

    def _ewma(self, current: float | None, sample: float) -> float:
        if current is None:
            return sample
        return current + self.config.ewma_alpha * (sample - current)
