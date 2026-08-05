"""
One run -> one number, with the reasoning kept attached.

The tuner needs a scalar to maximise; a person needs to know *why* a run scored
what it did, or the winning configuration is just another magic number. So
`evaluate` returns both: `total`, and the per-criterion breakdown that produced
it. Every report in the suite prints the breakdown next to the score, which is
what makes it possible to look at a 0.62 and say "fidelity is fine, it just
never settles" instead of shrugging.

Where the numbers come from:

- `swarm.metrics.RunSummary` -- the population's story (§12).
- `swarm.quality.QualityMetrics` -- the picture's (this package's `quality.py`).
- three derived readings computed here, because they only mean anything once
  those two are put side by side.

The criteria themselves, their bands and their weights live in
`config/tuning_config.py`. Nothing in this module decides what "good" is; it
only does the arithmetic.
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List

from config.swarm_config import SwarmConfig
from config.tuning_config import Criterion, ObjectiveConfig
from swarm.metrics import RunSummary
from swarm.quality import QualityMetrics


@dataclass
class CriterionScore:
    """One criterion's verdict on one run."""
    key: str
    value: float
    score: float           # [0, 1]
    weight: float
    contribution: float    # score * weight, i.e. how much of `total` it carries
    doc: str


@dataclass
class ObjectiveScore:
    """The composite, and every piece of it."""
    total: float                        # in [0, 1]; the number the tuner maximises
    raw: float                          # before the regression gate
    gate: float                         # the multiplier the gate applied
    parts: List[CriterionScore] = field(default_factory=list)

    @property
    def weakest(self) -> CriterionScore:
        """The criterion losing the most weighted score -- what to fix next."""
        return min(self.parts, key=lambda p: p.contribution - p.weight)

    def as_row(self) -> Dict[str, float]:
        """Flat `{score_<key>: value}` mapping for the history CSV."""
        row = {'score': self.total, 'score_raw': self.raw, 'score_gate': self.gate}
        row.update({f'score_{p.key}': p.score for p in self.parts})
        return row


def flatten(summary: RunSummary, quality: QualityMetrics, config: SwarmConfig) -> Dict[str, float]:
    """
    Merge the two metric records into the flat namespace the criteria index into.

    The two overlap on three names -- both measure error and improvement -- and
    they do not mean the same thing: `RunSummary` measures the whole frame,
    `QualityMetrics` measures the subject only. On this target the background is
    82% of the pixels and is already correct, so the global figure is roughly a
    fifth of the real one. The masked reading wins the plain name because it is
    the one that answers the question; the global keeps a `global_` prefix
    rather than being dropped, because a large gap between the two means the
    swarm is painting the background.
    """
    flat = {k: v for k, v in asdict(summary).items()}
    flat['global_improvement'] = flat.pop('improvement')
    flat['global_final_error'] = flat.pop('final_error')
    flat['global_baseline_error'] = flat.pop('baseline_error')
    flat.update(asdict(quality))

    steps = max(1, int(summary.steps))
    cap = max(1, int(config.population_cap))

    # --- the three derived readings ---
    # Detail in the right places, measured as a gain over the input rather than
    # as an absolute: the absolute depends on how blurred the stage-2 output is,
    # the gain does not.
    flat['alignment_gain'] = quality.gradient_alignment - quality.base_gradient_alignment
    # Population as a share of the cap, so the criterion survives a change of
    # `population_cap` between the lab and production.
    flat['population_fill'] = summary.mean_population / cap
    # Reproductions per agent per step -- the rate at which heredity happens.
    flat['birth_rate'] = summary.total_births / steps / cap

    return flat


def _gate(improvement: float, objective: ObjectiveConfig) -> float:
    """
    The one hard rule, as a curve that never quite reaches zero.

    A run that hands back a worse picture than stage 2 gave it has failed
    whatever else it achieved -- pretty texture on a degraded image is still a
    degraded image. But the penalty has to keep *ranking* failures, not flatten
    them: the first calibration attempts sit around -13% improvement, and a
    linear ramp to zero over a couple of percent scored every one of them
    exactly 0.0, which left the search with no gradient at all and turned the
    first generations into a random walk.

    `span / (span - improvement)` halves the score at `regression_span` of
    damage and keeps decaying after that without ever hitting the floor, so
    "slightly bad" always outranks "very bad" and the search can climb out of
    the failing region it starts in.
    """
    if not objective.regression_gate:
        return 1.0
    if improvement >= 0.0:
        return 1.0
    span = max(objective.regression_span, 1e-9)
    return span / (span - improvement)


def evaluate(summary: RunSummary, quality: QualityMetrics, config: SwarmConfig,
             objective: ObjectiveConfig = None) -> ObjectiveScore:
    """Score one finished run against the criteria in `config/tuning_config.py`."""
    objective = objective or ObjectiveConfig()
    flat = flatten(summary, quality, config)

    parts: List[CriterionScore] = []
    raw = 0.0
    for criterion in objective.criteria:
        value = flat.get(criterion.key)
        score = criterion.score(value)
        raw += score * criterion.weight
        parts.append(CriterionScore(
            key=criterion.key, value=float(value) if value is not None else float('nan'),
            score=score, weight=criterion.weight,
            contribution=score * criterion.weight, doc=criterion.doc,
        ))

    gate = _gate(quality.improvement, objective)
    return ObjectiveScore(total=raw * gate, raw=raw, gate=gate, parts=parts)


def describe(score: ObjectiveScore, width: int = 22) -> str:
    """A one-run scorecard, for logs and for the HTML report's tooltip text."""
    lines = [f'score {score.total:.4f}  (raw {score.raw:.4f} x gate {score.gate:.2f})']
    for part in sorted(score.parts, key=lambda p: -p.weight):
        bar = '#' * int(round(part.score * 20))
        lines.append(f'  {part.key:<{width}} {part.value: 9.4f}  {part.score:4.2f} '
                     f'x{part.weight:.2f}  {bar}')
    return '\n'.join(lines)


def criteria_reference(objective: ObjectiveConfig = None) -> List[Criterion]:
    """The active criteria, for reports that want to print the definitions."""
    return list((objective or ObjectiveConfig()).criteria)
