"""
Timing of the single timeline (PLAN 3.6).

The old 40/60 split was made of disjoint percentages, which cannot express what
the pipeline is supposed to look like now: the stages overlap. Each stage gets
an interval in seconds and an easing curve, and the intervals are meant to
cross -- the NCA enters while the tree is still growing, the swarm enters on
tissue that is already mature.

`validate()` enforces exactly that: every stage must start before the previous
one ends, otherwise the phases silently go back to being sequential and all the
work of 3.1 is lost.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

from utils.log import get_logger

logger = get_logger(__name__)


@dataclass
class StageWindow:
    """When a stage is active, in seconds, and how its progress is shaped."""

    start: float
    end: float
    easing: str = 'linear'

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class TimingConfig:
    total_duration: float = 25.0   # length of the finished video, hold included
    hold: float = 3.0              # final stillness, only the sway breathing

    sca: StageWindow = field(default_factory=lambda: StageWindow(0.0, 7.0, 'ease_out'))
    nca: StageWindow = field(default_factory=lambda: StageWindow(2.0, 16.0, 'ease_in_out'))
    swarm: StageWindow = field(default_factory=lambda: StageWindow(12.0, 22.0, 'linear'))
    scaffold_fade: StageWindow = field(
        default_factory=lambda: StageWindow(10.0, 18.0, 'ease_in_out'))

    # Order in which the overlap constraint is checked. scaffold_fade is not a
    # stage, it is a consequence, so it is left out.
    _chain: Tuple[str, ...] = ('sca', 'nca', 'swarm')

    def windows(self) -> List[Tuple[str, StageWindow]]:
        return [(name, getattr(self, name)) for name in self._chain]

    def validate(self, strict: bool = False):
        """
        Check the overlap constraint and the total length.

        With `strict` the first violation raises; otherwise it is logged as a
        warning, because a sequential timing is a valid thing to render on
        purpose -- it just is not what the design wants.
        """
        problems = []

        for (_, current), (next_name, following) in zip(self.windows(), self.windows()[1:]):
            if following.start >= current.end:
                problems.append(
                    f"stage '{next_name}' starts at {following.start}s, after the previous "
                    f"stage ends at {current.end}s: the phases are sequential, not overlapped"
                )

        for name, window in self.windows():
            if window.end <= window.start:
                problems.append(f"stage '{name}' has an empty window "
                                f"({window.start}s -> {window.end}s)")

        last_end = max(w.end for _, w in self.windows())
        if last_end + self.hold > self.total_duration + 1e-6:
            problems.append(
                f"the last stage ends at {last_end}s and the hold is {self.hold}s, "
                f"which exceeds total_duration ({self.total_duration}s)"
            )

        for problem in problems:
            if strict:
                raise ValueError(f'TimingConfig: {problem}')
            logger.warning('TimingConfig: %s', problem)

        return not problems
