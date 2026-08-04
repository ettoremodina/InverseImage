"""
Configuration for progressive NCA seeding (PLAN 3.2).

Kept out of `NCAConfig` on purpose: that dataclass is pickled inside every
trained checkpoint, so adding fields to it makes old models fail to restore.
Seeding is an inference-time decision anyway.
"""

from dataclasses import dataclass
from typing import Literal

SeedOrder = Literal['depth', 'random', 'none']


@dataclass
class ProgressiveSeedingConfig:
    """
    When each seed is injected into the NCA state.

    The tree and the tissue are supposed to grow together: a seed lights up when
    the branch that carries it has been drawn, not all of them at t=0.

    Note that the current model is trained with every seed on from the first
    step, so a staggered seeding is out of distribution. That is the point of
    the experiment in PLAN 3.2 step 1: render it as is and look for the two ways
    it can break -- already grown regions perturbed by a neighbour lighting up,
    and late seeds that fail to take.
    """

    enabled: bool = True

    # 'depth'  : birth time from the depth of the branch tip the seed sits on;
    # 'random' : birth time drawn at random (useful as a control);
    # 'none'   : every seed at step 0, i.e. the old behaviour.
    order: SeedOrder = 'depth'

    # Fraction of the rollout over which the seeding is spread. 0.55 means the
    # last seed lights up at 55% of the NCA steps, leaving the rest of the
    # rollout to close the tissue.
    spread: float = 0.55

    # Small random offset added to the birth times, as a fraction of the
    # rollout: seeds at the same depth should not all pop at the same instant.
    jitter: float = 0.02

    seed: int = 3
