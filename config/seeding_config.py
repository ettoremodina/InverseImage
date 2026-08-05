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

    # ==================== smoothness ====================
    # Two different things make seeding look abrupt, and these fix one each.

    # 1. Each seed was written as a hard 0 -> 1 step, so it appeared at full
    #    strength in a single frame. With `ramp_steps > 0` a seed fades in over
    #    that many steps instead. The injection only ever raises a cell (max),
    #    so the NCA's own growth is never clamped back down.
    ramp_steps: int = 8
    ramp_easing: str = 'ease_in_out'

    # 2. Birth times come from branch depth, and the depth histogram is lumpy --
    #    at the defaults 1000 seeds landed on 99 instants, with 33 seeds sharing
    #    the worst one, so they popped in visible batches. 'rank' spaces the
    #    seeds evenly over the window by their depth *order* instead of their
    #    depth *value*, which keeps the tree-following sequence but removes the
    #    clumping. 'time' is the old literal-depth behaviour.
    distribute: str = 'rank'
