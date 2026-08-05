"""
Progressive seeding of the NCA state (PLAN 3.2).

Today every seed is written into the state at t=0 (`nca/data.py`), so the
tissue starts everywhere at once and the SCA -> NCA transition is a cut. Here
each seed carries a birth step derived from the depth of the branch tip it sits
on, and is injected when the tree has actually reached it.

Step 1 of the plan is exactly this: rendering only, model untouched. The model
was trained with all the seeds on, so a staggered seeding is out of
distribution -- the schedule is the instrument for finding out how badly.

The retraining of step 2, if it turns out to be needed, reuses this same
`SeedSchedule` inside the training rollout.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from config.seeding_config import ProgressiveSeedingConfig
from rendering.easing import apply_easing
from utils.log import get_logger

logger = get_logger(__name__)

Position = Tuple[int, int]


def birth_times_from_sca(positions: Sequence[Position], sca_data: dict,
                         target_size: int) -> np.ndarray:
    """
    Birth time in [0, 1] for each seed, from the depth of the tree at that point.

    The seeds come from the branch tips but do not carry their depth, so it is
    recovered geometrically: each seed is matched to the closest point of the
    exported polylines and inherits its depth. That keeps this working on render
    data written before the depths were exported alongside the seeds.
    """
    from scipy.spatial import cKDTree

    polylines = sca_data['polylines']
    if not polylines or len(positions) == 0:
        return np.zeros(len(positions))

    points = np.concatenate([np.asarray(p['points'], dtype=np.float64) for p in polylines])
    depths = np.concatenate([np.asarray(p['depths'], dtype=np.float64) for p in polylines])

    # Seeds live on the NCA grid; the polylines live in source pixels.
    scale_x = sca_data['source_width'] / target_size
    scale_y = sca_data['source_height'] / target_size
    queries = np.asarray(positions, dtype=np.float64) * np.array([scale_x, scale_y])

    _, indices = cKDTree(points).query(queries, k=1)
    matched = depths[indices]

    peak = max(float(depths.max()), 1e-9)
    return np.clip(matched / peak, 0.0, 1.0)


class SeedSchedule:
    """
    Which seeds are planted at which step of the rollout.

    Use it as the `hook` of `CAModel.forward` / `CAModel.generate_frames`: it is
    called between two steps and writes the seeds whose time has come.
    """

    def __init__(self, positions: Sequence[Position], birth_steps: Sequence[int],
                 channel_start: int = 3, value: float = 1.0,
                 ramp_steps: int = 0, ramp_easing: str = 'ease_in_out'):
        self.positions = [tuple(int(v) for v in p) for p in positions]
        self.birth_steps = [int(s) for s in birth_steps]
        self.channel_start = channel_start
        self.value = value
        self.ramp_steps = max(0, int(ramp_steps))
        self.ramp_easing = ramp_easing

        # A seed is "active" for `ramp_steps` after its birth, so a step has to
        # touch every seed still ramping, not only the ones born on it.
        self.by_step: Dict[int, List[Position]] = {}
        for position, step in zip(self.positions, self.birth_steps):
            for offset in range(self.ramp_steps + 1):
                self.by_step.setdefault(step + offset, []).append((position, offset))

    # ------------------------------------------------------------- factories
    @classmethod
    def from_birth_times(cls, positions: Sequence[Position], birth_times: np.ndarray,
                         steps: int, config: ProgressiveSeedingConfig = None,
                         channel_start: int = 3) -> 'SeedSchedule':
        """Turn normalised birth times into concrete step numbers."""
        config = config or ProgressiveSeedingConfig()
        times = np.asarray(birth_times, dtype=np.float64).copy()
        staggered = config.enabled and config.order != 'none'

        if not staggered:
            times[:] = 0.0
        elif config.order == 'random':
            times = np.random.default_rng(config.seed).random(len(times))

        if staggered and getattr(config, 'distribute', 'time') == 'rank' and len(times) > 1:
            # Replace each birth time by its rank, normalised. Branch depth is
            # distributed very unevenly -- most seeds sit on tips of similar
            # depth -- so using the raw value makes them arrive in lumps. The
            # rank keeps the order the tree grows in and spaces the arrivals.
            order = np.argsort(times, kind='stable')
            ranks = np.empty(len(times), dtype=np.float64)
            ranks[order] = np.arange(len(times), dtype=np.float64)
            times = ranks / (len(times) - 1)

        if config.jitter > 0 and staggered:
            rng = np.random.default_rng(config.seed)
            times = times + rng.normal(0.0, config.jitter, size=len(times))

        times = np.clip(times, 0.0, 1.0) * max(0.0, min(1.0, config.spread))
        birth_steps = np.clip((times * steps).astype(int), 0, max(0, steps - 1))

        return cls(positions, birth_steps, channel_start=channel_start,
                   ramp_steps=config.ramp_steps if staggered else 0,
                   ramp_easing=config.ramp_easing)

    @classmethod
    def from_sca(cls, positions: Sequence[Position], sca_data: dict, target_size: int,
                 steps: int, config: ProgressiveSeedingConfig = None,
                 channel_start: int = 3) -> 'SeedSchedule':
        """The usual entry point: seeds + SCA render data -> schedule."""
        config = config or ProgressiveSeedingConfig()
        times = (birth_times_from_sca(positions, sca_data, target_size)
                 if config.order == 'depth' else np.zeros(len(positions)))
        schedule = cls.from_birth_times(positions, times, steps, config, channel_start)
        logger.info('Seeding: %s', schedule.describe())
        return schedule

    # ---------------------------------------------------------------- runtime
    def inject(self, state: torch.Tensor, step: int) -> torch.Tensor:
        """
        Write the seeds active at `step` into the state, in place.

        Only the hidden channels are touched (`channel_start` onwards, alpha
        included), exactly as `create_seed` does.

        With `ramp_steps > 0` a seed is not switched on but faded in: the value
        written grows along the easing curve over its ramp. The write is a
        `maximum`, never an assignment, for two reasons -- a cell the NCA has
        already grown past the ramp value must not be pulled back down, and the
        seed must keep asserting itself while it fades in rather than being
        erased by the first update step after its birth.
        """
        due = self.by_step.get(step)
        if not due:
            return state

        size_y, size_x = state.shape[-2], state.shape[-1]
        for (x, y), offset in due:
            if not (0 <= x < size_x and 0 <= y < size_y):
                continue

            if self.ramp_steps:
                t = min(1.0, (offset + 1) / (self.ramp_steps + 1))
                amount = self.value * apply_easing(self.ramp_easing, t)
            else:
                amount = self.value

            cell = state[:, self.channel_start:, y, x]
            state[:, self.channel_start:, y, x] = torch.clamp(cell, min=amount)

        return state

    def initial_state(self, channel_n: int, size: int, device: str = 'cpu') -> torch.Tensor:
        """An empty state of the right shape; the schedule fills it as it goes."""
        return torch.zeros(1, channel_n, size, size, device=device)

    def describe(self) -> str:
        if not self.birth_steps:
            return 'no seeds'
        steps = np.asarray(self.birth_steps)
        _, counts = np.unique(steps, return_counts=True)
        ramp = f', fading in over {self.ramp_steps} steps' if self.ramp_steps else ', hard on/off'
        return (f'{len(steps)} seeds over steps {steps.min()}-{steps.max()} '
                f'({len(counts)} distinct births, at most {counts.max()} at once){ramp}')
