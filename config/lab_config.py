"""
Configuration for the swarm laboratory (Evolutionary_Swarm.md §11).

`SwarmConfig` holds what the *simulation* does; this file holds what the *lab*
does with it -- which image, how long, where the artefacts land, and the named
presets and studies that make a calibration session reproducible instead of a
sequence of remembered command lines.

Three registries, all plain data:

- `PRESETS`  -- named `{dotted param: value}` override sets. `--preset X` runs
  one, `compare` runs two side by side.
- `STUDIES`  -- named hyperparameter grids. `--study X` runs the whole grid and
  writes a showcase folder.
- `SHOWCASE` -- the curated set of presets rendered into the comparison gallery.

A preset is deliberately a *diff* from `SwarmConfig` defaults, not a full
config: reading it tells you what the experiment changed, which is the thing
you actually want to know six weeks later.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from config.common import get_device
from config.swarm_config import SwarmConfig, apply_overrides


# ==================== §11.1 the surrogate ====================

@dataclass
class SurrogateConfig:
    """
    How to fake the stage-2 output (§11.1).

    `nca_size` alone reproduces the real defect -- missing high frequency. The
    other two exist so a calibration can be checked against a starting point
    that is *wrong* and not merely *soft*, which is the honest test.
    """
    nca_size: int = 128
    color_noise: float = 0.0
    posterize_levels: int = 0


# ==================== the lab itself ====================

@dataclass
class LabConfig:
    """Everything the lab needs that is not a property of the swarm."""

    image: str = 'images/jellyfish.png'
    output_root: str = 'outputs/swarm'

    # §11.3 sizing. `steps` here overrides SwarmConfig.sim_steps for lab runs,
    # so one flag changes both the climate clock and the loop length.
    work_size: int = 256
    population_cap: int = 4096
    steps: int = 600
    seed: int = 0
    device: str = field(default_factory=get_device)

    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)

    # Metrics every step are ~6 device syncs; at lab scale that is comparable
    # to the step itself, so sweeps trade curve resolution for throughput.
    metrics_stride: int = 2

    # --- artefacts ---
    filmstrip_frames: int = 4     # intermediate canvases captured per run, excluding base/final
    thumb_size: int = 192         # contact-sheet cell size, px
    display_scale: int = 3        # live/compare window magnification

    def to_swarm_config(self, overrides: Dict[str, Any] = None) -> SwarmConfig:
        """A `SwarmConfig` at lab scale, with `overrides` applied on top."""
        base = SwarmConfig(
            work_size=self.work_size,
            population_cap=self.population_cap,
            sim_steps=self.steps,
            device=self.device,
            seed=self.seed,
            metrics_stride=self.metrics_stride,
        )
        return apply_overrides(base, overrides or {})


# ==================== presets (§11.2 compare) ====================
#
# `doc` is not decoration: a preset without a stated hypothesis is a magic
# number, and the whole point of the lab is to stop producing those.

@dataclass
class Preset:
    doc: str
    overrides: Dict[str, Any] = field(default_factory=dict)


# The two calibration results every other preset is built on. Both were found
# in the lab, and the studies that found them are named alongside.

# `survival`: the §15 metabolism starves. Gain is scaled up and both costs cut
# until an agent that paints well can reach the reproduction threshold.
_METABOLISM = {'gain_scale': 16.0, 'cost_life': 0.002, 'cost_deposit': 0.008}

# `locality`: the one that matters. A gene is only worth inheriting if its owner
# stays where that colour is right -- at speed 1.0 an agent crosses the whole
# canvas twice per run, so its colour is right nowhere in particular and the
# selection signal is noise. Near-stationary agents with widely scattered
# children move the search into *space* (§2.1's spatial mutation) and leave the
# colour gene free to refine. This is the difference between the swarm beating
# its input and merely smearing it.
_ROOTED = {**_METABOLISM, 'tolerance': 0.0, 'deposit_alpha': 0.06,
           'speed': 0.1, 'birth_jitter': 6.0}


PRESETS: Dict[str, Preset] = {
    'defaults': Preset(
        doc='The §15 starting values, untouched. Documented as a starting point, '
            'not a tuned one -- and measurably starving: population pins to the floor.',
        overrides={},
    ),

    # ---- the two fixes, isolated so the showcase shows what each one bought ----
    'fed': Preset(
        doc='Metabolism only: gain scaled up and both costs cut, so an agent that paints '
            'well can reach the reproduction threshold. Reproduction restarts -- and the '
            'picture still gets worse, which is how we know metabolism was not the problem.',
        overrides=dict(_METABOLISM),
    ),
    'rooted': Preset(
        doc='The calibrated baseline: `fed` plus near-stationary agents and widely '
            'scattered children. The first configuration whose agents eat more than they '
            'spend, and the first that improves on the stage-2 input instead of smearing it.',
        overrides=dict(_ROOTED),
    ),

    # ---- the look (§15, ordered by visual impact) ----
    'impressionist': Preset(
        doc='`rooted` with a coarse fitness scale and a fat brush: error measured on a '
            'reduced image, so gain is shared across a whole cell and the strokes go gestural.',
        overrides={**_ROOTED, 'fitness_scale': 4, 'brush_radius': 3.0, 'deposit_alpha': 0.2},
    ),
    'watercolour': Preset(
        doc='`rooted` with a thin, layered stroke and slow pigment turnover -- the opposite '
            'end of §15 `deposit_alpha` from `impressionist`.',
        overrides={**_ROOTED, 'deposit_alpha': 0.03, 'brush_radius': 1.0, 'decay_max': 0.02},
    ),
    'filaments': Preset(
        doc='`rooted` with narrow sensors, persistent pheromone and mobile agents: the '
            'Physarum end of §15, tight trails instead of even coverage. Costs accuracy '
            'for structure on purpose.',
        overrides={**_ROOTED, 'speed': 0.6, 'sensor_angle': 0.35, 'sensor_distance': 9.0,
                   'evaporation': 0.98, 'deposit_pheromone': 2.0},
    ),

    # ---- the climate arc (§8) ----
    'famine': Preset(
        doc='`rooted` with an engineered arc: cheap life early (population boom), '
            'fourfold cost by the end (anticipated famine), gain tapering throughout.',
        overrides={**_ROOTED,
                   'climate_cost_life': {'start': 0.5, 'end': 4.0, 'easing': 'ease_in'},
                   'climate_gain_scale': {'start': 1.5, 'end': 0.8, 'easing': 'linear'}},
    ),

    # ---- the line the design draws (§2.2) ----
    'guided': Preset(
        doc='CHEATING, on purpose. guidance > 0 lets agents smell the error field directly '
            '(§2.2). Kept as the upper bound the honest runs are measured against, never '
            'as an output setting.',
        overrides={**_ROOTED, 'guidance': 3.0},
    ),
}

# The curated gallery: which presets get rendered side by side by `--mode showcase`.
# Ordered as an argument -- defaults starves, fed eats but still smears, rooted
# is the one that works, then the looks built on it, then the cheat as a ceiling.
SHOWCASE: List[str] = ['defaults', 'fed', 'rooted', 'impressionist',
                       'watercolour', 'filaments', 'famine', 'guided']


# ==================== studies (§11.2 sweep) ====================

@dataclass
class Axis:
    """One swept parameter: a dotted path into `SwarmConfig` and the values to try."""
    param: str
    values: List[Any]


@dataclass
class Study:
    """
    A named grid. `base` is applied to every cell, so a study can be run on top
    of an earlier study's winner instead of on top of defaults -- which is the
    only way a sequence of one-parameter sweeps means anything when the
    parameters are as coupled as §13 warns.
    """
    doc: str
    axes: List[Axis]
    base: Dict[str, Any] = field(default_factory=dict)
    steps: int = None          # None = LabConfig.steps


STUDIES: Dict[str, Study] = {
    # --- the two diagnostic studies, run against §15 defaults ---
    'survival': Study(
        doc='Does anybody live? gain_scale against cost_life, on the §15 defaults. '
            '§13 lists "estinzione precoce" as a named failure mode and this grid locates '
            'its edge: gain_scale dominates, cost_life barely registers at these levels, '
            'and no cell in the grid beats the stage-2 input.',
        axes=[Axis('gain_scale', [1.0, 4.0, 8.0, 16.0]),
              Axis('cost_life', [0.0005, 0.002, 0.005, 0.01])],
    ),
    'locality': Study(
        doc='The study that found the real defect: how far an agent travels (§5.3) '
            'against how far its children land (§2.1). A gene is only heritable fitness '
            'if its owner stays where that colour is right -- this grid is where mean gain '
            'first turns positive.',
        axes=[Axis('speed', [0.05, 0.1, 0.2, 0.5]),
              Axis('birth_jitter', [1.5, 6.0, 12.0])],
        base=_METABOLISM | {'tolerance': 0.0, 'deposit_alpha': 0.06},
    ),

    # --- the look, all on top of the calibrated baseline ---
    'tolerance': Study(
        doc='§13 calls tolerance the first parameter to tune, and §16 pairs it with '
            'fitness_scale: together they decide faithful-versus-gestural.',
        axes=[Axis('tolerance', [0.0, 0.01, 0.02, 0.04]),
              Axis('fitness_scale', [1, 2, 4])],
        base=_ROOTED,
    ),
    'brush': Study(
        doc='The stroke itself (§15): width against opacity, watercolour to tempera.',
        axes=[Axis('brush_radius', [1.0, 1.5, 3.0]),
              Axis('deposit_alpha', [0.03, 0.06, 0.15, 0.3])],
        base=_ROOTED,
    ),
    'pigment': Study(
        doc='How fast the picture forgets (§5.9) against how long the trails last '
            '(§5.10) -- the pair that decides whether it settles or seethes.',
        axes=[Axis('decay_max', [0.01, 0.05, 0.15]),
              Axis('evaporation', [0.9, 0.95, 0.99])],
        base=_ROOTED,
    ),
    'exploration': Study(
        doc='Mutation amplitude against the spatial mutation rate (§2.1): colour variety '
            'against lineages that root in place.',
        axes=[Axis('mutation_sigma', [0.005, 0.02, 0.05]),
              Axis('birth_jitter', [1.5, 6.0, 12.0])],
        base=_ROOTED,
    ),
    'population': Study(
        doc='Density (§11.3) against the reproduction bar (§5.8): how many agents the '
            'canvas can actually feed, and how selective they have to be.',
        axes=[Axis('population_cap', [1024, 4096, 16384]),
              Axis('reproduction_threshold', [1.0, 1.5, 2.5])],
        base=_ROOTED,
    ),
}
