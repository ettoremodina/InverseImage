"""
What "a good swarm run" means, as numbers -- and where the tuner is allowed to look.

Three registries, all plain data, mirroring `lab_config.py`:

- `CRITERIA`  -- the success criteria of stage 3, each one a measurable band
  with a weight. This is the file that decides what the tuner optimises, so it
  is the file to edit when the answer is "technically better but I don't like
  it": move a weight, do not patch the search.
- `SPACES`    -- named search spaces, i.e. which `SwarmConfig` parameters may
  move and between which bounds.
- `TuningConfig` -- the budget and the mechanics of the search itself.

**Every band below was measured, not guessed.** The reference points come from
running `swarm.quality.measure` on the lab target (`images/jellyfish.png`, 256px,
nca_size 128) in three states:

| reading | stage-2 input | perfect copy of the target | `rooted` preset, 600 steps |
| --- | --- | --- | --- |
| `improvement` | 0.000 | 1.000 | −0.005 |
| `detail_ratio` | 0.385 | 1.000 | 0.386 |
| `gradient_alignment` | 0.875 | 1.000 | 0.875 |
| `stroke_coherence` | 0.000 | 0.496 | 0.276 |
| `coverage` | 0.000 | 0.945 | 0.141 |
| `chroma_ratio` | 0.972 | 1.000 | 0.972 |

The "perfect copy" column is not a goal -- §13 calls a run that reaches it an
expensive photocopier -- but it is the natural scale: it says what these
numbers read when the picture is *right*, so a band can be placed between
"did nothing" and "cheated" instead of at an arbitrary constant.

The `rooted` column is why this file exists at all. That preset is the current
calibrated baseline and it is measurably doing nothing: it repaints 14% of the
subject and leaves the error slightly worse than it found it.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

from config.common import get_device


# ==================== the criteria ====================

ScoreMode = Literal['higher', 'lower', 'band']


@dataclass
class Criterion:
    """
    One success criterion: a metric, a target region, and what it is worth.

    `mode` decides how the measured value becomes a sub-score in [0, 1]:

    - `higher` -- 0 at `lo`, 1 at `hi`, linear between, saturating outside.
    - `lower`  -- the mirror image: 1 at `lo`, 0 at `hi`.
    - `band`   -- 1 anywhere in [lo, hi], falling linearly to 0 over `soft`
      on each side.

    Everything is linear and continuous on purpose. A hard pass/fail gives the
    search no gradient to climb: a configuration that misses by a hair and one
    that misses by a mile would score identically, and a population-based
    search would then wander at random. Saturation at the good end is equally
    deliberate -- past the band, more is not better, which is what stops the
    tuner from optimising towards the photocopier.
    """

    key: str            # field name in the flattened metrics dict
    doc: str            # why this is a success criterion, in one line
    weight: float
    mode: ScoreMode
    lo: float
    hi: float
    soft: float = 0.0   # band only: falloff width outside the plateau

    def score(self, value: float) -> float:
        if value is None:
            return 0.0
        span = max(self.hi - self.lo, 1e-12)

        if self.mode == 'higher':
            return _clamp01((value - self.lo) / span)
        if self.mode == 'lower':
            return _clamp01((self.hi - value) / span)

        if self.lo <= value <= self.hi:
            return 1.0
        soft = max(self.soft, 1e-12)
        distance = (self.lo - value) if value < self.lo else (value - self.hi)
        return _clamp01(1.0 - distance / soft)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


# The success criteria of stage 3, grouped by the question each one answers.
# Weights sum to 1.0, so the composite score is directly readable as a
# percentage of "everything we asked for".
CRITERIA: List[Criterion] = [

    # ---------- does it improve the picture? (Evolutionary_Swarm §1, §12) ----------
    Criterion(
        key='improvement', weight=0.24, mode='higher', lo=0.0, hi=0.12,
        doc='Error reduction on the subject versus the stage-2 input. Full marks at 12%: '
            'a visible gain, well short of the photocopier §13 warns about.',
    ),
    Criterion(
        key='detail_ratio', weight=0.12, mode='band', lo=0.85, hi=1.15, soft=0.45,
        doc='High-frequency energy relative to the target. The input sits at 0.39 -- '
            'the swarm is supposed to eat exactly that deficit (§3). Above 1.15 it is '
            'adding grain, not detail.',
    ),
    Criterion(
        key='alignment_gain', weight=0.10, mode='higher', lo=-0.005, hi=0.05,
        doc='How much better the canvas gradients line up with the target than the input '
            'did. This is what separates restored texture from sprayed noise: noise '
            'raises detail_ratio and lowers this.',
    ),

    # ---------- does it look like painting? (§5.2, §5.4) ----------
    Criterion(
        key='stroke_coherence', weight=0.12, mode='higher', lo=0.25, hi=0.55,
        doc='Structure-tensor anisotropy of the pigment layer: elongated marks score high, '
            'isotropic speckle low. The target\'s own missing-detail layer reads 0.50.',
    ),
    Criterion(
        key='coverage', weight=0.08, mode='band', lo=0.50, hi=0.98, soft=0.35,
        doc='Share of the subject the swarm actually repainted. Below half and stage 3 is '
            'decoration on top of stage 2 rather than a stage.',
    ),
    Criterion(
        key='chroma_ratio', weight=0.06, mode='band', lo=0.92, hi=1.15, soft=0.25,
        doc='Mean chroma against the target. Guards §13\'s "deriva cromatica": mutation '
            'without enough pressure drifts the palette towards grey.',
    ),
    Criterion(
        key='flicker', weight=0.06, mode='band', lo=0.0002, hi=0.002, soft=0.002,
        doc='Per-step canvas change at regime. Zero is the crystallisation §16 forbids; '
            'too high is §13\'s "equilibrio rumoroso", a picture that never settles.',
    ),

    # ---------- is it actually an evolving population? (§5.7, §5.8, §13) ----------
    Criterion(
        key='population_fill', weight=0.08, mode='band', lo=0.15, hi=0.90, soft=0.15,
        doc='Settled population as a share of the cap. Pinned at the floor means the '
            'respawn is doing the work; pinned at the cap means nothing is being selected.',
    ),
    Criterion(
        key='birth_rate', weight=0.07, mode='higher', lo=0.0002, hi=0.004,
        doc='Reproductions per agent per step. Zero births is a random sprayer: no '
            'heredity, no selection, no evolution -- the whole premise gone.',
    ),
    Criterion(
        key='floor_fraction', weight=0.07, mode='lower', lo=0.0, hi=0.5,
        doc='Share of the run spent at the population floor -- §13\'s "estinzione precoce" '
            'read directly off the population curve.',
    ),
]


@dataclass
class ObjectiveConfig:
    """
    How the criteria fold into the single number the search maximises.

    `regression_gate` is the one non-negotiable: a run that leaves the picture
    worse than stage 2 handed it over is a failure whatever else it scores, so
    the whole composite is multiplied down. The multiplier is a curve rather
    than a switch, and it never quite reaches zero, for a reason found the hard
    way -- see `swarm.objective._gate`: the first configurations sit deep in
    the failing region, and a penalty that flattens them all to zero leaves the
    search nothing to climb.
    """

    criteria: List[Criterion] = field(default_factory=lambda: list(CRITERIA))
    regression_gate: bool = True
    regression_span: float = 0.02      # damage at which the gate halves the score

    # The search stops early when the best score reaches this. 1.0 is
    # unreachable by construction (it would require the photocopy), so this is
    # "good enough to go and look at", not "perfect".
    target_score: float = 0.75


# ==================== the search space ====================

Scale = Literal['log', 'linear', 'int', 'int_log']


@dataclass
class Param:
    """
    One tunable parameter: a dotted path into `SwarmConfig` and its bounds.

    `scale='log'` for anything whose useful range spans orders of magnitude --
    which is most of the metabolism, where the difference between 1 and 10 is
    the same *kind* of difference as between 100 and 1000. Sampling those
    linearly wastes almost the whole budget in the top decade.
    """
    path: str
    low: float
    high: float
    scale: Scale = 'log'

    def to_value(self, unit: float) -> Any:
        """Map a search coordinate in [0, 1] to the parameter's own units."""
        unit = _clamp01(unit)
        if self.scale in ('log', 'int_log'):
            import math
            lo = math.log(max(self.low, 1e-12))
            value = math.exp(lo + (math.log(max(self.high, 1e-12)) - lo) * unit)
        else:
            value = self.low + (self.high - self.low) * unit
        return int(round(value)) if self.scale in ('int', 'int_log') else float(value)

    def to_unit(self, value: float) -> float:
        """The inverse, for seeding the search from an existing preset."""
        if self.scale in ('log', 'int_log'):
            import math
            lo = math.log(max(self.low, 1e-12))
            hi = math.log(max(self.high, 1e-12))
            return _clamp01((math.log(max(float(value), 1e-12)) - lo) / max(hi - lo, 1e-12))
        return _clamp01((float(value) - self.low) / max(self.high - self.low, 1e-12))


# The metabolism (§5.6) and what it takes to reproduce (§5.8). This is the
# subspace the `survival` and `locality` studies were groping at by hand: the
# gain an agent gets is of order 1e-4 per step while the costs are of order
# 1e-3, so `gain_scale` has to reach into the hundreds before anybody can eat.
_ECONOMY = [
    Param('gain_scale', 1.0, 2000.0, 'log'),
    Param('cost_life', 1e-5, 0.02, 'log'),
    Param('cost_deposit', 1e-4, 0.2, 'log'),
    Param('reproduction_threshold', 0.2, 4.0, 'log'),
    Param('initial_energy', 0.2, 2.0, 'log'),
    Param('energy_cap', 1.0, 10.0, 'log'),
]

# The mark itself (§5.4) and how far a lineage roams (§5.3, §2.1).
_STROKE = [
    Param('deposit_alpha', 0.01, 0.5, 'log'),
    Param('brush_radius', 1.0, 4.0, 'linear'),
    Param('speed', 0.05, 2.0, 'log'),
    Param('birth_jitter', 0.5, 16.0, 'log'),
    Param('mutation_sigma', 0.002, 0.12, 'log'),
    Param('tolerance', 0.0, 0.05, 'linear'),
]

# How the picture forgets (§5.9) and how long the trails last (§5.10).
_PIGMENT = [
    Param('decay_min', 0.0002, 0.02, 'log'),
    Param('decay_max', 0.005, 0.3, 'log'),
    Param('decay_e_ref', 0.03, 0.4, 'log'),
    Param('evaporation', 0.85, 0.995, 'linear'),
]

# Where the agents go looking (§5.1, §5.2, §5.2b).
_PERCEPTION = [
    Param('sensor_angle', 0.15, 1.2, 'linear'),
    Param('sensor_distance', 2.0, 16.0, 'log'),
    Param('rotation_angle', 0.1, 1.5, 'linear'),
    Param('angle_noise', 0.0, 0.6, 'linear'),
    Param('w_nutrient', 0.2, 4.0, 'log'),
    Param('w_crowd', 0.0, 2.0, 'linear'),
    Param('tumble_angle', 0.2, 2.0, 'linear'),
]

# `guidance` is deliberately absent from every space. It is the cheating knob
# (§2.2) and an optimiser handed it would find it immediately -- it is the
# single fastest way to lower the error, which is exactly why the design keeps
# it at zero. The tuner asserts it stayed there.
SPACES: Dict[str, List[Param]] = {
    'economy': _ECONOMY,
    'stroke': _STROKE,
    'pigment': _PIGMENT,
    'perception': _PERCEPTION,
    'core': _ECONOMY + _STROKE + [Param('decay_max', 0.005, 0.3, 'log')],
    'full': _ECONOMY + _STROKE + _PIGMENT + _PERCEPTION,
}


# ==================== the search itself ====================

@dataclass
class TuningConfig:
    """
    Budget and mechanics for `swarm.tuning`.

    The search is a (μ+λ) evolution strategy with an adaptive step size, which
    is a deliberate choice of the *simplest thing that works here* rather than
    a fashionable one: the objective is stochastic-ish, non-differentiable and
    mildly multi-modal, evaluations cost seconds, and the budget is hundreds --
    a regime where an ES beats both grid search (which cannot afford 12
    dimensions) and Bayesian optimisation (whose surrogate needs more
    structure than this objective offers). It also has the pleasant property of
    being the same algorithm as the thing it is tuning, one level up.
    """

    space: str = 'core'
    image: str = 'images/jellyfish.png'
    output_root: str = 'outputs/swarm/tuning'

    # --- fidelity of each evaluation ---
    work_size: int = 256
    population_cap: int = 4096
    steps: int = 800               # per evaluation during the search
    final_steps: int = 2400        # the winner is re-run this long before it is believed
    seeds: List[int] = field(default_factory=lambda: [0])
    final_seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    metrics_stride: int = 4        # curve resolution the search does not need at full rate

    # --- the search ---
    initial_random: int = 32       # random probes before the ES starts
    generations: int = 40
    elites: int = 6                # μ
    children: int = 12             # λ
    sigma_init: float = 0.25       # step size in the normalised [0, 1] space
    sigma_min: float = 0.02
    sigma_max: float = 0.5
    sigma_grow: float = 1.15       # applied when a generation improves the best
    sigma_shrink: float = 0.85     # and when it does not
    patience: int = 10             # generations without improvement before stopping

    # `seed_from` names a preset whose values become one of the initial probes,
    # so a search never starts worse-informed than the last calibration.
    seed_from: str = 'rooted'

    # --- execution ---
    workers: int = 6               # evaluation processes; 0 or 1 runs in-process
    device: str = field(default_factory=get_device)
    threads_per_worker: int = 2

    # --- what to look at while it runs ---
    animate_every: int = 5         # generations between animations of the current best
    animation_steps: int = 1200
    animation_fps: int = 30
    animation_capture_stride: int = 4   # simulation steps per animation frame
    thumb_size: int = 192

    def evaluations_budget(self) -> int:
        return self.initial_random + self.generations * self.children
