"""
Configuration for the evolutionary swarm (stage 3, Evolutionary_Swarm.md).

Defaults are the *lab* scale (§11.3): 256px, 4096 agents, fast enough to
iterate on. Production scale (512px, 20-50k agents) is a config override, not
a different code path.

Every `climate.*` field is a `ClimateCurve` -- `start == end` means "constant"
(§8, §16), so switching a parameter between a fixed value and an annealed one
is a two-number edit, not a code change.

**A climate curve is a multiplier on its scalar, not a replacement for it.**
`cost_life` is the value; `climate_cost_life` is the schedule applied to it, and
`ClimateCurve(1.0, 1.0)` means "no climate". Before this was made explicit the
curves carried absolute values, which silently made `cost_life`, `gain_scale`
and `mutation_sigma` dead knobs -- the lab's trackbars and sweeps moved them and
nothing happened, because `Simulation` only ever read the curve. Multiplying is
what keeps both ends live: tune the scalar, shape the arc.
"""

import copy
from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass
from typing import Any, Dict, Literal, Optional, get_args, get_origin


@dataclass
class ClimateCurve:
    """
    A multiplier on its scalar parameter, eased over the run (§8).

    `start == end` is a constant, and `ClimateCurve(1.0, 1.0)` -- the default
    everywhere -- means the scalar is used as written.
    """
    start: float
    end: float
    easing: str = 'linear'

    def value_at(self, t: float) -> float:
        from rendering.easing import apply_easing
        return self.start + (self.end - self.start) * apply_easing(self.easing, t)

    @property
    def is_constant(self) -> bool:
        return self.start == self.end


RotationMode = Literal['proportional', 'classic']
BoundaryMode = Literal['reflect', 'clamp_random']
ErrorMetric = Literal['l2', 'l2_sq']
RepopulationMode = Literal['nutrient', 'survivor']


@dataclass
class SwarmConfig:
    # ==================== scale & device ====================
    work_size: int = 256
    population_cap: int = 4096
    min_population_fraction: float = 0.05   # floor, as a fraction of population_cap (§5.7)
    device: str = 'cpu'
    seed: int = 0

    # ==================== 5.1 perception ====================
    sensor_angle: float = 0.6          # radians, offset of the two side sensors
    sensor_distance: float = 6.0       # pixels, at work_size scale
    w_nutrient: float = 1.0            # weight of nutrient in the attractiveness field
    w_crowd: float = 0.0               # repulsion from local agent density; 0 = off (§5.1)
    guidance: float = 0.0              # cheating knob, default 0 -- never raise for real renders (§2.2)

    # ==================== 5.2 rotation ====================
    rotation_mode: RotationMode = 'proportional'
    rotation_angle: float = 0.5        # radians, max turn per step
    angle_noise: float = 0.15          # radians, std of the always-on heading jitter

    # ==================== 5.2b propriocezione (run-and-tumble) ====================
    tumble_enabled: bool = True
    tumble_threshold: float = 0.0      # tumble when Δgain falls below this
    tumble_angle: float = 1.2          # radians, half-width of the tumble turn

    # ==================== 5.3 movement ====================
    speed: float = 1.0                 # pixels/step at work_size scale
    boundary_mode: BoundaryMode = 'reflect'
    nutrient_threshold: float = 0.1    # below this, an agent is "off the tissue"
    starvation_cost: float = 0.02      # extra cost/step while off the tissue

    # ==================== 5.4 deposit ====================
    brush_radius: float = 1.5          # pixels; kernel support, gaussian sigma = radius/2
    deposit_alpha: float = 0.15
    deposit_pheromone: float = 1.0

    # ==================== 5.5 consumo ====================
    fitness_scale: int = 1             # error measured every N pixels; 1 = faithful (§16)
    tolerance: float = 0.02            # OKLab distance below which nothing feeds
    error_metric: ErrorMetric = 'l2'
    gain_scale: float = 1.0

    # ==================== 5.6 costs ====================
    # Tuned so cost_deposit * (typical mass/step, ~1.0 at the defaults above)
    # sits near the typical achievable gain/step (empirically ~0.005-0.05 at
    # lab scale) instead of dwarfing it -- see the lab (§11) before trusting
    # these numbers on a different brush_radius/deposit_alpha.
    cost_deposit: float = 0.02         # multiplies the pigment mass laid down
    cost_life: float = 0.005           # fixed cost per step
    energy_cap: float = 3.0
    max_age: Optional[int] = None      # None = no hard senescence cutoff

    # ==================== 5.7 death & repopulation ====================
    initial_energy: float = 1.0
    repopulation_mode: RepopulationMode = 'nutrient'  # never 'error' -- see §5.7

    # ==================== 5.8 reproduction ====================
    reproduction_threshold: float = 1.5
    mutation_sigma: float = 0.02       # OKLab units
    birth_jitter: float = 1.5          # pixels -- also the spatial mutation rate (§2.1)
    birth_dir_sigma: float = 0.3       # radians

    # ==================== 5.9 pigment aging ====================
    decay_min: float = 0.002           # λ_min: never zero -- see §5.9, no crystallisation
    decay_max: float = 0.05            # λ_max: how fast wrong pigment fades
    decay_e_ref: float = 0.15          # error scale at which λ saturates towards λ_max

    # ==================== 5.10 field respiration ====================
    evaporation: float = 0.95
    pheromone_diffuse: bool = True

    # ==================== §8 climate ====================
    # Multipliers on the scalars above, not replacements -- see the module
    # docstring. (1.0, 1.0) is "no climate".
    climate_cost_life: ClimateCurve = field(default_factory=lambda: ClimateCurve(1.0, 1.0))
    climate_gain_scale: ClimateCurve = field(default_factory=lambda: ClimateCurve(1.0, 1.0))
    # 1.5 -> 0.5 on mutation_sigma: bold strokes first, retouching last (§8).
    climate_mutation_sigma: ClimateCurve = field(default_factory=lambda: ClimateCurve(1.5, 0.5, 'ease_in_out'))

    # ==================== timing (§8) ====================
    sim_steps: int = 800
    step_stride: int = 1               # write a frame every N steps
    fps: int = 30

    # ==================== diagnostics (§12) ====================
    # Each metrics row costs ~6 device syncs, which at lab scale is comparable
    # to the step itself; stride > 1 buys sweep throughput at the cost of curve
    # resolution. The final step is always recorded regardless.
    metrics_stride: int = 1

    def climate_at(self, t: float):
        """
        The three climate-modulated values at normalised time `t` (§8),
        as `(cost_life, gain_scale, mutation_sigma)`.
        """
        return (
            self.cost_life * self.climate_cost_life.value_at(t),
            self.gain_scale * self.climate_gain_scale.value_at(t),
            self.mutation_sigma * self.climate_mutation_sigma.value_at(t),
        )


@dataclass
class SwarmStageConfig:
    """
    Wiring for stage 3 inside the timeline (PLAN §3.3) -- everything about
    *plugging the swarm in* that is not a property of the swarm itself.

    The seam the design doc predicted is one method: instead of a static
    `nutrient`, the simulation receives the NCA alpha of the current frame, so
    the swarm can only live where stage 2 has actually grown tissue. Nothing
    else about `Simulation` changes between the lab and production.
    """

    enabled: bool = True

    # `preset` names an entry in config/lab_config.PRESETS. Production runs the
    # configuration the lab calibrated, by name, instead of a second copy of the
    # numbers that then drifts out of sync.
    preset: str = 'rooted'

    # The simulation runs at `work_size` and its layer is scaled to the canvas.
    # Running at full supersampled canvas resolution would multiply the agent
    # count by four for detail the video resolve throws away anyway.
    work_size: int = 512

    # §8: simulated length and video length are independent. 200 frames of
    # window at 12 steps/frame is 2400 steps of evolution.
    steps_per_frame: int = 12
    warmup_steps: int = 60          # steps run before the first visible frame

    # How much of the window the layer takes to reach full opacity. Without it
    # the swarm pops in on the frame its window opens.
    blend_in: float = 0.2
    blend_easing: str = 'ease_in_out'

    nutrient_blur: float = 2.0      # px, softens the tissue edge into a gradient
    population_scale: bool = True   # scale population_cap with work_size vs lab 256


# ==================== overrides: the lab's way in ====================
#
# Sweeps, JSON configs and presets all funnel through `apply_overrides`. The
# point of routing them through one typed setter is that a misspelled parameter
# raises instead of silently creating a new attribute nobody reads -- the
# failure mode that makes a sweep look like it ran and produced no effect.


def _coerce(value: Any, annotation: Any, path: str) -> Any:
    """Cast a JSON/CLI value to what the dataclass field declares."""
    if get_origin(annotation) is Literal:
        allowed = get_args(annotation)
        if value not in allowed:
            raise ValueError(f'{path}: expected one of {allowed}, got {value!r}')
        return value

    # Optional[X] -- 'none'/None clears, anything else follows X.
    if get_origin(annotation) is Optional or type(None) in get_args(annotation):
        if value is None or (isinstance(value, str) and value.lower() in ('none', 'null', '')):
            return None
        inner = [a for a in get_args(annotation) if a is not type(None)]
        return _coerce(value, inner[0], path) if inner else value

    if annotation is bool:
        if isinstance(value, str):
            if value.lower() not in ('true', 'false', '1', '0', 'yes', 'no'):
                raise ValueError(f'{path}: expected a boolean, got {value!r}')
            return value.lower() in ('true', '1', 'yes')
        return bool(value)

    if annotation is ClimateCurve:
        if isinstance(value, ClimateCurve):
            return copy.deepcopy(value)
        if isinstance(value, dict):
            return ClimateCurve(**value)
        if isinstance(value, (list, tuple)):
            return ClimateCurve(*value)
        raise ValueError(f'{path}: expected a ClimateCurve, dict or [start, end, easing], got {value!r}')

    if annotation in (int, float, str):
        return annotation(value)

    return value


def known_params() -> Dict[str, Any]:
    """Every settable dotted path -> its declared type, for validation and help."""
    paths = {}
    for f in dataclass_fields(SwarmConfig):
        if f.type is ClimateCurve or f.type == 'ClimateCurve':
            paths[f.name] = ClimateCurve
            for sub in dataclass_fields(ClimateCurve):
                paths[f'{f.name}.{sub.name}'] = sub.type
        else:
            paths[f.name] = f.type
    return paths


def set_param(config: 'SwarmConfig', path: str, value: Any) -> None:
    """
    Set one parameter by dotted path, in place, with type coercion.

    Dotted paths reach into the climate curves, so a sweep can vary the *arc*
    (`climate_cost_life.end`) and not only the level (`cost_life`) -- the A/B
    §8 asks for, without a bespoke code path for it.
    """
    valid = known_params()
    if path not in valid:
        raise KeyError(f'unknown swarm parameter {path!r}; '
                       f'see config/swarm_config.py for the {len(valid)} valid paths')

    annotation = valid[path]
    if isinstance(annotation, str):     # from __future__ annotations / string types
        annotation = {'int': int, 'float': float, 'bool': bool, 'str': str}.get(annotation, annotation)

    head, _, tail = path.partition('.')
    if tail:
        curve = getattr(config, head)
        setattr(curve, tail, _coerce(value, annotation, path))
    else:
        setattr(config, head, _coerce(value, annotation, path))


def apply_overrides(config: 'SwarmConfig', overrides: Dict[str, Any]) -> 'SwarmConfig':
    """Deep-copy `config` and apply a flat `{dotted path: value}` mapping to the copy."""
    out = copy.deepcopy(config)
    for path, value in (overrides or {}).items():
        set_param(out, path, value)
    return out


def config_to_dict(config: 'SwarmConfig') -> Dict[str, Any]:
    """JSON-serialisable dump of a config, climate curves included."""
    out = {}
    for f in dataclass_fields(config):
        value = getattr(config, f.name)
        out[f.name] = {k: getattr(value, k) for k in ('start', 'end', 'easing')} \
            if is_dataclass(value) else value
    return out
