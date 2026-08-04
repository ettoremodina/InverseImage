"""
Configuration for the evolutionary swarm (stage 3, Evolutionary_Swarm.md).

Defaults are the *lab* scale (§11.3): 256px, 4096 agents, fast enough to
iterate on. Production scale (512px, 20-50k agents) is a config override, not
a different code path.

Every `climate.*` field is a `ClimateCurve` -- `start == end` means "constant"
(§8, §16), so switching a parameter between a fixed value and an annealed one
is a two-number edit, not a code change.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ClimateCurve:
    """One value, eased over the course of the simulation. start == end == constant."""
    start: float
    end: float
    easing: str = 'linear'

    def value_at(self, t: float) -> float:
        from rendering.easing import apply_easing
        return self.start + (self.end - self.start) * apply_easing(self.easing, t)


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
    climate_cost_life: ClimateCurve = field(default_factory=lambda: ClimateCurve(0.01, 0.01))
    climate_gain_scale: ClimateCurve = field(default_factory=lambda: ClimateCurve(1.0, 1.0))
    climate_mutation_sigma: ClimateCurve = field(default_factory=lambda: ClimateCurve(0.03, 0.01, 'ease_in_out'))

    # ==================== timing (§8) ====================
    sim_steps: int = 800
    step_stride: int = 1               # write a frame every N steps
    fps: int = 30
