"""
Diagnostics (Evolutionary_Swarm.md §12) -- the swarm's learning curve.

Split out of `simulation.py` because the lab (§11) needs to *summarise* a run,
not just watch it: a per-step row is what you plot, a `RunSummary` is what you
sort a sweep by.

The two summary fields worth knowing about before reading the rest:

- **`improvement`** -- `1 - final_error / baseline_error`, where the baseline is
  the error of the untouched starting canvas (the NCA output, or its §11.1
  surrogate in the lab). This is the only number that answers "did the swarm
  help at all". A negative value means the run made the picture *worse* than
  what stage 2 handed it, which is a failure no amount of pretty texture
  redeems.
- **`floor_fraction`** -- how much of the run was spent pinned at the
  `min_population_fraction` floor. Anything near 1.0 means the population is
  being kept alive by the respawn floor rather than by eating, i.e. §13's
  *estinzione precoce*: selection never gets a chance to act because nobody
  lives long enough to reproduce.
"""

import csv
from dataclasses import dataclass, fields as dataclass_fields, asdict
from typing import List

import torch


@dataclass
class StepMetrics:
    """One row of the §12 diagnostics."""

    step: int
    mean_error: float          # raw ‖canvas - target‖, unclamped by tolerance
    population: int
    gene_diversity: float      # summed per-channel variance of alive genes
    total_energy: float
    mean_gain: float

    # --- added for the lab: the readings that make a failed run legible ---
    mean_energy: float         # total_energy / population; distinguishes "few rich" from "many poor"
    positive_gain_fraction: float   # share of alive agents that ate this step
    reward_coverage: float     # share of pixels with e_eff > 0 -- how much food exists at all
    births: int                # reproductions (§5.8)
    deaths: int                # energy/age deaths (§5.7), before the floor refills

    # Mean per-step ‖Δcanvas‖ on the tissue since the previous recorded row.
    # This is the direct reading of §5.9's λ_min trade-off -- a picture that has
    # settled has a small non-zero value, one that seethes has a large one, and
    # one that has crystallised (which the design forbids) has zero.
    canvas_delta: float = 0.0


METRIC_FIELDS = tuple(f.name for f in dataclass_fields(StepMetrics))


@dataclass
class RunSummary:
    """One row per run in a sweep -- the whole of a simulation, in numbers."""

    baseline_error: float      # error of the starting canvas vs target, before step 1
    final_error: float
    best_error: float
    improvement: float         # 1 - final/baseline; < 0 means the swarm hurt (see module docstring)
    best_improvement: float

    final_population: int
    mean_population: float     # over the settled tail of the run
    floor_fraction: float      # share of steps pinned at the population floor
    extinct: bool              # floor_fraction > 0.9 -- §13 "estinzione precoce"

    mean_energy: float
    mean_gain: float
    positive_gain_fraction: float
    gene_diversity: float
    total_births: int
    total_deaths: int

    steps: int
    wall_time: float
    steps_per_sec: float


SUMMARY_FIELDS = tuple(f.name for f in dataclass_fields(RunSummary))


def collect(sim, gain_i: torch.Tensor, births: int, deaths: int,
            canvas_delta: float = 0.0) -> StepMetrics:
    """
    Read one row off a live `Simulation`.

    Every `.item()` here is a device sync, which is why `SwarmConfig
    .metrics_stride` exists: at lab scale the step itself is small enough that
    these reads, not the simulation, dominate a headless sweep.
    """
    agents = sim.agents
    alive = agents.alive
    n_alive = int(alive.sum().item())

    raw_error = sim.fields.compute_error(raw=True)
    effective_error = (raw_error - sim.config.tolerance).clamp(min=0.0)

    if n_alive:
        energy_alive = agents.energy[alive]
        total_energy = float(energy_alive.sum().item())
        mean_energy = total_energy / n_alive
        mean_gain = float(gain_i[alive].mean().item())
        positive = float((gain_i[alive] > 0).float().mean().item())
        diversity = float(agents.gene[alive].var(dim=0).sum().item()) if n_alive > 1 else 0.0
    else:
        total_energy = mean_energy = mean_gain = positive = diversity = 0.0

    return StepMetrics(
        step=sim.step_count,
        mean_error=float(raw_error.mean().item()),
        population=n_alive,
        gene_diversity=diversity,
        total_energy=total_energy,
        mean_gain=mean_gain,
        mean_energy=mean_energy,
        positive_gain_fraction=positive,
        reward_coverage=float((effective_error > 0).float().mean().item()),
        births=births,
        deaths=deaths,
        canvas_delta=canvas_delta,
    )


def summarize(history: List[StepMetrics], baseline_error: float, population_floor: int,
              wall_time: float, tail_fraction: float = 0.25) -> RunSummary:
    """
    Fold a run's history into one comparable row.

    `tail_fraction` is the share of the run treated as "settled": the early
    steps are a transient by design (§7, the narrative arc), so averaging over
    the whole run would mix the boom into the verdict.
    """
    if not history:
        raise ValueError('cannot summarize an empty history')

    tail_start = max(0, int(len(history) * (1.0 - tail_fraction)))
    tail = history[tail_start:] or history[-1:]

    errors = [m.mean_error for m in history]
    final_error = history[-1].mean_error
    best_error = min(errors)
    denominator = max(baseline_error, 1e-9)

    at_floor = sum(1 for m in history if m.population <= population_floor)
    floor_fraction = at_floor / len(history)

    return RunSummary(
        baseline_error=baseline_error,
        final_error=final_error,
        best_error=best_error,
        improvement=1.0 - final_error / denominator,
        best_improvement=1.0 - best_error / denominator,
        final_population=history[-1].population,
        mean_population=sum(m.population for m in tail) / len(tail),
        floor_fraction=floor_fraction,
        extinct=floor_fraction > 0.9,
        mean_energy=sum(m.mean_energy for m in tail) / len(tail),
        mean_gain=sum(m.mean_gain for m in tail) / len(tail),
        positive_gain_fraction=sum(m.positive_gain_fraction for m in tail) / len(tail),
        gene_diversity=history[-1].gene_diversity,
        total_births=sum(m.births for m in history),
        total_deaths=sum(m.deaths for m in history),
        steps=history[-1].step,
        wall_time=wall_time,
        steps_per_sec=history[-1].step / max(wall_time, 1e-9),
    )


def write_history_csv(history: List[StepMetrics], path) -> None:
    """The full per-step history -- the input to the §12 curves."""
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METRIC_FIELDS))
        writer.writeheader()
        writer.writerows(asdict(m) for m in history)
