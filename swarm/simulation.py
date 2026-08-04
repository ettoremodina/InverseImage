"""
The step loop (§5) and the climate (§8). Everything else in `swarm/` is a
mechanism; `Simulation` is the only place that calls them in the order the
design doc specifies.
"""

from dataclasses import dataclass
from typing import List

import torch

from config.swarm_config import SwarmConfig
from swarm.agents import AgentState, perceive, rotate, advance, deposit, spawn_agents
from swarm.fields import Fields
from swarm.selection import consume, apply_costs, kill_and_repopulate, reproduce
from swarm.colorspace import oklab_to_srgb, srgb_to_oklab


@dataclass
class StepMetrics:
    """One row of the §12 diagnostics."""
    step: int
    mean_error: float
    population: int
    gene_diversity: float
    total_energy: float
    mean_gain: float


class Simulation:
    """
    Owns `AgentState` + `Fields` and drives them through the per-step
    pipeline (§5.1-5.10). Reusable by both the lab (§11) and the eventual
    timeline integration (PLAN §3.3) -- the only thing that differs between
    the two is who calls `fields.breathe(nutrient=...)` and with what.
    """

    def __init__(self, config: SwarmConfig, target_oklab: torch.Tensor, base_oklab: torch.Tensor,
                 nutrient: torch.Tensor):
        self.config = config
        self.device = torch.device(config.device)
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(config.seed)

        self.fields = Fields(target_oklab, base_oklab, nutrient, config, self.device)
        self.agents = AgentState(config, self.device)

        all_slots = torch.arange(config.population_cap, device=self.device)
        spawn_agents(self.agents, self.fields, config, all_slots, self.rng, gene_source=self.fields.base)

        self.step_count = 0
        self.history: List[StepMetrics] = []

    @classmethod
    def from_srgb(cls, config: SwarmConfig, target_srgb: torch.Tensor, base_srgb: torch.Tensor,
                  nutrient: torch.Tensor) -> 'Simulation':
        """Convenience constructor taking sRGB images in [0, 1] instead of OKLab."""
        device = torch.device(config.device)
        target_oklab = srgb_to_oklab(target_srgb.to(device))
        base_oklab = srgb_to_oklab(base_srgb.to(device))
        return cls(config, target_oklab, base_oklab, nutrient.to(device))

    # ------------------------------------------------------------ climate (§8)

    def _climate_t(self) -> float:
        return min(1.0, self.step_count / max(1, self.config.sim_steps))

    def _climate_values(self):
        t = self._climate_t()
        cfg = self.config
        return (
            cfg.climate_cost_life.value_at(t),
            cfg.climate_gain_scale.value_at(t),
            cfg.climate_mutation_sigma.value_at(t),
        )

    # ------------------------------------------------------------ one step (§5)

    def step(self):
        cfg = self.config
        cost_life, gain_scale, mutation_sigma = self._climate_values()

        # error_before doubles as the guidance signal (§5.1) and the "before"
        # snapshot for attribution (§5.5) -- one computation, two uses.
        error_before = self.fields.compute_error()

        s_l, s_c, s_r = perceive(self.agents, self.fields, cfg,
                                  error=error_before if cfg.guidance > 0 else None)
        rotate(self.agents, s_l, s_c, s_r, cfg, self.rng)
        pos_prev, off_tissue = advance(self.agents, self.fields, cfg, self.rng)

        splat = deposit(self.agents, self.fields, pos_prev, cfg)

        error_after = self.fields.compute_error()
        gain_i = consume(self.agents, splat, error_before, error_after)
        apply_costs(self.agents, splat, gain_i, off_tissue, cfg, gain_scale, cost_life)

        kill_and_repopulate(self.agents, self.fields, cfg, self.rng, mutation_sigma)
        reproduce(self.agents, self.fields, cfg, mutation_sigma, self.rng)

        self.fields.age_pigment(error_after)
        self.fields.breathe()

        self.step_count += 1
        self._record_metrics(gain_i)

    # ------------------------------------------------------------ diagnostics (§12)

    def _record_metrics(self, gain_i: torch.Tensor):
        alive = self.agents.alive
        n_alive = int(alive.sum().item())

        mean_error = float(self.fields.compute_error(raw=True).mean().item())
        total_energy = float(self.agents.energy[alive].sum().item()) if n_alive else 0.0
        mean_gain = float(gain_i[alive].mean().item()) if n_alive else 0.0
        gene_diversity = float(self.agents.gene[alive].var(dim=0).sum().item()) if n_alive > 1 else 0.0

        self.history.append(StepMetrics(
            step=self.step_count, mean_error=mean_error, population=n_alive,
            gene_diversity=gene_diversity, total_energy=total_energy, mean_gain=mean_gain,
        ))

    # ------------------------------------------------------------ rendering

    def render_canvas_srgb(self) -> torch.Tensor:
        """(H, W, 3) uint8 sRGB of the current canvas."""
        rgb = oklab_to_srgb(self.fields.canvas)
        return (rgb * 255.0 + 0.5).clamp(0, 255).byte()
