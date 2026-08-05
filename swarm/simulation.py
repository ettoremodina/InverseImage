"""
The step loop (§5) and the climate (§8). Everything else in `swarm/` is a
mechanism; `Simulation` is the only place that calls them in the order the
design doc specifies.
"""

from typing import List

import torch

from config.swarm_config import SwarmConfig
from swarm.agents import AgentState, perceive, rotate, advance, deposit, spawn_agents
from swarm.fields import Fields
from swarm.metrics import StepMetrics, collect
from swarm.selection import consume, apply_costs, kill_and_repopulate, reproduce
from swarm.colorspace import oklab_to_srgb, srgb_to_oklab

__all__ = ['Simulation', 'StepMetrics']


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
        self._pending_births = 0
        self._pending_deaths = 0

        # The error of the untouched starting canvas. Everything the swarm does
        # is judged against this: it is what stage 2 already achieved for free.
        self.baseline_error = float(self.fields.compute_error(raw=True).mean().item())
        self.population_floor = int(config.min_population_fraction * config.population_cap)

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
        """
        `(cost_life, gain_scale, mutation_sigma)` for this step. The curves are
        multipliers on the config scalars -- see `SwarmConfig.climate_at`.
        """
        return self.config.climate_at(self._climate_t())

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

        deaths = kill_and_repopulate(self.agents, self.fields, cfg, self.rng, mutation_sigma)
        births = reproduce(self.agents, self.fields, cfg, mutation_sigma, self.rng)

        self.fields.age_pigment(error_after)
        self.fields.breathe()

        self.step_count += 1
        self._record_metrics(gain_i, births, deaths)

    # ------------------------------------------------------------ diagnostics (§12)

    def _record_metrics(self, gain_i: torch.Tensor, births: int, deaths: int):
        """
        Append a §12 row, honouring `metrics_stride`. The last step of a run is
        always recorded, so a summary never has to interpolate its final value.

        Births and deaths accumulate across skipped steps rather than being
        sampled: they are counts, and a sampled count is just a wrong count.
        """
        self._pending_births += births
        self._pending_deaths += deaths

        stride = max(1, self.config.metrics_stride)
        is_last = self.step_count >= self.config.sim_steps
        if self.step_count % stride and not is_last:
            return

        self.history.append(collect(self, gain_i, self._pending_births, self._pending_deaths))
        self._pending_births = self._pending_deaths = 0

    # ------------------------------------------------------------ rendering

    def _render_srgb(self, field_oklab: torch.Tensor) -> torch.Tensor:
        """(H, W, 3) uint8 sRGB of any OKLab field."""
        rgb = oklab_to_srgb(field_oklab)
        return (rgb * 255.0 + 0.5).clamp(0, 255).byte()

    def render_canvas_srgb(self) -> torch.Tensor:
        """(H, W, 3) uint8 sRGB of the current canvas."""
        return self._render_srgb(self.fields.canvas)

    def render_base_srgb(self) -> torch.Tensor:
        """The starting canvas -- what stage 2 handed over, the baseline to beat."""
        return self._render_srgb(self.fields.base)

    def render_target_srgb(self) -> torch.Tensor:
        """Ground truth. For side-by-side artefacts only; no mechanism may read this."""
        return self._render_srgb(self.fields.target)
