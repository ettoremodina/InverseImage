"""
Consumo (§5.5), costs (§5.6), death and repopulation (§5.7), reproduction
(§5.8).

This is the only module that converts `error` into energy -- the single
point where the target is allowed to have a veto (Evolutionary_Swarm.md §1).
Nothing here ever uses `error` to decide *where* an agent goes; that
distinction (§5.7) is what keeps repopulation from becoming cheating in
disguise.
"""

import torch

from config.swarm_config import SwarmConfig
from swarm.agents import AgentState, Splat, spawn_agents
from swarm.colorspace import clamp_gamut


# ==================== 5.5 consumo ====================

def consume(agents: AgentState, splat: Splat, error_before: torch.Tensor,
            error_after: torch.Tensor) -> torch.Tensor:
    """
    Attribute the drop in error to whoever caused it. Conservative --
    `sum(gain_i) == sum(gain(p))` -- and split among agents on a pixel in
    proportion to how much of that pixel's paint is theirs (`a_i(p) / W(p)`).
    The competition for a pixel is not programmed anywhere; it falls out of
    this bookkeeping, because two agents sharing a pixel split its ration.

    Also rolls the propriocezione history used by `agents.rotate` (§5.2b).
    """
    gain_field = (error_before - error_after).reshape(-1)
    per_pixel_gain = gain_field[splat.flat_idx] / splat.total_w[splat.flat_idx].clamp(min=1e-8)
    entry_gain = splat.a * per_pixel_gain

    gain_i = torch.zeros(agents.n, device=agents.device).scatter_add_(0, splat.agent_idx, entry_gain)

    agents.gain_prev = agents.gain_last
    agents.gain_last = gain_i
    return gain_i


# ==================== 5.6 costs ====================

def apply_costs(agents: AgentState, splat: Splat, gain_i: torch.Tensor, off_tissue: torch.Tensor,
                 config: SwarmConfig, gain_scale: float, cost_life: float):
    """
    Turn this step's gain into an energy delta. The deposit cost is
    proportional to the pigment mass actually laid down (`sum_p a_i(p)`), not
    a flat per-step fee -- that is what makes repainting an already-correct
    pixel a losing move (gain ~= 0, cost > 0) instead of free (§13,
    "degenerazione golosa").
    """
    deposit_mass = torch.zeros(agents.n, device=agents.device).scatter_add_(0, splat.agent_idx, splat.a)

    delta = gain_scale * gain_i \
        - config.cost_deposit * deposit_mass \
        - cost_life \
        - config.starvation_cost * off_tissue.float()

    agents.energy = (agents.energy + delta).clamp(max=config.energy_cap)
    agents.age = agents.age + 1


# ==================== 5.7 death & repopulation ====================

def kill_and_repopulate(agents: AgentState, fields, config: SwarmConfig, rng: torch.Generator,
                         mutation_sigma: float):
    """
    Agents at negative energy, or past `max_age` when set, die. If the
    population then falls below the floor, empty slots are refilled from
    `nutrient` -- never from `error`, which would be exactly the cheating the
    target's veto-only rule forbids (§5.7).
    """
    dead = agents.alive & (agents.energy < 0)
    if config.max_age is not None:
        dead = dead | (agents.alive & (agents.age > config.max_age))
    agents.alive = agents.alive & ~dead

    min_population = int(config.min_population_fraction * config.population_cap)
    n_alive = agents.n_alive
    if n_alive < min_population:
        free = (~agents.alive).nonzero(as_tuple=True)[0]
        need = min(min_population - n_alive, int(free.shape[0]))
        if need > 0:
            spawn_agents(agents, fields, config, free[:need], rng, gene_source=fields.canvas,
                         mutation_sigma=mutation_sigma)


# ==================== 5.8 reproduction ====================

def reproduce(agents: AgentState, fields, config: SwarmConfig, mutation_sigma: float,
              rng: torch.Generator):
    """
    Agents above `reproduction_threshold` split their energy with a mutated
    child (§5.8): fission by halving, so reproduction is zero-sum on energy
    and a parent that reproduces too often weakens itself. Free slots are
    assigned to eligible parents in random order, not by highest energy --
    the threshold already filters for fitness, so shuffling keeps exploration
    open instead of collapsing onto the fittest few.
    """
    device = agents.device
    eligible = (agents.alive & (agents.energy > config.reproduction_threshold)).nonzero(as_tuple=True)[0]
    free = (~agents.alive).nonzero(as_tuple=True)[0]

    n = min(int(eligible.shape[0]), int(free.shape[0]))
    if n == 0:
        return

    order = torch.randperm(int(eligible.shape[0]), generator=rng, device=device)[:n]
    parents = eligible[order]
    children = free[:n]

    agents.energy[parents] = agents.energy[parents] / 2.0

    jitter = (torch.rand(n, 2, device=device, generator=rng) - 0.5) * 2 * config.birth_jitter
    child_pos = agents.pos[parents] + jitter
    child_pos[:, 0] = child_pos[:, 0].clamp(0, fields.width - 1)
    child_pos[:, 1] = child_pos[:, 1].clamp(0, fields.height - 1)

    mutation = torch.randn(n, 3, device=device, generator=rng) * mutation_sigma

    agents.pos[children] = child_pos
    agents.dir[children] = agents.dir[parents] + torch.randn(n, device=device, generator=rng) * config.birth_dir_sigma
    agents.gene[children] = clamp_gamut(agents.gene[parents] + mutation)
    agents.energy[children] = agents.energy[parents]
    agents.age[children] = 0
    agents.alive[children] = True
    agents.gain_last[children] = 0.0
    agents.gain_prev[children] = 0.0
