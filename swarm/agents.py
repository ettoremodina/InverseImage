"""
Agent state, perception (§5.1), rotation incl. propriocezione (§5.2, §5.2b),
movement (§5.3) and deposit (§5.4).

Two invariants hold throughout this module, and every function here is
written to preserve them:

- **Order-independence.** Every write is an accumulate-then-resolve
  (`scatter_add_` / `index_add_`), never a sequential per-agent write. Two
  agents landing on the same pixel must produce the same result regardless of
  GPU scheduling (Evolutionary_Swarm.md §5, the transversal constraint).
- **The target never appears.** Nothing in this file reads `fields.target`.
  The one narrow exception -- `guidance`, wired through `perceive` -- is
  opt-in, defaults to 0, and is documented in §2.2 as the line between
  legitimate propriocezione and cheating.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from config.swarm_config import SwarmConfig
from swarm.colorspace import clamp_gamut


# ==================== state ====================

class AgentState:
    """
    Per-agent tensors, all shape (N,) or (N, k), N = population_cap.

    Dead slots (`alive=False`) are never cleared -- every consumer masks by
    `alive` instead. Positions of dead agents are meaningless and must not be
    trusted; only `deposit` and `age`-dependent costs are masked explicitly
    here, because those are the two places a stray dead-agent contribution
    would actually leak into a field.
    """

    def __init__(self, config: SwarmConfig, device):
        n = config.population_cap
        self.device = device

        self.pos = torch.zeros(n, 2, device=device)
        self.dir = torch.zeros(n, device=device)
        self.gene = torch.zeros(n, 3, device=device)          # OKLab
        self.energy = torch.zeros(n, device=device)
        self.age = torch.zeros(n, dtype=torch.long, device=device)
        self.alive = torch.zeros(n, dtype=torch.bool, device=device)

        # Rolling gain history for propriocezione (§5.2b): as of the start of
        # a step, gain_last is t-1 and gain_prev is t-2.
        self.gain_last = torch.zeros(n, device=device)
        self.gain_prev = torch.zeros(n, device=device)

    @property
    def n(self) -> int:
        return self.pos.shape[0]

    @property
    def n_alive(self) -> int:
        return int(self.alive.sum().item())


# ==================== sampling ====================

def _bilinear_sample(field: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Bilinear sample of `field` -- (H, W) or (H, W, C) -- at pixel coordinates
    (x, y) of arbitrary matching shape. Returns a tensor shaped like x/y (plus
    a trailing channel dim if `field` has one).

    Shared by perception, nutrient lookup and gene lookup so there is exactly
    one interpolation convention in the whole package.
    """
    height, width = field.shape[0], field.shape[1]
    is_scalar = field.dim() == 2
    channels = field.shape[2] if not is_scalar else 1

    inp = field.reshape(height, width, channels).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

    gx = (x + 0.5) / width * 2.0 - 1.0
    gy = (y + 0.5) / height * 2.0 - 1.0
    shape = x.shape
    grid = torch.stack([gx, gy], dim=-1).reshape(1, -1, 1, 2)

    sampled = F.grid_sample(inp, grid, mode='bilinear', padding_mode='border', align_corners=False)
    sampled = sampled.squeeze(0).squeeze(-1).T.reshape(*shape, channels)

    return sampled.squeeze(-1) if is_scalar else sampled


def sample_positions_from_field(field: torch.Tensor, count: int, rng: torch.Generator,
                                 jitter: float = 0.5) -> torch.Tensor:
    """
    `count` positions drawn with probability proportional to a non-negative
    field (nutrient, typically). Used for the initial population and for
    repopulation (§5.7) -- **never** for anything driven by `error`.
    """
    device = field.device
    probs = field.reshape(-1).clamp(min=0)
    total = probs.sum()

    if total <= 0:
        idx = torch.randint(0, field.numel(), (count,), generator=rng, device=device)
    else:
        idx = torch.multinomial(probs / total, count, replacement=True, generator=rng)

    width = field.shape[1]
    xi = (idx % width).float()
    yi = (idx // width).float()
    xi = xi + (torch.rand(count, generator=rng, device=device) - 0.5) * jitter
    yi = yi + (torch.rand(count, generator=rng, device=device) - 0.5) * jitter
    return torch.stack([xi, yi], dim=-1)


# ==================== 5.1 perception ====================

def _agent_density(agents: AgentState, height: int, width: int) -> torch.Tensor:
    """Lightly blurred count of alive agents per pixel, for `w_crowd` (§5.1)."""
    device = agents.device
    xi = agents.pos[:, 0].long().clamp(0, width - 1)
    yi = agents.pos[:, 1].long().clamp(0, height - 1)
    flat_idx = torch.where(agents.alive, yi * width + xi, torch.full_like(yi, -1))
    valid = flat_idx >= 0

    density = torch.zeros(height * width, device=device)
    density.scatter_add_(0, flat_idx[valid], torch.ones(int(valid.sum()), device=device))
    density = density.reshape(1, 1, height, width)
    density = F.avg_pool2d(F.pad(density, (1, 1, 1, 1), mode='replicate'), 3, stride=1)
    return density.squeeze(0).squeeze(0)


def population_density(agents: AgentState, height: int, width: int) -> torch.Tensor:
    """Public wrapper around the crowd-density field, for diagnostics/lab views (§11.2)."""
    return _agent_density(agents, height, width)


def perceive(agents: AgentState, fields, config: SwarmConfig, error: torch.Tensor = None):
    """
    Sample the attractiveness field at the three sensors (§5.1). Reads
    `pheromone` and `nutrient`, optionally `error` when `guidance > 0`
    (default 0 -- see §2.2). Returns (S_L, S_C, S_R), each shape (N,).
    """
    attract = fields.pheromone + config.w_nutrient * fields.nutrient

    if config.w_crowd > 0:
        attract = attract - config.w_crowd * _agent_density(agents, fields.height, fields.width)
    if config.guidance > 0 and error is not None:
        attract = attract + config.guidance * error

    offsets = torch.tensor([-config.sensor_angle, 0.0, config.sensor_angle], device=agents.device)
    angles = agents.dir.unsqueeze(1) + offsets.unsqueeze(0)                    # (N, 3)
    sx = agents.pos[:, 0:1] + config.sensor_distance * torch.cos(angles)
    sy = agents.pos[:, 1:2] + config.sensor_distance * torch.sin(angles)

    samples = _bilinear_sample(attract, sx, sy)                               # (N, 3)
    return samples[:, 0], samples[:, 1], samples[:, 2]


# ==================== 5.2 + 5.2b rotation ====================

def rotate(agents: AgentState, s_l: torch.Tensor, s_c: torch.Tensor, s_r: torch.Tensor,
           config: SwarmConfig, rng: torch.Generator):
    """
    Turn towards the stronger sensor (§5.2), then override with a random
    tumble when the agent's own gain has stalled or dropped (§5.2b). The
    tumble reads only `agents.gain_last/gain_prev` -- the agent's own recent
    success, not the field -- which is why it does not count as cheating.
    """
    n = agents.n
    device = agents.device

    if config.rotation_mode == 'classic':
        is_max_c = (s_c >= s_l) & (s_c >= s_r)
        is_min_c = (s_c <= s_l) & (s_c <= s_r)
        turn_right = s_r > s_l

        delta = torch.zeros(n, device=device)
        delta = torch.where(turn_right, torch.full_like(delta, config.rotation_angle), delta)
        delta = torch.where(~turn_right, torch.full_like(delta, -config.rotation_angle), delta)
        delta = torch.where(is_max_c, torch.zeros_like(delta), delta)

        random_turn = (torch.rand(n, device=device, generator=rng) * 2 - 1) * config.rotation_angle
        delta = torch.where(is_min_c, random_turn, delta)
    else:
        denom = (s_l + s_c + s_r).clamp(min=1e-6)
        delta = config.rotation_angle * (s_r - s_l) / denom

    delta = delta + torch.randn(n, device=device, generator=rng) * config.angle_noise

    if config.tumble_enabled:
        d_gain = agents.gain_last - agents.gain_prev
        tumbling = (d_gain < config.tumble_threshold) & agents.alive
        tumble_delta = (torch.rand(n, device=device, generator=rng) * 2 - 1) * config.tumble_angle
        delta = torch.where(tumbling, tumble_delta, delta)

    agents.dir = agents.dir + delta


# ==================== 5.3 movement ====================

def _reflect(coord: torch.Tensor, size: int):
    """Mirror a coordinate back into [0, size). Returns (coord, crossed_mask)."""
    upper = float(size - 1)
    over = coord >= size
    under = coord < 0
    out = coord.clone()
    out = torch.where(over, 2 * upper - coord, out)
    out = torch.where(under, -coord, out)
    return out.clamp(0.0, upper), (over | under)


def advance(agents: AgentState, fields, config: SwarmConfig, rng: torch.Generator):
    """
    Move along `dir` at `speed`, handling the canvas border (§5.3). Returns
    `(pos_prev, off_tissue)`: the pre-move position (needed for the deposit
    trail) and a mask of agents below `nutrient_threshold` at their new spot
    (fed into the starvation cost in §5.6).
    """
    pos_prev = agents.pos.clone()

    dx = config.speed * torch.cos(agents.dir)
    dy = config.speed * torch.sin(agents.dir)
    new_pos = agents.pos + torch.stack([dx, dy], dim=-1)

    if config.boundary_mode == 'reflect':
        new_x, crossed_x = _reflect(new_pos[:, 0], fields.width)
        new_y, crossed_y = _reflect(new_pos[:, 1], fields.height)
        agents.dir = torch.where(crossed_x, math.pi - agents.dir, agents.dir)
        agents.dir = torch.where(crossed_y, -agents.dir, agents.dir)
        agents.pos = torch.stack([new_x, new_y], dim=-1)
    else:  # 'clamp_random'
        upper_x, upper_y = float(fields.width - 1), float(fields.height - 1)
        out_of_bounds = (new_pos[:, 0] < 0) | (new_pos[:, 0] >= fields.width) | \
                         (new_pos[:, 1] < 0) | (new_pos[:, 1] >= fields.height)
        agents.pos = torch.stack([new_pos[:, 0].clamp(0, upper_x), new_pos[:, 1].clamp(0, upper_y)], dim=-1)
        random_dir = torch.rand(agents.n, device=agents.device, generator=rng) * 2 * math.pi
        agents.dir = torch.where(out_of_bounds, random_dir, agents.dir)

    nutrient_here = _bilinear_sample(fields.nutrient, agents.pos[:, 0], agents.pos[:, 1])
    off_tissue = nutrient_here < config.nutrient_threshold

    return pos_prev, off_tissue


# ==================== 5.4 deposit ====================

_KERNEL_CACHE = {}


def _kernel_offsets(radius: float, device):
    """Cached (dy, dx, gaussian weight) of a disc-shaped brush kernel."""
    key = (round(float(radius), 3), str(device))
    if key not in _KERNEL_CACHE:
        r = max(1, int(math.ceil(radius)))
        span = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
        dy, dx = torch.meshgrid(span, span, indexing='ij')
        dist2 = dy * dy + dx * dx
        inside = dist2 <= (radius ** 2 + 1e-6)
        sigma = max(radius / 2.0, 1e-3)
        w = torch.exp(-dist2 / (2 * sigma * sigma))
        _KERNEL_CACHE[key] = (dy[inside].reshape(-1), dx[inside].reshape(-1), w[inside].reshape(-1))
    return _KERNEL_CACHE[key]


def _splat_trail(pos_prev: torch.Tensor, pos: torch.Tensor, alive: torch.Tensor,
                  config: SwarmConfig, width: int, height: int, device):
    """
    Flattened (yi, xi, kernel_weight, agent_idx) for every kernel sample
    along every agent's trail this step. `kernel_weight` is the raw brush
    kernel, with neither `deposit_alpha` nor `deposit_pheromone` applied yet
    -- `deposit` and `selection.consume` each scale it for their own purpose,
    but both start from this one splat so they always agree on who touched
    which pixel by how much.
    """
    trail_samples = max(1, math.ceil(config.speed / 0.5))
    t = torch.linspace(0.0, 1.0, trail_samples, device=device).view(1, -1, 1)   # (1, K, 1)

    trail = pos_prev.unsqueeze(1) + (pos.unsqueeze(1) - pos_prev.unsqueeze(1)) * t   # (N, K, 2)

    dy, dx, kw = _kernel_offsets(config.brush_radius, device)                  # (M,)
    n, k, m = trail.shape[0], trail.shape[1], dy.shape[0]

    px = trail[:, :, 0].unsqueeze(-1) + dx.view(1, 1, -1)                      # (N, K, M)
    py = trail[:, :, 1].unsqueeze(-1) + dy.view(1, 1, -1)
    weight = kw.view(1, 1, -1).expand(n, k, m)
    agent_idx = torch.arange(n, device=device).view(-1, 1, 1).expand(n, k, m)
    alive_mask = alive.view(-1, 1, 1).expand(n, k, m)

    xi = px.reshape(-1).round().long().clamp(0, width - 1)
    yi = py.reshape(-1).round().long().clamp(0, height - 1)
    weight = torch.where(alive_mask.reshape(-1), weight.reshape(-1), torch.zeros(n * k * m, device=device))
    agent_idx = agent_idx.reshape(-1)

    return yi, xi, weight, agent_idx


@dataclass
class Splat:
    """Everything `selection.consume` needs to attribute gain the same way `deposit` painted it."""
    flat_idx: torch.Tensor      # (E,) pixel index (y*W+x) per splat entry
    a: torch.Tensor             # (E,) a_i(p) = deposit_alpha * kernel_weight, per entry
    agent_idx: torch.Tensor     # (E,) which agent each entry belongs to
    total_w: torch.Tensor       # (H*W,) W(p) = sum of a_i(p) over agents touching p


def deposit(agents: AgentState, fields, pos_prev: torch.Tensor, config: SwarmConfig) -> Splat:
    """
    Paint the trail segment (pos_prev -> pos) onto canvas and pheromone
    (§5.4). Order-independent: accumulates `W(p)` and `Σ a_i(p)·gene_i` across
    every agent touching a pixel and resolves once, so the result does not
    depend on scatter scheduling.
    """
    device = agents.device
    h, w = fields.height, fields.width

    yi, xi, kw, agent_idx = _splat_trail(pos_prev, agents.pos, agents.alive, config, w, h, device)
    flat_idx = yi * w + xi

    a = kw * config.deposit_alpha
    total_w = torch.zeros(h * w, device=device).scatter_add_(0, flat_idx, a)

    gene_per_entry = agents.gene[agent_idx] * a.unsqueeze(-1)
    weighted_color = torch.zeros(h * w, 3, device=device).index_add_(0, flat_idx, gene_per_entry)

    alpha = total_w.clamp(max=1.0)
    mixed_color = weighted_color / total_w.clamp(min=1e-8).unsqueeze(-1)

    canvas_flat = fields.canvas.reshape(h * w, 3)
    canvas_flat = (1 - alpha).unsqueeze(-1) * canvas_flat + alpha.unsqueeze(-1) * mixed_color
    fields.canvas = canvas_flat.reshape(h, w, 3)

    pher_add = torch.zeros(h * w, device=device).scatter_add_(0, flat_idx, kw * config.deposit_pheromone)
    fields.pheromone = (fields.pheromone.reshape(-1) + pher_add).reshape(h, w)

    return Splat(flat_idx=flat_idx, a=a, agent_idx=agent_idx, total_w=total_w)


# ==================== spawning ====================

def spawn_agents(agents: AgentState, fields, config: SwarmConfig, indices: torch.Tensor,
                  rng: torch.Generator, gene_source: torch.Tensor = None,
                  mutation_sigma: float = None):
    """
    Fill dead `indices` with fresh agents. Position is drawn from `nutrient`
    -- never from `error` (§5.7, the trap that would look like cheating
    dressed up as an implementation detail). Gene is sampled from
    `gene_source` (defaults to `fields.base`, i.e. the NCA output -- §3) at
    the spawn point, plus a small mutation, so generation zero already
    resembles stage 2 instead of starting from noise.

    `mutation_sigma` defaults to `config.mutation_sigma`; pass the
    climate-scheduled value (§8) explicitly so repopulation follows the same
    annealing as ordinary reproduction instead of a frozen constant.
    """
    count = int(indices.shape[0])
    if count == 0:
        return

    device = agents.device
    source = fields.base if gene_source is None else gene_source
    sigma = config.mutation_sigma if mutation_sigma is None else mutation_sigma

    pos = sample_positions_from_field(fields.nutrient, count, rng, jitter=1.0)
    gene = _bilinear_sample(source, pos[:, 0], pos[:, 1])
    gene = gene + torch.randn(count, 3, device=device, generator=rng) * sigma
    gene = clamp_gamut(gene)

    agents.pos[indices] = pos
    agents.dir[indices] = torch.rand(count, device=device, generator=rng) * 2 * math.pi
    agents.gene[indices] = gene
    agents.energy[indices] = config.initial_energy
    agents.age[indices] = 0
    agents.alive[indices] = True
    agents.gain_last[indices] = 0.0
    agents.gain_prev[indices] = 0.0
