"""
Field state and field-level mechanics (Evolutionary_Swarm.md §4).

`target` is read in exactly one place -- `Fields.compute_error` -- and nowhere
else in the package. That discipline is the whole point of the design (§1):
mutation proposes colour, selection disposes, and `error` is the only channel
through which the target can veto.
"""

import torch
import torch.nn.functional as F

from config.swarm_config import SwarmConfig


class Fields:
    """
    Canvas, base, pheromone, nutrient and target, all at `work_size`
    resolution. See Evolutionary_Swarm.md §4 for what each one is and who
    reads it.
    """

    def __init__(self, target_oklab: torch.Tensor, base_oklab: torch.Tensor,
                 nutrient: torch.Tensor, config: SwarmConfig, device):
        self.config = config
        self.device = device
        self.height, self.width = target_oklab.shape[:2]

        self.target = target_oklab.to(device)
        self.base = base_oklab.to(device)
        self.canvas = self.base.clone()
        self.nutrient = nutrient.to(device)
        self.pheromone = torch.zeros(self.height, self.width, device=device)

    # ------------------------------------------------------------ error

    def compute_error(self, raw: bool = False) -> torch.Tensor:
        """
        `‖canvas - target‖` per pixel in OKLab, measured at `fitness_scale`
        resolution and broadcast back to full res (§5.5.3): coarser
        `fitness_scale` is what turns the render from faithful to gestural,
        because attribution (§5.5) then shares gain across a whole coarse
        cell instead of a single pixel.

        Returns e_eff = max(0, e - tolerance) unless `raw` is set, in which
        case it returns e itself (useful for the loss curve in §12, where
        clamping at tolerance would flatten the interesting part of the
        curve).
        """
        cfg = self.config
        k = max(1, int(cfg.fitness_scale))

        canvas = self.canvas.permute(2, 0, 1).unsqueeze(0)   # (1, 3, H, W)
        target = self.target.permute(2, 0, 1).unsqueeze(0)

        if k > 1:
            canvas = F.avg_pool2d(canvas, k, ceil_mode=True)
            target = F.avg_pool2d(target, k, ceil_mode=True)

        diff = canvas - target
        if cfg.error_metric == 'l2_sq':
            e = (diff * diff).sum(dim=1, keepdim=True)
        else:
            e = diff.norm(dim=1, keepdim=True)

        if k > 1:
            e = F.interpolate(e, size=(self.height, self.width), mode='nearest')

        e = e.squeeze(0).squeeze(0)
        if raw:
            return e
        return (e - cfg.tolerance).clamp(min=0.0)

    # ------------------------------------------------------------ 5.9 aging

    def age_pigment(self, error_after: torch.Tensor):
        """
        Relax the canvas towards `base` at a rate that rises with local
        error (§5.9). `decay_min > 0` always -- the picture never
        crystallises, which is what keeps the process running forever
        instead of settling into a fixed point.
        """
        cfg = self.config
        lam = cfg.decay_min + (cfg.decay_max - cfg.decay_min) * \
            (error_after / cfg.decay_e_ref).clamp(max=1.0)
        self.canvas = self.canvas + lam.unsqueeze(-1) * (self.base - self.canvas)

    # ------------------------------------------------------------ 5.10 respiration

    def breathe(self, nutrient: torch.Tensor = None):
        """
        Pheromone diffuses (3x3 box blur) and evaporates. `nutrient` is
        refreshed from the driving NCA alpha in production; the lab passes
        None and keeps the static tissue.
        """
        cfg = self.config
        if cfg.pheromone_diffuse:
            p = self.pheromone.unsqueeze(0).unsqueeze(0)
            p = F.avg_pool2d(F.pad(p, (1, 1, 1, 1), mode='replicate'), 3, stride=1)
            self.pheromone = p.squeeze(0).squeeze(0)
        self.pheromone = self.pheromone * cfg.evaporation

        if nutrient is not None:
            self.nutrient = nutrient.to(self.device)
