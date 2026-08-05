"""
Stage 3 inside the timeline (PLAN §3.3, Evolutionary_Swarm.md §11.4).

`SwarmStage` is the adapter between the timeline's frame loop and the
`Simulation` the lab has been calibrating. It implements `StageLayer`: given
how far along its window it is and the NCA tissue of the current frame, it
advances the swarm and hands back an RGBA layer.

Three things about the timeline make this more than a wrapper, and all three
are handled here rather than in `Simulation`:

- **The NCA is still growing.** The swarm window opens at 12s and the NCA runs
  to 16s, so `base` and `nutrient` are moving targets. Every frame they are
  refreshed through `Fields.refresh_base`, which is the single seam the design
  doc reserved for this.
- **The swarm only exists where there is tissue.** The layer's alpha is the
  NCA alpha, so the swarm can never paint on the background, and the agents'
  nutrient is that same alpha, so they never walk there either.
- **Resolution.** The simulation runs at `work_size`, not at the supersampled
  canvas; the layer is scaled up once, at the end, next to everything else the
  frame resolves.

The camera crop is deliberately *not* this module's business: the timeline
crops after compositing, so the swarm paints the whole canvas and the camera
decides what is seen. A swarm that knew about the crop would produce different
pigment depending on the camera move, which is not a thing that should be true.
"""

from typing import Optional

import cv2
import numpy as np
import torch

from config.swarm_config import SwarmConfig, SwarmStageConfig
from swarm.colorspace import oklab_to_srgb, srgb_to_oklab
from swarm.simulation import Simulation
from rendering.easing import apply_easing
from utils.log import get_logger

logger = get_logger(__name__)


class SwarmStage:
    """
    The evolutionary swarm as a timeline stage.

    Construction is cheap and deliberately does no work: the simulation cannot
    be built until the first frame that actually has tissue on it, because that
    tissue *is* the starting canvas.
    """

    def __init__(self, target_rgb: np.ndarray, config: SwarmConfig,
                 stage_config: SwarmStageConfig, canvas_size: int):
        self.config = config
        self.stage = stage_config
        self.canvas_size = canvas_size
        self.work_size = config.work_size

        # Target at working resolution, in OKLab. This is the only place the
        # timeline's target image enters the swarm, and it reaches the agents
        # exclusively through `Fields.compute_error` (§1).
        target = cv2.resize(target_rgb, (self.work_size, self.work_size),
                            interpolation=cv2.INTER_AREA)
        self.target_oklab = srgb_to_oklab(torch.from_numpy(target.copy()))

        self.sim: Optional[Simulation] = None
        self.steps_run = 0

    # ------------------------------------------------------------ tissue -> fields

    def _tissue_to_fields(self, tissue: np.ndarray):
        """Split an RGBA NCA layer into `(base_oklab, nutrient)` at work size."""
        small = cv2.resize(tissue, (self.work_size, self.work_size),
                           interpolation=cv2.INTER_AREA)

        rgb = small[..., :3].astype(np.float32) / 255.0
        alpha = small[..., 3].astype(np.float32) / 255.0

        if self.stage.nutrient_blur > 0:
            alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=self.stage.nutrient_blur,
                                     sigmaY=self.stage.nutrient_blur)

        base_oklab = srgb_to_oklab(torch.from_numpy(rgb.copy()))
        return base_oklab, torch.from_numpy(alpha.copy())

    def _build(self, tissue: np.ndarray):
        """First contact: the tissue of this frame becomes the starting canvas."""
        base_oklab, nutrient = self._tissue_to_fields(tissue)
        self.sim = Simulation(self.config, self.target_oklab, base_oklab, nutrient)

        for _ in range(max(0, self.stage.warmup_steps)):
            self.sim.step()
        self.steps_run += max(0, self.stage.warmup_steps)

        logger.info('Swarm stage: %d agents at %dpx, %d steps/frame (warmup %d)',
                    self.config.population_cap, self.work_size,
                    self.stage.steps_per_frame, self.stage.warmup_steps)

    # ------------------------------------------------------------ StageLayer

    def layer(self, progress: float, time: float, beneath: np.ndarray,
              tissue: np.ndarray = None) -> Optional[np.ndarray]:
        """
        Advance the swarm and return its RGBA layer at canvas resolution.

        Returns None before there is any tissue to live on -- the timeline then
        composites nothing, which is the correct picture of a swarm that has not
        started rather than a blank layer painted over the frame.
        """
        if tissue is None or not tissue[..., 3].any():
            return None

        if self.sim is None:
            self._build(tissue)
        else:
            base_oklab, nutrient = self._tissue_to_fields(tissue)
            self.sim.fields.refresh_base(base_oklab, nutrient)

        for _ in range(max(1, self.stage.steps_per_frame)):
            self.sim.step()
        self.steps_run += max(1, self.stage.steps_per_frame)

        canvas = self.sim.render_canvas_srgb().cpu().numpy()
        canvas = cv2.resize(canvas, (self.canvas_size, self.canvas_size),
                            interpolation=cv2.INTER_LINEAR)

        # The swarm shows only where the NCA has tissue, and only as strongly as
        # the blend-in allows -- so the handover from stage 2 is a dissolve, not
        # a cut on the frame the window opens.
        fade = 1.0
        if self.stage.blend_in > 0:
            fade = apply_easing(self.stage.blend_easing,
                                min(1.0, progress / self.stage.blend_in))

        alpha = (tissue[..., 3].astype(np.float32) * fade).clip(0, 255)

        out = np.empty((self.canvas_size, self.canvas_size, 4), dtype=np.uint8)
        out[..., :3] = canvas
        out[..., 3] = alpha.astype(np.uint8)
        return out

    # ------------------------------------------------------------ diagnostics

    def report(self):
        """Log the §12 headline once the render is done."""
        if self.sim is None or not self.sim.history:
            logger.info('Swarm stage: never activated (no tissue in its window)')
            return

        last = self.sim.history[-1]
        improvement = 1.0 - last.mean_error / max(self.sim.baseline_error, 1e-9)
        logger.info('Swarm stage: %d steps | error %.5f vs stage-2 %.5f (%+.2f%%) | '
                    'pop %d | eating %.0f%%',
                    self.steps_run, last.mean_error, self.sim.baseline_error,
                    improvement * 100, last.population, last.positive_gain_fraction * 100)


def build_swarm_stage(pipeline) -> Optional[SwarmStage]:
    """
    Assemble the stage from a `PipelineConfig`.

    The swarm's parameters come from the lab preset named in
    `SwarmStageConfig.preset`, so production runs exactly what the lab
    calibrated instead of a second copy of the numbers.
    """
    from config.lab_config import PRESETS
    from config.swarm_config import apply_overrides
    from swarm.experiment import load_target

    stage_config = pipeline.swarm_stage
    if not stage_config.enabled:
        return None

    if stage_config.preset not in PRESETS:
        raise KeyError(f'unknown swarm preset {stage_config.preset!r}; '
                       f'known: {", ".join(sorted(PRESETS))}')

    work_size = stage_config.work_size
    population = pipeline.swarm.population_cap
    if stage_config.population_scale:
        # Keep agents-per-pixel constant with the lab's 256px calibration.
        population = int(population * (work_size / 256.0) ** 2)

    config = apply_overrides(pipeline.swarm, PRESETS[stage_config.preset].overrides)
    config.work_size = work_size
    config.population_cap = population
    config.device = pipeline.device

    window_frames = max(1, int(round(pipeline.timing.swarm.duration * pipeline.render_fps)))
    config.sim_steps = stage_config.warmup_steps + window_frames * stage_config.steps_per_frame
    config.metrics_stride = max(1, stage_config.steps_per_frame)

    target_rgb, _ = load_target(pipeline.target_image, work_size)

    logger.info("Swarm stage: preset '%s', %d agents at %dpx, %d simulated steps "
                "over %d frames", stage_config.preset, population, work_size,
                config.sim_steps, window_frames)

    return SwarmStage(target_rgb, config, stage_config, pipeline.canvas_size)
