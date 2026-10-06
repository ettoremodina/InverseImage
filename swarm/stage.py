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
from tqdm import tqdm

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
                 stage_config: SwarmStageConfig, canvas_size: int,
                 window_frames: int = 1):
        self.config = config
        self.stage = stage_config
        self.canvas_size = canvas_size
        self.work_size = config.work_size
        self._window_frames = max(1, int(window_frames))

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

        # The swarm shows only where the NCA has tissue, and only as strongly as
        # the blend-in allows -- so the handover from stage 2 is a dissolve, not
        # a cut on the frame the window opens.
        fade = 1.0
        if self.stage.blend_in > 0:
            fade = apply_easing(self.stage.blend_easing,
                                min(1.0, progress / self.stage.blend_in))

        return self._compose_layer(tissue, fade)

    def _compose_layer(self, tissue: np.ndarray, fade: float) -> np.ndarray:
        """The current canvas as an RGBA layer at canvas resolution, masked by the tissue."""
        canvas = self.sim.render_canvas_srgb().cpu().numpy()
        canvas = cv2.resize(canvas, (self.canvas_size, self.canvas_size),
                            interpolation=cv2.INTER_LINEAR)

        alpha = (tissue[..., 3].astype(np.float32) * fade).clip(0, 255)

        out = np.empty((self.canvas_size, self.canvas_size, 4), dtype=np.uint8)
        out[..., :3] = canvas
        out[..., 3] = alpha.astype(np.uint8)
        return out

    # ------------------------------------------------------------ stills

    def warm_to(self, progress: float, tissue: np.ndarray) -> Optional[np.ndarray]:
        """
        Advance the swarm to `progress` through its window against one tissue,
        and return the layer -- the still-frame path (`render.py --mode still`).

        The swarm is stateful: its canvas at 21s is the product of every step
        since the window opened, so a still cannot simply be evaluated at a
        time the way the SCA and NCA layers can. This runs those steps without
        rasterising the frames in between, which is the whole saving: the
        per-cell Cairo pass, not the simulation, is what makes a video slow.

        The approximation is deliberate and worth stating: the tissue is held
        fixed at the value it has *at the still's time*, instead of growing
        under the swarm as it does in the video. At any time after the NCA
        window closes -- which is every frame worth calibrating on, the NCA
        ends at 16s and the swarm runs to 22s -- the tissue is already static
        and the two are identical. Before that, the still shows the swarm as if
        it had always had the tissue it has now, and `render.py` says so.
        """
        window_frames = max(1, int(round(progress * self._window_frames)))
        steps = self.stage.warmup_steps + window_frames * max(1, self.stage.steps_per_frame)

        base_oklab, nutrient = self._tissue_to_fields(tissue)
        self.sim = Simulation(self.config, self.target_oklab, base_oklab, nutrient)

        logger.info('Swarm still: %d steps at %d agents, %dpx (tissue held fixed)',
                    steps, self.config.population_cap, self.work_size)
        for _ in tqdm(range(steps), desc='Swarm', leave=False):
            self.sim.step()
        self.steps_run = steps

        return self._compose_layer(tissue, fade=1.0)

    # ------------------------------------------------------------ diagnostics

    def report(self):
        """
        Log what the swarm actually contributed.

        This used to print `1 - final_error / sim.baseline_error` over the whole
        frame, and that number was worthless in production for two compounding
        reasons. `baseline_error` is captured when the simulation is built --
        at the *start* of the swarm window, when the NCA still has four seconds
        of growing to do -- so most of the reported drop was stage 2 finishing
        its own job, not stage 3 doing anything. And measured over the whole
        frame, ~80% of the pixels are background that both images already agree
        on, which shrinks whatever is left by another factor of five. It read
        +14.9% on a run whose real contribution was +1.5%.

        What is printed instead is `swarm.quality`: the canvas compared against
        the tissue it is painting on *right now*, on the tissue only, plus the
        share of that tissue it has actually put paint on. Those two numbers
        together are the ones that predict whether anything is visible.
        """
        if self.sim is None or not self.sim.history:
            logger.info('Swarm stage: never activated (no tissue in its window)')
            return

        from swarm.quality import measure_simulation

        quality = measure_simulation(self.sim)
        last = self.sim.history[-1]
        logger.info('Swarm stage: %d steps | %+.2f%% error on the tissue vs the NCA under it '
                    '| painted %.0f%% of it | stroke coherence %.2f | pop %d | eating %.0f%%',
                    self.steps_run, quality.improvement * 100, quality.coverage * 100,
                    quality.stroke_coherence, last.population,
                    last.positive_gain_fraction * 100)


def build_swarm_stage(pipeline, overrides: dict = None) -> Optional[SwarmStage]:
    """
    Assemble the stage from a `PipelineConfig`.

    The swarm's parameters come from the lab preset named in
    `SwarmStageConfig.preset`, so production runs exactly what the lab
    calibrated instead of a second copy of the numbers.

    `overrides` is applied **last**, after the preset and after the population
    has been scaled for the working resolution -- so `population_cap` given
    here means that many agents, literally, and not that many times four.
    Hand tuning against a still (`render.py --mode still --swarm-set ...`) is
    the only caller that passes it.
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

    if overrides:
        config = apply_overrides(config, overrides)
        logger.info('Swarm overrides: %s',
                    ' '.join(f'{k}={v}' for k, v in sorted(overrides.items())))

    target_rgb, _ = load_target(pipeline.target_image, config.work_size)

    logger.info("Swarm stage: preset '%s', %d agents at %dpx, %d simulated steps "
                "over %d frames", stage_config.preset, config.population_cap,
                config.work_size, config.sim_steps, window_frames)

    return SwarmStage(target_rgb, config, stage_config, pipeline.canvas_size, window_frames)
