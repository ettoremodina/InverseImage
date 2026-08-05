"""
The single timeline (PLAN 3.1).

This replaced the old sequential renderer, now deleted. That one produced
`combined.mp4` and `particles.mp4` separately and concatenated them with
ffmpeg, freezing the background for the last stage -- at the cut the tree's
sway stopped dead.

Here there is one loop over frames. Every stage has its own activity curve and
is evaluated at every frame, so the stages overlap: the NCA enters while the
tree is still growing, the scaffold dissolves under the flesh, and the sway
never stops.

Adding stage 3 is a matter of passing an object with a `layer()` method (see
`StageLayer`); the loop does not need to know anything else about it. That is
deliberate -- the evolutionary swarm lands later and must not require this file
to be rewritten.
"""

from typing import Any, Dict, Optional, Protocol

import numpy as np
from tqdm import tqdm

from config.camera_config import CameraConfig
from config.grading_config import GradingConfig
from config.render_config import NCARenderConfig, SCARenderConfig, ScaffoldFadeConfig
from config.timing_config import StageWindow, TimingConfig
from color.grading import Grader
from utils.log import get_logger

from .camera import Camera
from .easing import apply_easing
from .nca_renderer import NCARenderer, composite
from .sca_renderer import SCARenderer
from .supersample import resolve
from .utils import create_surface, get_time_dilated_indices, max_polyline_depth, \
    open_video_writer, surface_to_numpy

logger = get_logger(__name__)


class StageLayer(Protocol):
    """
    What the timeline needs from a stage in order to draw it.

    Given how far along its own window it is, the frame underneath it, and the
    NCA tissue of this frame, return an RGBA layer with straight alpha at canvas
    resolution, or None when it has nothing to draw.

    `tissue` is the stage-2 layer *before* it was composited into `beneath`, and
    it is passed separately because its alpha is the only record of where the
    NCA has actually grown: `beneath` is opaque everywhere, background included.
    The swarm needs that mask both to know where its agents may live and to know
    where it is allowed to show (PLAN §3.3).
    """

    def layer(self, progress: float, time: float, beneath: np.ndarray,
              tissue: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        ...


def window_progress(window: StageWindow, time: float) -> float:
    """Eased progress of one stage at a given time, clamped to [0, 1]."""
    if time <= window.start:
        return 0.0
    if time >= window.end or window.duration <= 0:
        return 1.0
    return apply_easing(window.easing, (time - window.start) / window.duration)


class TimelineRenderer:
    """
    One loop, every stage evaluated at every frame.

    The canvas is the internal (supersampled) resolution; the camera crop and
    the reduction to video size happen once per frame, at the very end, in a
    single resampling.
    """

    def __init__(self,
                 sca_config: SCARenderConfig,
                 nca_config: NCARenderConfig,
                 timing: TimingConfig,
                 camera_config: CameraConfig = None,
                 grading_config: GradingConfig = None,
                 scaffold_config: ScaffoldFadeConfig = None,
                 output_size: int = 512,
                 fps: int = 20):
        self.sca_config = sca_config
        self.nca_config = nca_config
        self.timing = timing
        self.scaffold = scaffold_config or ScaffoldFadeConfig()
        self.output_size = output_size
        self.fps = fps

        self.canvas_size = sca_config.output_width
        if nca_config.output_width != self.canvas_size:
            raise ValueError(
                'SCA and NCA renderers must share the internal canvas size '
                f'({sca_config.output_width} vs {nca_config.output_width})'
            )

        self.sca_renderer = SCARenderer(sca_config)
        self.nca_renderer = NCARenderer(nca_config)
        self.camera = Camera(camera_config, self.canvas_size)
        self.grader = Grader(grading_config)

    # ------------------------------------------------------------ background
    def _background(self) -> np.ndarray:
        color = self.sca_config.background_color
        background = np.empty((self.canvas_size, self.canvas_size, 4), dtype=np.uint8)
        background[..., :3] = [int(round(c * 255)) for c in color[:3]]
        background[..., 3] = 255
        return background

    # ------------------------------------------------------------------ tree
    def _tree_layer(self, sca_data: Dict[str, Any], depth_limit: Optional[float],
                    time: float, opacity: float) -> np.ndarray:
        """The tree on its own transparent layer, so it can be faded per pixel."""
        surface, ctx = create_surface(
            self.canvas_size, self.canvas_size, (0.0, 0.0, 0.0, 0.0),
            self.sca_config.antialiasing,
        )
        scale_x, scale_y = (
            self.canvas_size / sca_data['source_width'],
            self.canvas_size / sca_data['source_height'],
        )
        geom = self.sca_renderer.get_geometry(sca_data['polylines'])
        self.sca_renderer._draw_polylines(
            ctx, geom, scale_x, scale_y, depth_limit, time=time, opacity=opacity
        )
        return surface_to_numpy(surface, self.canvas_size, self.canvas_size,
                                unpremultiply=True)

    def _fade_scaffold(self, tree: np.ndarray, flesh: Optional[np.ndarray],
                       fade: float) -> np.ndarray:
        """
        Dissolve the tree where the flesh covers it (PLAN 3.4, 'alpha' mode).

        The coverage mask is the NCA layer's own alpha, blurred: a branch under
        a dense patch of tissue disappears, a branch out in the open stays.
        """
        import cv2

        if fade <= 0 or flesh is None:
            return tree

        coverage = flesh[..., 3].astype(np.float32) / 255.0
        sigma = max(0.1, self.scaffold.blur * self.nca_config.render_scale)
        coverage = cv2.GaussianBlur(coverage, (0, 0), sigmaX=sigma, sigmaY=sigma)
        coverage = np.clip(coverage * self.scaffold.coverage_gain, 0.0, 1.0)

        keep = 1.0 - fade * self.scaffold.strength * coverage
        tree = tree.copy()
        tree[..., 3] = (tree[..., 3].astype(np.float32) * keep).astype(np.uint8)
        return tree

    # ------------------------------------------------------------------ main
    def render(self, sca_data: Dict[str, Any], nca_data: Dict[str, Any],
               output_path: str, stages: Optional[Dict[str, StageLayer]] = None):
        """
        Render the whole video in one pass.

        Args:
            sca_data: SCA render data (polylines).
            nca_data: NCA frames as loaded from the npz.
            output_path: destination mp4.
            stages: optional extra stages by name. Only 'swarm' is consulted for
                now; it is the slot the evolutionary swarm plugs into.
        """
        timing = self.timing
        timing.validate()

        stages = stages or {}
        swarm = stages.get('swarm')
        if swarm is None:
            logger.info('Timeline: no swarm stage supplied, the swarm window '
                        '(%.1fs -> %.1fs) will render as a hold',
                        timing.swarm.start, timing.swarm.end)

        total_frames = max(1, int(round(timing.total_duration * self.fps)))
        max_depth = max_polyline_depth(sca_data['polylines'])

        nca_frames = nca_data['frames']
        nca_window_frames = max(1, int(round(timing.nca.duration * self.fps)))
        nca_indices = get_time_dilated_indices(
            len(nca_frames), nca_window_frames,
            self.nca_config.initial_repeats, self.nca_config.decay_rate,
        )

        background = self._background()
        smoothing = self.nca_config.temporal_smoothing
        accumulated: Optional[np.ndarray] = None
        flesh: Optional[np.ndarray] = None

        logger.info('Timeline: %d frames at %d fps (%.1fs), canvas %dpx -> video %dpx',
                    total_frames, self.fps, total_frames / self.fps,
                    self.canvas_size, self.output_size)

        with open_video_writer(output_path, self.fps) as writer:
            for index in tqdm(range(total_frames), desc='Timeline'):
                time = index / self.fps

                # ---------------------------------------------------- stages
                sca_progress = window_progress(timing.sca, time)
                nca_progress = window_progress(timing.nca, time)
                fade_progress = (window_progress(timing.scaffold_fade, time)
                                 if self.scaffold.enabled else 0.0)

                # ------------------------------------------------------- NCA
                if nca_progress > 0.0:
                    position = int(round(nca_progress * (len(nca_indices) - 1)))
                    frame = nca_frames[nca_indices[position]]
                    layer = self.nca_renderer.render_layer(frame, nca_progress)

                    if smoothing > 0:
                        current = layer.astype(np.float32)
                        accumulated = (current if accumulated is None
                                       else accumulated * smoothing + current * (1.0 - smoothing))
                        layer = accumulated.astype(np.uint8)

                    flesh = layer

                # ------------------------------------------------------- SCA
                depth_limit = None if sca_progress >= 1.0 else sca_progress * max_depth
                opacity = (1.0 - fade_progress * self.scaffold.strength
                           if self.scaffold.mode == 'time' else 1.0)
                tree = self._tree_layer(sca_data, depth_limit, time, opacity)

                if self.scaffold.mode == 'alpha':
                    tree = self._fade_scaffold(tree, flesh, fade_progress)

                # ------------------------------------------------- composite
                canvas = composite(background, tree)
                if flesh is not None:
                    canvas = composite(canvas, flesh)

                # ----------------------------------------------------- swarm
                # After the flesh, before the camera: the swarm repaints the
                # tissue stage 2 grew, and the crop still happens once, at the
                # end, on the finished canvas.
                if swarm is not None:
                    swarm_progress = window_progress(timing.swarm, time)
                    if swarm_progress > 0.0:
                        swarm_layer = swarm.layer(swarm_progress, time, canvas, flesh)
                        if swarm_layer is not None:
                            canvas = composite(canvas, swarm_layer)

                # ------------------------------------------ camera + resolve
                crop = self.camera.crop(index / max(1, total_frames - 1))
                frame_out = resolve(canvas, self.output_size, crop)

                writer.append_data(self.grader.apply(frame_out)[..., :3])

        logger.info('Saved timeline animation: %s', output_path)

    def render_still(self, sca_data: Dict[str, Any], nca_data: Dict[str, Any],
                     time: float) -> np.ndarray:
        """
        One frame at an arbitrary time, fully graded.

        Used for calibration: staring at a still frame is the only sane way to
        tune cellularity, lighting and grading without re-rendering a video.
        """
        timing = self.timing
        max_depth = max_polyline_depth(sca_data['polylines'])
        nca_frames = nca_data['frames']

        sca_progress = window_progress(timing.sca, time)
        nca_progress = window_progress(timing.nca, time)
        fade_progress = window_progress(timing.scaffold_fade, time) if self.scaffold.enabled else 0.0

        flesh = None
        if nca_progress > 0.0:
            position = int(round(nca_progress * (len(nca_frames) - 1)))
            flesh = self.nca_renderer.render_layer(nca_frames[position], nca_progress)

        depth_limit = None if sca_progress >= 1.0 else sca_progress * max_depth
        opacity = (1.0 - fade_progress * self.scaffold.strength
                   if self.scaffold.mode == 'time' else 1.0)
        tree = self._tree_layer(sca_data, depth_limit, time, opacity)
        if self.scaffold.mode == 'alpha':
            tree = self._fade_scaffold(tree, flesh, fade_progress)

        canvas = composite(self._background(), tree)
        if flesh is not None:
            canvas = composite(canvas, flesh)

        total_frames = max(1, int(round(timing.total_duration * self.fps)))
        crop = self.camera.crop(time * self.fps / max(1, total_frames - 1))
        return self.grader.apply(resolve(canvas, self.output_size, crop))
