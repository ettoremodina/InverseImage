"""
Combined SCA+NCA renderer -- LEGACY.

Creates a single video where:
1. SCA tree grows progressively by depth
2. NCA cells grow on top of the final SCA tree (which remains in background)

Superseded by `timeline.py`, which evaluates every stage at every frame instead
of running them one after the other (PLAN 3.1). Kept until the evolutionary
swarm replaces `particles/`, so the old sequential pipeline stays renderable and
comparable.
"""

import numpy as np
from dataclasses import replace
from tqdm import tqdm
from typing import Dict, Any

from config.render_config import SCARenderConfig, NCARenderConfig
from .base import Renderer
from .sca_renderer import SCARenderer
from .nca_renderer import NCARenderer, composite
from .utils import get_time_dilated_indices, max_polyline_depth, open_video_writer


class CombinedRenderer(Renderer):
    def __init__(self, render_config: SCARenderConfig = None, nca_config: NCARenderConfig = None):
        super().__init__(render_config or SCARenderConfig())

        # The NCA layer is composited on top of the SCA tree, so it must be
        # transparent. Copy first: the caller's config must not be mutated.
        self.nca_config = replace(
            nca_config or NCARenderConfig(),
            background_color=(0.0, 0.0, 0.0, 0.0)
        )

        self.sca_renderer = SCARenderer(self.config)
        self.nca_renderer = NCARenderer(self.nca_config)
        
        self._cached_sca_data = None
        self._cached_max_depth = None
        self._cached_scale = None
    
    def _composite(self, bg_img: np.ndarray, fg_img: np.ndarray) -> np.ndarray:
        """Alpha-composite the NCA layer over the SCA layer."""
        return composite(bg_img, fg_img)

    def _cache_sca_metadata(self, sca_data: Dict[str, Any]):
        """Cache SCA metadata to avoid recomputation."""
        if self._cached_sca_data is not sca_data:
            self._cached_sca_data = sca_data
            self._cached_max_depth = max_polyline_depth(sca_data['polylines'])
            self._cached_scale = self._compute_scale(sca_data['source_width'], sca_data['source_height'])

    def render_frame(self, sca_data: Dict[str, Any], nca_frame: np.ndarray = None,
                     max_depth_limit: int = None, time: float = 0.0) -> np.ndarray:
        """Render a single combined frame."""
        self._cache_sca_metadata(sca_data)

        surface, ctx = self._create_surface()

        scale_x, scale_y = self._cached_scale
        geom = self.sca_renderer.get_geometry(sca_data['polylines'])

        self.sca_renderer._draw_polylines(
            ctx, geom, scale_x, scale_y, max_depth_limit, time=time
        )
        sca_image = self._surface_to_numpy(surface)

        if nca_frame is not None:
            h, w = nca_frame.shape[:2]
            nca_image = self.nca_renderer.render_frame(nca_frame, w, h)
            return self._composite(sca_image, nca_image)
            
        return sca_image

    def render_animation(self, sca_data: Dict[str, Any], nca_data: Dict[str, Any],
                         output_path: str, fps: int, sca_frames: int, nca_frames: int,
                         resolve=None):
        """
        Render combined SCA->NCA animation.

        Args:
            sca_data: SCA render data with branches
            nca_data: NCA frames data
            output_path: Output video path
            fps: Frames per second
            sca_frames: Number of frames for SCA growth phase
            nca_frames: Number of frames for NCA growth phase
            resolve: optional per-frame callable (frame, index) -> frame, used to
                crop/downscale the supersampled canvas and grade it
        """
        self._cache_sca_metadata(sca_data)
        
        max_depth = self._cached_max_depth
        nca_frames_data = nca_data["frames"]

        time = 0.0
        dt = 1.0 / fps
        total = 0

        with open_video_writer(output_path, fps) as writer:
            # Phase 1: SCA growth
            if sca_frames > 0:
                print(f"Rendering {sca_frames} SCA frames...")
                for i in tqdm(range(sca_frames), desc="SCA Phase"):
                    t_frac = i / max(sca_frames - 1, 1)
                    target_depth = int(t_frac * max_depth)
                    frame = self.render_frame(sca_data, nca_frame=None,
                                              max_depth_limit=target_depth, time=time)
                    writer.append_data(frame if resolve is None else resolve(frame, total))
                    time += dt
                    total += 1

            # Phase 2: NCA growth over the fully grown tree
            if nca_frames > 0:
                nca_indices = get_time_dilated_indices(
                    len(nca_frames_data),
                    nca_frames,
                    self.nca_config.initial_repeats,
                    self.nca_config.decay_rate
                )

                smoothing = self.nca_config.temporal_smoothing
                accumulated_nca_frame = None

                print(f"Rendering {len(nca_indices)} NCA frames...")
                for idx in tqdm(nca_indices, desc="NCA Phase"):
                    sca_bg = self.render_frame(sca_data, nca_frame=None, max_depth_limit=None, time=time)
                    nca_fg = self.nca_renderer.render_frame(
                        nca_frames_data[idx], nca_data['source_width'], nca_data['source_height']
                    )

                    if smoothing > 0:
                        nca_fg_float = nca_fg.astype(np.float32)
                        if accumulated_nca_frame is None:
                            accumulated_nca_frame = nca_fg_float
                        else:
                            accumulated_nca_frame = (
                                accumulated_nca_frame * smoothing + nca_fg_float * (1.0 - smoothing)
                            )
                        nca_fg = accumulated_nca_frame.astype(np.uint8)

                    frame = self._composite(sca_bg, nca_fg)
                    writer.append_data(frame if resolve is None else resolve(frame, total))
                    time += dt
                    total += 1

        print(f"Saved combined animation: {output_path}")
        print(f"  Total frames: {total} (SCA: {sca_frames}, NCA: {nca_frames})")
        print(f"  Duration: {total / fps:.2f}s at {fps} fps")
