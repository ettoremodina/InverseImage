"""
Combined SCA+NCA renderer.

Creates a single video where:
1. SCA tree grows progressively by depth
2. NCA cells grow on top of the final SCA tree (which remains in background)
"""

import cairo
import numpy as np
import imageio
import cv2
from tqdm import tqdm
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path

from config.render_config import SCARenderConfig, NCARenderConfig
from .base import Renderer
from .sca_renderer import SCARenderer
from .nca_renderer import NCARenderer
from .utils import get_time_dilated_indices


class CombinedRenderer(Renderer):
    def __init__(self, render_config: SCARenderConfig = None, nca_config: NCARenderConfig = None):
        super().__init__(render_config or SCARenderConfig())
        self.nca_config = nca_config or NCARenderConfig()
        
        self.nca_config.background_color = (0.0, 0.0, 0.0, 0.0)
        
        self.sca_renderer = SCARenderer(self.config)
        self.nca_renderer = NCARenderer(self.nca_config)
        
        self._cached_sca_data = None
        self._cached_max_depth = None
        self._cached_scale = None
    
    def _composite_fast(self, bg_img: np.ndarray, fg_img: np.ndarray) -> np.ndarray:
        """Fast alpha compositing using pre-multiplied alpha."""
        alpha_fg = fg_img[..., 3:4].astype(np.float32) / 255.0
        
        out = bg_img.astype(np.float32)
        out[..., :3] = fg_img[..., :3] * alpha_fg + bg_img[..., :3] * (1.0 - alpha_fg)
        out[..., 3] = 255
        
        return out.astype(np.uint8)
    
    def _composite(self, bg_img: np.ndarray, fg_img: np.ndarray) -> np.ndarray:
        return self._composite_fast(bg_img, fg_img)
    
    def _cache_sca_metadata(self, sca_data: Dict[str, Any]):
        """Cache SCA metadata to avoid recomputation."""
        if self._cached_sca_data is not sca_data:
            self._cached_sca_data = sca_data
            all_branches = sca_data['branches']
            self._cached_max_depth = max(b.get('depth', 0) for b in all_branches) if all_branches else 1
            self._cached_scale = self._compute_scale(sca_data['source_width'], sca_data['source_height'])

    def render_frame(self, sca_data: Dict[str, Any], nca_frame: np.ndarray = None, 
                     max_depth_limit: int = None, time: float = 0.0) -> np.ndarray:
        """Render a single combined frame."""
        self._cache_sca_metadata(sca_data)
        
        surface, ctx = self._create_surface()
        
        all_branches = sca_data['branches']
        scale_x, scale_y = self._cached_scale
        
        self.sca_renderer._draw_branches(
            ctx, all_branches, scale_x, scale_y, 
            self._cached_max_depth, max_depth_limit, time=time
        )
        sca_image = self._surface_to_numpy(surface)
        
        if nca_frame is not None:
            h, w = nca_frame.shape[:2]
            nca_image = self.nca_renderer.render_frame(nca_frame, w, h)
            return self._composite(sca_image, nca_image)
            
        return sca_image

    def render_animation(self, sca_data: Dict[str, Any], nca_data: Dict[str, Any],
                         output_path: str, fps: int, sca_frames: int, nca_frames: int):
        """
        Render combined SCA->NCA animation.
        
        Args:
            sca_data: SCA render data with branches
            nca_data: NCA frames data
            output_path: Output video path
            fps: Frames per second
            sca_frames: Number of frames for SCA growth phase
            nca_frames: Number of frames for NCA growth phase
        """
        self._cache_sca_metadata(sca_data)
        
        all_branches = sca_data['branches']
        max_depth = self._cached_max_depth
        
        nca_frames_data = nca_data["frames"]
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        rendered_frames = []
        
        time = 0.0
        dt = 1.0 / fps
        
        # Phase 1: SCA growth
        if sca_frames > 0:
            print(f"Rendering {sca_frames} SCA frames...")
            for i in tqdm(range(sca_frames), desc="SCA Phase"):
                t_frac = i / max(sca_frames - 1, 1)
                target_depth = int(t_frac * max_depth)
                rendered_frames.append(self.render_frame(sca_data, nca_frame=None, max_depth_limit=target_depth, time=time))
                time += dt
        
        # Phase 2: NCA growth
        if self.nca_config.temporal_smoothing <= 0 and nca_frames > 0:
            nca_indices = get_time_dilated_indices(
                len(nca_frames_data), 
                nca_frames, 
                self.nca_config.initial_repeats, 
                self.nca_config.decay_rate
            )
            
            print(f"Rendering {len(nca_indices)} NCA frames...")
            for idx in tqdm(nca_indices, desc="NCA Phase"):
                nca_frame = nca_frames_data[idx]
                rendered_frames.append(self.render_frame(sca_data, nca_frame=nca_frame, max_depth_limit=None, time=time))
                time += dt
        
        # Phase 2 with Smoothing (Sequential - temporal dependency)
        if self.nca_config.temporal_smoothing > 0 and nca_frames > 0:
            print(f"Rendering NCA phase (sequential due to smoothing)...")
            accumulated_nca_frame = None
            nca_indices = get_time_dilated_indices(
                len(nca_frames_data), 
                nca_frames, 
                self.nca_config.initial_repeats, 
                self.nca_config.decay_rate
            )
            
            scale_x, scale_y = self._cached_scale
            
            for idx in tqdm(nca_indices, desc="NCA Phase"):
                surface, ctx = self._create_surface()
                self.sca_renderer._draw_branches(ctx, all_branches, scale_x, scale_y, max_depth, None, time=time)
                sca_bg = self._surface_to_numpy(surface)
                
                nca_frame = nca_frames_data[idx]
                nca_fg = self.nca_renderer.render_frame(nca_frame, nca_data['source_width'], nca_data['source_height'])
                
                nca_fg_float = nca_fg.astype(np.float32)
                if accumulated_nca_frame is None:
                    accumulated_nca_frame = nca_fg_float
                else:
                    alpha = 1.0 - self.nca_config.temporal_smoothing
                    accumulated_nca_frame = accumulated_nca_frame * (1.0 - alpha) + nca_fg_float * alpha
                nca_fg = accumulated_nca_frame.astype(np.uint8)
                
                final_frame = self._composite(sca_bg, nca_fg)
                rendered_frames.append(final_frame)
                time += dt
        
        imageio.mimsave(output_path, rendered_frames, fps=fps)
        print(f"Saved combined animation: {output_path}")
        print(f"  Total frames: {len(rendered_frames)} (SCA: {sca_frames}, NCA: {nca_frames})")
        print(f"  Duration: {len(rendered_frames) / fps:.2f}s at {fps} fps")
        
        imageio.mimsave(output_path, rendered_frames, fps=fps)
        print(f"Saved combined animation: {output_path}")
        print(f"  Total frames: {len(rendered_frames)} (SCA: {sca_frames}, NCA: {nca_frames})")
        print(f"  Duration: {len(rendered_frames) / fps:.2f}s at {fps} fps")
