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
from typing import Dict, Any, Tuple, List
from pathlib import Path

from .config import RenderConfig, NCARenderConfig
from .base import Renderer
from .sca_renderer import SCARenderer
from .nca_renderer import NCARenderer


class CombinedRenderer(Renderer):
    def __init__(self, render_config: RenderConfig = None, nca_config: NCARenderConfig = None):
        super().__init__(render_config or RenderConfig())
        self.nca_config = nca_config or NCARenderConfig()
        
        # Force NCA renderer to be transparent for compositing
        self.nca_config.background_color = (0.0, 0.0, 0.0, 0.0)
        
        # Initialize sub-renderers for drawing logic
        self.sca_renderer = SCARenderer(self.config)
        self.nca_renderer = NCARenderer(self.nca_config)
    
    def _composite(self, bg_img: np.ndarray, fg_img: np.ndarray) -> np.ndarray:
        """Composite foreground over background using alpha blending."""
        fg = fg_img.astype(float) / 255.0
        bg = bg_img.astype(float) / 255.0
        
        alpha_fg = fg[..., 3:4]
        
        # Standard alpha blending
        out_rgb = fg[..., :3] * alpha_fg + bg[..., :3] * (1.0 - alpha_fg)
        
        # Result is opaque
        out = np.dstack((out_rgb, np.ones_like(alpha_fg)))
        return (out * 255).astype(np.uint8)

    def render_frame(self, sca_data: Dict[str, Any], nca_frame: np.ndarray = None, 
                     max_depth_limit: int = None, time: float = 0.0) -> np.ndarray:
        """Render a single combined frame."""
        surface, ctx = self._create_surface()
        
        all_branches = sca_data['branches']
        max_depth = max(b.get('depth', 0) for b in all_branches) if all_branches else 1
        scale_x, scale_y = self._compute_scale(sca_data['source_width'], sca_data['source_height'])
        
        # Draw SCA tree
        self.sca_renderer._draw_branches(ctx, all_branches, scale_x, scale_y, max_depth, max_depth_limit, time=time)
        sca_image = self._surface_to_numpy(surface)
        
        # Draw NCA cells if provided
        if nca_frame is not None:
            nca_image = self.nca_renderer.render_frame(nca_frame, sca_data['source_width'], sca_data['source_height'])
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
        all_branches = sca_data['branches']
        max_depth = max(b.get('depth', 0) for b in all_branches) if all_branches else 1
        
        source_w = sca_data['source_width']
        source_h = sca_data['source_height']
        scale_x = self.config.output_width / source_w
        scale_y = self.config.output_height / source_h
        
        nca_frames_data = nca_data["frames"]
        nca_source_w = nca_data["source_width"]
        nca_source_h = nca_data["source_height"]
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        rendered_frames = []
        
        time = 0.0
        dt = 1.0 / fps
        
        # Phase 1: SCA growth (progressive by depth)
        print(f"Rendering SCA growth phase ({sca_frames} frames)...")
        if sca_frames > 0:
            for i in tqdm(range(sca_frames), desc="SCA phase"):
                # Calculate which depth to show
                t = i / max(sca_frames - 1, 1)
                target_depth = int(t * max_depth)
                
                # Create new surface for this frame (needed for swaying)
                surface, ctx = self._create_surface()
                
                # Draw branches up to target depth
                self.sca_renderer._draw_branches(ctx, all_branches, scale_x, scale_y, max_depth, target_depth, time=time)
                
                rendered_frames.append(self._surface_to_numpy(surface))
                time += dt
        
        # Phase 2: NCA growth on top of full SCA tree
        print(f"Rendering NCA growth phase ({nca_frames} frames)...")
        accumulated_nca_frame = None
        
        if nca_frames > 0:
            # Sample NCA frames evenly
            nca_indices = np.linspace(0, len(nca_frames_data) - 1, nca_frames, dtype=int)
            
            for idx in tqdm(nca_indices, desc="NCA phase"):
                # 1. Render SCA background (with sway)
                surface, ctx = self._create_surface()
                self.sca_renderer._draw_branches(ctx, all_branches, scale_x, scale_y, max_depth, None, time=time)
                sca_bg = self._surface_to_numpy(surface)
                
                # 2. Render NCA overlay
                nca_frame = nca_frames_data[idx]
                nca_fg = self.nca_renderer.render_frame(nca_frame, nca_source_w, nca_source_h)
                
                # Temporal smoothing for NCA layer
                if self.nca_config.temporal_smoothing > 0:
                    nca_fg_float = nca_fg.astype(np.float32)
                    if accumulated_nca_frame is None:
                        accumulated_nca_frame = nca_fg_float
                    else:
                        alpha = 1.0 - self.nca_config.temporal_smoothing
                        accumulated_nca_frame = accumulated_nca_frame * (1.0 - alpha) + nca_fg_float * alpha
                    nca_fg = accumulated_nca_frame.astype(np.uint8)
                
                # 3. Composite
                final_frame = self._composite(sca_bg, nca_fg)
                
                rendered_frames.append(final_frame)
                time += dt
        
        # Save animation
        imageio.mimsave(output_path, rendered_frames, fps=fps)
        print(f"Saved combined animation: {output_path}")
        print(f"  Total frames: {len(rendered_frames)} (SCA: {sca_frames}, NCA: {nca_frames})")
        print(f"  Duration: {len(rendered_frames) / fps:.2f}s at {fps} fps")
