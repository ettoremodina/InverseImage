"""
Combined SCA+NCA renderer.

Creates a single video where:
1. SCA tree grows progressively by depth
2. NCA cells grow on top of the final SCA tree (which remains in background)
"""

import cairo
import numpy as np
import imageio
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
        
        # Initialize sub-renderers for drawing logic
        self.sca_renderer = SCARenderer(self.config)
        self.nca_renderer = NCARenderer(self.nca_config)
    
    def _draw_nca_cells(self, ctx: cairo.Context, frame: np.ndarray, 
                       source_width: int, source_height: int):
        """Draw NCA cells on top of current surface, fully occluding branches beneath."""
        cell_w, cell_h = self.nca_renderer._compute_scale(source_width, source_height)
        
        h, w = frame.shape[:2]
        
        # First pass: draw background rectangles to fully cover branches
        bg_r, bg_g, bg_b, bg_a = self.config.background_color
        for y in range(h):
            for x in range(w):
                r, g, b, a = frame[y, x]
                
                if a < self.nca_config.alpha_threshold:
                    continue
                
                # Draw a full cell-sized background rectangle to occlude branches
                ctx.set_source_rgba(bg_r, bg_g, bg_b, 1.0)
                ctx.rectangle(x * cell_w, y * cell_h, cell_w, cell_h)
                ctx.fill()
        
        # Second pass: draw the actual cells on top
        self.nca_renderer._draw_cells(ctx, frame, source_width, source_height)
    
    def render_frame(self, sca_data: Dict[str, Any], nca_frame: np.ndarray = None, 
                     max_depth_limit: int = None) -> np.ndarray:
        """Render a single combined frame."""
        surface, ctx = self._create_surface()
        
        all_branches = sca_data['branches']
        max_depth = max(b.get('depth', 0) for b in all_branches) if all_branches else 1
        scale_x, scale_y = self._compute_scale(sca_data['source_width'], sca_data['source_height'])
        
        # Draw SCA tree
        self.sca_renderer._draw_branches(ctx, all_branches, scale_x, scale_y, max_depth, max_depth_limit)
        
        # Draw NCA cells if provided
        if nca_frame is not None:
            self._draw_nca_cells(ctx, nca_frame, sca_data['source_width'], sca_data['source_height'])
            
        return self._surface_to_numpy(surface)

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
        
        # Pre-sort branches by depth for faster incremental rendering
        branches_by_depth = {}
        for b in all_branches:
            d = b.get('depth', 0)
            if d not in branches_by_depth:
                branches_by_depth[d] = []
            branches_by_depth[d].append(b)
        
        source_w = sca_data['source_width']
        source_h = sca_data['source_height']
        scale_x = self.config.output_width / source_w
        scale_y = self.config.output_height / source_h
        
        nca_frames_data = nca_data["frames"]
        nca_source_w = nca_data["source_width"]
        nca_source_h = nca_data["source_height"]
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        rendered_frames = []
        
        # Create persistent surface for SCA growth
        # We draw incrementally on this surface
        surface, ctx = self._create_surface()
        current_rendered_depth = -1
        
        # Phase 1: SCA growth (progressive by depth)
        print(f"Rendering SCA growth phase ({sca_frames} frames)...")
        if sca_frames > 0:
            for i in tqdm(range(sca_frames), desc="SCA phase"):
                # Calculate which depth to show
                t = i / max(sca_frames - 1, 1)
                target_depth = int(t * max_depth)
                
                # Draw only new depths incrementally
                if target_depth > current_rendered_depth:
                    for d in range(current_rendered_depth + 1, target_depth + 1):
                        if d in branches_by_depth:
                            self.sca_renderer._draw_branches(ctx, branches_by_depth[d], scale_x, scale_y, max_depth, None)
                    current_rendered_depth = target_depth
                
                rendered_frames.append(self._surface_to_numpy(surface))
        
        # Ensure full tree is drawn before NCA phase
        if current_rendered_depth < max_depth:
            for d in range(current_rendered_depth + 1, max_depth + 1):
                if d in branches_by_depth:
                    self.sca_renderer._draw_branches(ctx, branches_by_depth[d], scale_x, scale_y, max_depth, None)
        
        # Phase 2: NCA growth on top of full SCA tree
        print(f"Rendering NCA growth phase ({nca_frames} frames)...")
        if nca_frames > 0:
            # Sample NCA frames evenly
            nca_indices = np.linspace(0, len(nca_frames_data) - 1, nca_frames, dtype=int)
            
            # The 'surface' now contains the full SCA tree. We use it as a source pattern.
            sca_background_surface = surface
            
            for idx in tqdm(nca_indices, desc="NCA phase"):
                # Create new surface for this frame
                frame_surface, frame_ctx = self._create_surface()
                
                # Blit the cached SCA background (extremely fast)
                frame_ctx.set_source_surface(sca_background_surface, 0, 0)
                frame_ctx.paint()
                
                # Draw NCA cells on top
                nca_frame = nca_frames_data[idx]
                self._draw_nca_cells(frame_ctx, nca_frame, nca_source_w, nca_source_h)
                
                rendered_frames.append(self._surface_to_numpy(frame_surface))
        
        # Save animation
        imageio.mimsave(output_path, rendered_frames, fps=fps)
        print(f"Saved combined animation: {output_path}")
        print(f"  Total frames: {len(rendered_frames)} (SCA: {sca_frames}, NCA: {nca_frames})")
        print(f"  Duration: {len(rendered_frames) / fps:.2f}s at {fps} fps")
