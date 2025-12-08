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

from .config import RenderConfig
from .nca_renderer import NCARenderConfig


class CombinedRenderer:
    def __init__(self, render_config: RenderConfig = None, nca_config: NCARenderConfig = None):
        self.render_config = render_config or RenderConfig()
        self.nca_config = nca_config or NCARenderConfig()
    
    def _create_surface(self) -> Tuple[cairo.ImageSurface, cairo.Context]:
        surface = cairo.ImageSurface(
            cairo.FORMAT_ARGB32,
            self.render_config.output_width,
            self.render_config.output_height
        )
        ctx = cairo.Context(surface)
        
        if self.render_config.antialiasing:
            ctx.set_antialias(cairo.ANTIALIAS_BEST)
        
        r, g, b, a = self.render_config.background_color
        ctx.set_source_rgba(r, g, b, a)
        ctx.paint()
        
        return surface, ctx
    
    def _draw_sca_branches(self, ctx: cairo.Context, branches: List[Dict], 
                          scale_x: float, scale_y: float, max_depth: int, max_depth_limit: int = None):
        """Draw SCA branches up to max_depth_limit."""
        r, g, b, a = self.render_config.branch_color
        ctx.set_source_rgba(r, g, b, a)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        
        for branch in branches:
            if max_depth_limit is not None and branch.get('depth', 0) > max_depth_limit:
                continue
            
            x1, y1 = branch['start']
            x2, y2 = branch['end']
            
            x1_scaled = x1 * scale_x
            y1_scaled = y1 * scale_y
            x2_scaled = x2 * scale_x
            y2_scaled = y2 * scale_y
            
            depth = branch.get('depth', 0)
            t = depth / max_depth if max_depth > 0 else 0
            width = self.render_config.branch_base_width * (1 - t) + self.render_config.branch_tip_width * t
            
            ctx.set_line_width(width)
            ctx.move_to(x1_scaled, y1_scaled)
            ctx.line_to(x2_scaled, y2_scaled)
            ctx.stroke()
    
    def _draw_nca_cells(self, ctx: cairo.Context, frame: np.ndarray, 
                       source_width: int, source_height: int):
        """Draw NCA cells on top of current surface, fully occluding branches beneath."""
        cell_w = self.nca_config.output_width / source_width
        cell_h = self.nca_config.output_height / source_height
        cell_size = min(cell_w, cell_h)
        radius = (cell_size / 2) * self.nca_config.cell_scale
        
        h, w = frame.shape[:2]
        
        # First pass: draw background rectangles to fully cover branches
        bg_r, bg_g, bg_b, bg_a = self.render_config.background_color
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
        for y in range(h):
            for x in range(w):
                r, g, b, a = frame[y, x]
                
                if a < self.nca_config.alpha_threshold:
                    continue
                
                cx = (x + 0.5) * cell_w
                cy = (y + 0.5) * cell_h
                
                if self.nca_config.cell_shape == "circle":
                    draw_radius = radius * (0.5 + 0.5 * a)
                    ctx.set_source_rgba(r, g, b, 1.0)
                    ctx.arc(cx, cy, draw_radius, 0, 2 * np.pi)
                    ctx.fill()
                else:
                    draw_size = cell_size * self.nca_config.cell_scale * (0.5 + 0.5 * a)
                    ctx.set_source_rgba(r, g, b, 1.0)
                    half = draw_size / 2
                    ctx.rectangle(cx - half, cy - half, draw_size, draw_size)
                    ctx.fill()
    
    def _surface_to_numpy(self, surface: cairo.ImageSurface) -> np.ndarray:
        buf = surface.get_data()
        arr = np.ndarray(
            shape=(self.render_config.output_height, self.render_config.output_width, 4),
            dtype=np.uint8,
            buffer=buf
        )
        arr_copy = arr.copy()
        arr_rgba = np.zeros_like(arr_copy)
        arr_rgba[:, :, 0] = arr_copy[:, :, 2]  # R
        arr_rgba[:, :, 1] = arr_copy[:, :, 1]  # G
        arr_rgba[:, :, 2] = arr_copy[:, :, 0]  # B
        arr_rgba[:, :, 3] = arr_copy[:, :, 3]  # A
        return arr_rgba
    
    def render_combined_animation(self, sca_data: Dict[str, Any], nca_data: Dict[str, Any],
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
        scale_x = self.render_config.output_width / source_w
        scale_y = self.render_config.output_height / source_h
        
        nca_frames_data = nca_data["frames"]
        nca_source_w = nca_data["source_width"]
        nca_source_h = nca_data["source_height"]
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        rendered_frames = []
        
        # Phase 1: SCA growth (progressive by depth)
        print(f"Rendering SCA growth phase ({sca_frames} frames)...")
        if sca_frames > 0:
            for i in tqdm(range(sca_frames), desc="SCA phase"):
                surface, ctx = self._create_surface()
                
                # Calculate which depth to show
                t = i / max(sca_frames - 1, 1)
                current_depth = int(t * max_depth)
                
                self._draw_sca_branches(ctx, all_branches, scale_x, scale_y, max_depth, current_depth)
                
                rendered_frames.append(self._surface_to_numpy(surface))
        
        # Phase 2: NCA growth on top of full SCA tree
        print(f"Rendering NCA growth phase ({nca_frames} frames)...")
        if nca_frames > 0:
            # Sample NCA frames evenly
            nca_indices = np.linspace(0, len(nca_frames_data) - 1, nca_frames, dtype=int)
            
            for idx in tqdm(nca_indices, desc="NCA phase"):
                surface, ctx = self._create_surface()
                
                # Draw full SCA tree in background
                self._draw_sca_branches(ctx, all_branches, scale_x, scale_y, max_depth, None)
                
                # Draw NCA cells on top
                nca_frame = nca_frames_data[idx]
                self._draw_nca_cells(ctx, nca_frame, nca_source_w, nca_source_h)
                
                rendered_frames.append(self._surface_to_numpy(surface))
        
        # Save animation
        imageio.mimsave(output_path, rendered_frames, fps=fps)
        print(f"Saved combined animation: {output_path}")
        print(f"  Total frames: {len(rendered_frames)} (SCA: {sca_frames}, NCA: {nca_frames})")
        print(f"  Duration: {len(rendered_frames) / fps:.2f}s at {fps} fps")
