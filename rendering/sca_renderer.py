"""
SCA tree renderer using Cairo.
Demonstrates upsampling: low-res simulation data → high-res rendered output.
"""

import cairo
import numpy as np
import imageio
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from pathlib import Path

from .config import RenderConfig


class SCARenderer:
    def __init__(self, config: RenderConfig = None):
        self.config = config or RenderConfig()
    
    def _create_surface(self) -> Tuple[cairo.ImageSurface, cairo.Context]:
        surface = cairo.ImageSurface(
            cairo.FORMAT_ARGB32,
            self.config.output_width,
            self.config.output_height
        )
        ctx = cairo.Context(surface)
        
        if self.config.antialiasing:
            ctx.set_antialias(cairo.ANTIALIAS_BEST)
        
        r, g, b, a = self.config.background_color
        ctx.set_source_rgba(r, g, b, a)
        ctx.paint()
        
        return surface, ctx
    
    def _compute_scale(self, source_width: int, source_height: int) -> Tuple[float, float]:
        scale_x = self.config.output_width / source_width
        scale_y = self.config.output_height / source_height
        return scale_x, scale_y
    
    def _draw_branches(self, ctx: cairo.Context, branches: List[Dict], 
                       scale_x: float, scale_y: float, max_depth: int):
        r, g, b, a = self.config.branch_color
        ctx.set_source_rgba(r, g, b, a)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        
        for branch in branches:
            x1, y1 = branch['start']
            x2, y2 = branch['end']
            
            x1_scaled = x1 * scale_x
            y1_scaled = y1 * scale_y
            x2_scaled = x2 * scale_x
            y2_scaled = y2 * scale_y
            
            depth = branch.get('depth', 0)
            t = depth / max_depth if max_depth > 0 else 0
            width = self.config.branch_base_width * (1 - t) + self.config.branch_tip_width * t
            
            ctx.set_line_width(width)
            ctx.move_to(x1_scaled, y1_scaled)
            ctx.line_to(x2_scaled, y2_scaled)
            ctx.stroke()
    
    def render_frame(self, data: Dict[str, Any], max_depth_limit: int = None) -> np.ndarray:
        surface, ctx = self._create_surface()
        
        source_w = data['source_width']
        source_h = data['source_height']
        scale_x, scale_y = self._compute_scale(source_w, source_h)
        
        all_branches = data['branches']
        global_max_depth = max(b.get('depth', 0) for b in all_branches) if all_branches else 1
        
        if max_depth_limit is not None:
            branches = [b for b in all_branches if b.get('depth', 0) <= max_depth_limit]
        else:
            branches = all_branches
        
        self._draw_branches(ctx, branches, scale_x, scale_y, global_max_depth)
        
        return self._surface_to_numpy(surface)
    
    def _surface_to_numpy(self, surface: cairo.ImageSurface) -> np.ndarray:
        buf = surface.get_data()
        arr = np.ndarray(
            shape=(self.config.output_height, self.config.output_width, 4),
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
    
    def save_frame(self, data: Dict[str, Any], output_path: str):
        frame = self.render_frame(data)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(output_path, frame)
    
    def render_animation(self, data: Dict[str, Any], output_path: str, 
                         fps: int = 30, frame_skip: int = 1):
        """
        Render growth animation by progressively revealing branches by depth.
        """
        all_branches = data['branches']
        max_depth = max(b.get('depth', 0) for b in all_branches) if all_branches else 1
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        frames = []
        depths_to_render = list(range(0, max_depth + 1, frame_skip))
        
        for depth in tqdm(depths_to_render, desc="Rendering frames"):
            frame = self.render_frame(data, max_depth_limit=depth)
            frames.append(frame)
        
        frames.append(self.render_frame(data))
        
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"  Saved animation: {output_path}")
