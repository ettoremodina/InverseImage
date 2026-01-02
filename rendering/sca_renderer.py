"""
SCA tree renderer using Cairo.
Demonstrates upsampling: low-res simulation data → high-res rendered output.
"""

import cairo
import numpy as np
import imageio
import math
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from config.render_config import SCARenderConfig
from .base import Renderer


class SCARenderer(Renderer):
    def __init__(self, config: SCARenderConfig = None):
        super().__init__(config or SCARenderConfig())
    
    def _draw_branches(self, ctx: cairo.Context, branches: List[Dict], 
                       scale_x: float, scale_y: float, max_depth: int, 
                       max_depth_limit: int = None, time: float = 0.0):
        
        r1, g1, b1, a1 = self.config.branch_color
        r2, g2, b2, a2 = self.config.branch_color_end
        sway_mag = self.config.sway_magnitude
        sway_freq = self.config.sway_frequency
        base_w = self.config.branch_base_width
        tip_w = self.config.branch_tip_width

        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        
        end_point_map = {tuple(b['end']): b for b in branches}
        inv_max_depth = 1.0 / max_depth if max_depth > 0 else 0
        
        use_sway = sway_mag > 0
        
        for branch in branches:
            depth = branch.get('depth', 0)
            if max_depth_limit is not None and depth > max_depth_limit:
                continue
            
            t = depth * inv_max_depth
            
            r = r1 + (r2 - r1) * t
            g = g1 + (g2 - g1) * t
            b = b1 + (b2 - b1) * t
            a = a1 + (a2 - a1) * t
            ctx.set_source_rgba(r, g, b, a)

            x_start, y_start = branch['start']
            x_end, y_end = branch['end']
            
            parent = end_point_map.get(tuple(branch['start']))
            if parent:
                x_prev, y_prev = parent['start']
                d_prev = parent.get('depth', 0)
            else:
                x_prev, y_prev = x_start, y_start
                d_prev = 0

            if use_sway:
                t_prev = d_prev * inv_max_depth
                s_a_prev = sway_mag * (t_prev * t_prev)
                px = (x_prev + math.sin(time * sway_freq + d_prev * 0.2 + y_prev * 0.05) * s_a_prev) * scale_x
                py = y_prev * scale_y
                
                s_a = sway_mag * (t * t)
                sx = (x_start + math.sin(time * sway_freq + depth * 0.2 + y_start * 0.05) * s_a) * scale_x
                sy = y_start * scale_y
                
                t_next = (depth + 1) * inv_max_depth
                s_a_next = sway_mag * (t_next * t_next)
                ex = (x_end + math.sin(time * sway_freq + (depth + 1) * 0.2 + y_end * 0.05) * s_a_next) * scale_x
                ey = y_end * scale_y
            else:
                px, py = x_prev * scale_x, y_prev * scale_y
                sx, sy = x_start * scale_x, y_start * scale_y
                ex, ey = x_end * scale_x, y_end * scale_y
            
            if parent:
                mx1, my1 = (px + sx) * 0.5, (py + sy) * 0.5
            else:
                mx1, my1 = sx, sy
                
            mx2, my2 = (sx + ex) * 0.5, (sy + ey) * 0.5
            
            width = base_w + (tip_w - base_w) * t
            ctx.set_line_width(width)
            
            ctx.move_to(mx1, my1)
            if parent:
                cp1x = mx1 + 0.6666666666666666 * (sx - mx1)
                cp1y = my1 + 0.6666666666666666 * (sy - my1)
                cp2x = mx2 + 0.6666666666666666 * (sx - mx2)
                cp2y = my2 + 0.6666666666666666 * (sy - my2)
                ctx.curve_to(cp1x, cp1y, cp2x, cp2y, mx2, my2)
            else:
                ctx.line_to(mx2, my2)
            
            ctx.stroke()
            
            is_visual_tip = branch.get('is_tip', False) or (max_depth_limit is not None and depth == max_depth_limit)
            
            if is_visual_tip:
                ctx.move_to(mx2, my2)
                ctx.line_to(ex, ey)
                ctx.stroke()
    
    def render_frame(self, data: Dict[str, Any], max_depth_limit: int = None, time: float = 0.0) -> np.ndarray:
        surface, ctx = self._create_surface()
        
        source_w = data['source_width']
        source_h = data['source_height']
        scale_x, scale_y = self._compute_scale(source_w, source_h)
        
        all_branches = data['branches']
        global_max_depth = max(b.get('depth', 0) for b in all_branches) if all_branches else 1
        
        self._draw_branches(ctx, all_branches, scale_x, scale_y, global_max_depth, max_depth_limit, time=time)
        
        return self._surface_to_numpy(surface)
    
    def save_frame(self, data: Dict[str, Any], output_path: str):
        frame = self.render_frame(data)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(output_path, frame)
    
    def render_animation(self, data: Dict[str, Any], output_path: str, 
                         fps: int = 30, frame_skip: int = 1):
        """Render growth animation by progressively revealing branches by depth."""
        all_branches = data['branches']
        max_depth = max(b.get('depth', 0) for b in all_branches) if all_branches else 1
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        frames = []
        depths_to_render = list(range(0, max_depth + 1, frame_skip))
        
        time = 0.0
        dt = 1.0 / fps
        
        print("Rendering SCA frames...")
        for depth in tqdm(depths_to_render, desc="Rendering SCA frames"):
            frames.append(self.render_frame(data, max_depth_limit=depth, time=time))
            time += dt
        
        frames.append(self.render_frame(data, max_depth_limit=None, time=time))
        
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"  Saved animation: {output_path}")
