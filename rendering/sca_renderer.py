"""
SCA tree renderer using Cairo.
Demonstrates upsampling: low-res simulation data → high-res rendered output.
"""

import cairo
import numpy as np
import imageio
import math
import multiprocessing
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from pathlib import Path

from config.render_config import SCARenderConfig
from .base import Renderer


def render_sca_frame_wrapper(args):
    config, data, max_depth_limit, time = args
    renderer = SCARenderer(config)
    return renderer.render_frame(data, max_depth_limit=max_depth_limit, time=time)


class SCARenderer(Renderer):
    def __init__(self, config: SCARenderConfig = None):
        super().__init__(config or SCARenderConfig())
    
    def _draw_branches(self, ctx: cairo.Context, branches: List[Dict], 
                       scale_x: float, scale_y: float, max_depth: int, 
                       max_depth_limit: int = None, time: float = 0.0):
        
        r1, g1, b1, a1 = self.config.branch_color
        r2, g2, b2, a2 = self.config.branch_color_end

        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        
        # Build parent map for spline interpolation
        # Map end_point -> branch
        end_point_map = {tuple(b['end']): b for b in branches}
        
        for branch in branches:
            if max_depth_limit is not None and branch.get('depth', 0) > max_depth_limit:
                continue
            
            depth = branch.get('depth', 0)
            t = depth / max_depth if max_depth > 0 else 0
            
            # Color interpolation
            r = r1 + (r2 - r1) * t
            g = g1 + (g2 - g1) * t
            b = b1 + (b2 - b1) * t
            a = a1 + (a2 - a1) * t
            ctx.set_source_rgba(r, g, b, a)

            # Get coordinates
            x_start, y_start = branch['start']
            x_end, y_end = branch['end']
            
            # Find parent to determine previous point
            parent = end_point_map.get(tuple(branch['start']))
            if parent:
                x_prev, y_prev = parent['start']
            else:
                x_prev, y_prev = x_start, y_start # Root case

            # Apply Swaying to all points
            def apply_sway(x, y, d):
                if self.config.sway_magnitude <= 0:
                    return x, y
                
                t_sway = d / max_depth if max_depth > 0 else 0
                sway_amount = self.config.sway_magnitude * (t_sway ** 2)
                phase = d * 0.2 + y * 0.05
                offset_x = math.sin(time * self.config.sway_frequency + phase) * sway_amount
                return x + offset_x, y

            # We need depths for sway calculation
            d_curr = depth
            d_prev = parent.get('depth', 0) if parent else 0
            d_next = d_curr + 1 # Approximate for end point
            
            px, py = apply_sway(x_prev, y_prev, d_prev)
            sx, sy = apply_sway(x_start, y_start, d_curr)
            ex, ey = apply_sway(x_end, y_end, d_next)
            
            # Scale points
            px, py = px * scale_x, py * scale_y
            sx, sy = sx * scale_x, sy * scale_y
            ex, ey = ex * scale_x, ey * scale_y
            
            # Calculate midpoints for Chaikin/Bezier smoothing
            # We draw from Mid(Prev, Start) to Mid(Start, End)
            # Control point is Start
            
            if parent:
                mx1, my1 = (px + sx) / 2, (py + sy) / 2
            else:
                mx1, my1 = sx, sy # Start at root
                
            mx2, my2 = (sx + ex) / 2, (sy + ey) / 2
            
            width = self.config.branch_base_width * (1 - t) + self.config.branch_tip_width * t
            ctx.set_line_width(width)
            
            # Draw the main curve segment for this branch (covering the joint at Start)
            ctx.move_to(mx1, my1)
            if parent:
                ctx.curve_to(sx, sy, sx, sy, mx2, my2) # Quadratic bezier using cubic command (CP1=CP2=Control)
                # Actually for quadratic P0->P2 with control P1:
                # Cubic equivalent: CP1 = P0 + 2/3(P1-P0), CP2 = P2 + 2/3(P1-P2)
                cp1x = mx1 + (2/3) * (sx - mx1)
                cp1y = my1 + (2/3) * (sy - my1)
                cp2x = mx2 + (2/3) * (sx - mx2)
                cp2y = my2 + (2/3) * (sy - my2)
                ctx.curve_to(cp1x, cp1y, cp2x, cp2y, mx2, my2)
            else:
                ctx.line_to(mx2, my2) # Straight line from root to first mid
            
            ctx.stroke()
            
            # If this is a tip (or we are at the limit), we need to finish the segment
            # Draw from Mid(Start, End) to End
            # We can just draw a straight line for the tip segment, or curve it if we had next point
            # Since it's a tip, straight line is fine.
            
            # Check if it's a tip in the full tree OR if we hit the render limit
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
        """
        Render growth animation by progressively revealing branches by depth.
        """
        all_branches = data['branches']
        max_depth = max(b.get('depth', 0) for b in all_branches) if all_branches else 1
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        frames = []
        depths_to_render = list(range(0, max_depth + 1, frame_skip))
        
        # Prepare tasks
        tasks = []
        time = 0.0
        dt = 1.0 / fps
        
        for depth in depths_to_render:
            tasks.append((self.config, data, depth, time))
            time += dt
        
        # Add final frame
        tasks.append((self.config, data, None, time))
        
        # Parallel rendering
        num_cores = max(1, multiprocessing.cpu_count() - 1)
        print(f"Rendering with {num_cores} cores...")
        
        with multiprocessing.Pool(processes=num_cores) as pool:
            frames = list(tqdm(pool.imap(render_sca_frame_wrapper, tasks), total=len(tasks), desc="Rendering SCA frames (Parallel)"))
        
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"  Saved animation: {output_path}")
