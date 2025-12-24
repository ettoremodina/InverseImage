"""
NCA renderer using Cairo.
Upsamples low-res NCA frames to high-res with configurable cell shapes.

Future extensions: jitter, glow, spawn animations, texture fills.
"""

import cairo
import numpy as np
import imageio
from tqdm import tqdm
from typing import Dict, Any, Tuple
from pathlib import Path

from .config import NCARenderConfig
from .base import Renderer


class NCARenderer(Renderer):
    def __init__(self, config: NCARenderConfig = None):
        super().__init__(config or NCARenderConfig())
    
    def _draw_cell_circle(self, ctx: cairo.Context, cx: float, cy: float, 
                          radius: float, r: float, g: float, b: float, a: float):
        ctx.set_source_rgba(r, g, b, a)
        ctx.arc(cx, cy, radius, 0, 2 * np.pi)
        ctx.fill()
    
    def _draw_cell_square(self, ctx: cairo.Context, cx: float, cy: float,
                          size: float, r: float, g: float, b: float, a: float):
        ctx.set_source_rgba(r, g, b, a)
        half = size / 2
        ctx.rectangle(cx - half, cy - half, size, size)
        ctx.fill()
    
    def _draw_cells(self, ctx: cairo.Context, frame: np.ndarray, source_width: int, source_height: int):
        """Draw NCA cells on the given context."""
        cell_w, cell_h = self._compute_scale(source_width, source_height)
        cell_size = min(cell_w, cell_h)
        radius = (cell_size / 2) * self.config.cell_scale
        
        h, w = frame.shape[:2]
        
        for y in range(h):
            for x in range(w):
                r, g, b, a = frame[y, x]
                
                if a < self.config.alpha_threshold:
                    continue
                
                cx = (x + 0.5) * cell_w
                cy = (y + 0.5) * cell_h
                
                if self.config.cell_shape == "circle":
                    draw_radius = radius * (0.5 + 0.5 * a)
                    self._draw_cell_circle(ctx, cx, cy, draw_radius, r, g, b, a)
                else:
                    draw_size = cell_size * self.config.cell_scale * (0.5 + 0.5 * a)
                    self._draw_cell_square(ctx, cx, cy, draw_size, r, g, b, a)

    def render_frame(self, frame: np.ndarray, source_width: int, source_height: int) -> np.ndarray:
        """
        Render a single NCA frame.
        
        Args:
            frame: RGBA array of shape [H, W, 4], values in [0, 1]
            source_width, source_height: original grid dimensions
        
        Returns:
            RGBA numpy array of shape [output_height, output_width, 4]
        """
        surface, ctx = self._create_surface()
        self._draw_cells(ctx, frame, source_width, source_height)
        return self._surface_to_numpy(surface)
    
    def render_animation(self, data: Dict[str, Any], output_path: str, fps: int = 30):
        """
        Render full NCA animation from loaded data.
        
        Args:
            data: Dict from load_nca_frames() with keys: frames, source_width, source_height
            output_path: Path for output video (mp4)
            fps: Frames per second
        """
        frames_data = data["frames"]
        source_w = data["source_width"]
        source_h = data["source_height"]
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        rendered_frames = []
        for frame in tqdm(frames_data, desc="Rendering NCA frames"):
            rendered = self.render_frame(frame, source_w, source_h)
            rendered_frames.append(rendered)
        
        imageio.mimsave(output_path, rendered_frames, fps=fps)
        print(f"Saved animation: {output_path}")
    
    def save_frame(self, frame: np.ndarray, source_width: int, source_height: int, 
                   output_path: str):
        """Render and save a single frame as PNG."""
        rendered = self.render_frame(frame, source_width, source_height)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(output_path, rendered)
