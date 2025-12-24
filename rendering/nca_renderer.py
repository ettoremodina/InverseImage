"""
NCA renderer using Cairo.
Upsamples low-res NCA frames to high-res with configurable cell shapes.
"""

import cairo
import numpy as np
import imageio
from tqdm import tqdm
from typing import Dict, Any, List, Optional
from pathlib import Path

from config.render_config import NCARenderConfig
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
        
        base_scale = self.config.cell_scale
        radius = (cell_size / 2) * base_scale
        
        # Optimization: Vectorize alpha check to avoid iterating over empty space
        # Get indices where alpha >= threshold
        ys, xs = np.where(frame[..., 3] >= self.config.alpha_threshold)
        
        if len(ys) == 0:
            return

        for y, x in zip(ys, xs):
            r, g, b, a = frame[y, x]
            
            cx = (x + 0.5) * cell_w
            cy = (y + 0.5) * cell_h
            
            if self.config.cell_shape == "circle":
                self._draw_cell_circle(ctx, cx, cy, radius, r, g, b, a)
            else:
                draw_size = cell_size * base_scale
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
    
    def _calculate_frame_repeats(self, frame_idx: int, total_frames: int) -> int:
        """
        Calculate how many times a frame should be repeated based on its index.
        Uses an exponential decay curve to make early frames persist longer.
        """
        # Initial repeats for the first frame
        initial_repeats = 30 
        # Decay rate - lower means faster drop off
        decay_rate = 0.9
        
        repeats = int(initial_repeats * (decay_rate ** frame_idx))
        return max(1, repeats)

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
        
        base_rendered_frames = []
        
        print("Rendering NCA frames...")
        for frame in tqdm(frames_data, desc="Rendering NCA frames"):
            base_rendered_frames.append(self.render_frame(frame, source_w, source_h))
        
        # Apply temporal smoothing if enabled
        if self.config.temporal_smoothing > 0:
            print("Applying temporal smoothing...")
            smoothed_frames = []
            accumulated_frame = None
            for frame in base_rendered_frames:
                frame_float = frame.astype(np.float32)
                if accumulated_frame is None:
                    accumulated_frame = frame_float
                else:
                    alpha = 1.0 - self.config.temporal_smoothing
                    accumulated_frame = accumulated_frame * (1.0 - alpha) + frame_float * alpha
                smoothed_frames.append(accumulated_frame.astype(np.uint8))
            base_rendered_frames = smoothed_frames

        # Apply frame persistence (time dilation)
        final_frames = []
        print("Applying frame persistence...")
        for i, frame in enumerate(base_rendered_frames):
            repeats = self._calculate_frame_repeats(i, len(base_rendered_frames))
            for _ in range(repeats):
                final_frames.append(frame)
        
        imageio.mimsave(output_path, final_frames, fps=fps)
        print(f"Saved animation: {output_path}")
    
    def save_frame(self, frame: np.ndarray, source_width: int, source_height: int, 
                   output_path: str):
        """Render and save a single frame as PNG."""
        rendered = self.render_frame(frame, source_width, source_height)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(output_path, rendered)
