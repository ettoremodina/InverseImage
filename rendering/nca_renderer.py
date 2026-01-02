"""
NCA renderer using Cairo.
Upsamples low-res NCA frames to high-res with configurable cell shapes.
"""

import cairo
import numpy as np
import imageio
import cv2
from tqdm import tqdm
from typing import Dict, Any, List, Optional
from pathlib import Path

from config.render_config import NCARenderConfig
from .base import Renderer


class NCARenderer(Renderer):
    def __init__(self, config: NCARenderConfig = None):
        super().__init__(config or NCARenderConfig())
        self._circle_mask_cache = {}
    
    def _get_circle_mask(self, radius: int) -> np.ndarray:
        """Get cached circular mask for given radius."""
        if radius not in self._circle_mask_cache:
            size = radius * 2 + 1
            y, x = np.ogrid[:size, :size]
            center = radius
            mask = ((x - center) ** 2 + (y - center) ** 2) <= radius ** 2
            self._circle_mask_cache[radius] = mask
        return self._circle_mask_cache[radius]
    
    def _draw_cells_fast(self, surface: cairo.ImageSurface, frame: np.ndarray, 
                         source_width: int, source_height: int):
        """Fast numpy-based cell drawing directly to surface buffer."""
        cell_w, cell_h = self._compute_scale(source_width, source_height)
        cell_size = min(cell_w, cell_h)
        base_scale = self.config.cell_scale
        radius = int((cell_size / 2) * base_scale)
        
        if radius < 1:
            radius = 1
        
        ys, xs = np.where(frame[..., 3] >= self.config.alpha_threshold)
        if len(ys) == 0:
            return
        
        buf = surface.get_data()
        arr = np.ndarray(
            shape=(self.config.output_height, self.config.output_width, 4),
            dtype=np.uint8,
            buffer=buf
        )
        
        out_h, out_w = self.config.output_height, self.config.output_width
        circle_mask = self._get_circle_mask(radius)
        mask_size = radius * 2 + 1
        
        for y, x in zip(ys, xs):
            r, g, b, a = frame[y, x]
            
            cx = int((x + 0.5) * cell_w)
            cy = int((y + 0.5) * cell_h)
            
            x1, y1 = cx - radius, cy - radius
            x2, y2 = x1 + mask_size, y1 + mask_size
            
            # Clip to bounds
            mx1 = max(0, -x1)
            my1 = max(0, -y1)
            mx2 = mask_size - max(0, x2 - out_w)
            my2 = mask_size - max(0, y2 - out_h)
            
            ox1 = max(0, x1)
            oy1 = max(0, y1)
            ox2 = min(out_w, x2)
            oy2 = min(out_h, y2)
            
            if ox1 >= ox2 or oy1 >= oy2:
                continue
            
            region_mask = circle_mask[my1:my2, mx1:mx2]
            
            # BGRA format for Cairo, alpha blend
            colors = np.array([b * 255, g * 255, r * 255, a * 255], dtype=np.uint8)
            
            target = arr[oy1:oy2, ox1:ox2]
            alpha_f = a
            
            for c in range(4):
                target[:, :, c] = np.where(
                    region_mask,
                    (colors[c] * alpha_f + target[:, :, c] * (1 - alpha_f)).astype(np.uint8),
                    target[:, :, c]
                )
        
        surface.mark_dirty()
    
    def _draw_cells_cairo(self, ctx: cairo.Context, frame: np.ndarray, 
                          source_width: int, source_height: int):
        """Cairo-based cell drawing (fallback for squares or high quality)."""
        cell_w, cell_h = self._compute_scale(source_width, source_height)
        cell_size = min(cell_w, cell_h)
        base_scale = self.config.cell_scale
        radius = (cell_size / 2) * base_scale
        
        ys, xs = np.where(frame[..., 3] >= self.config.alpha_threshold)
        if len(ys) == 0:
            return
        
        if self.config.cell_shape == "circle":
            for y, x in zip(ys, xs):
                r, g, b, a = frame[y, x]
                cx = (x + 0.5) * cell_w
                cy = (y + 0.5) * cell_h
                ctx.set_source_rgba(r, g, b, a)
                ctx.arc(cx, cy, radius, 0, 2 * np.pi)
                ctx.fill()
        else:
            draw_size = cell_size * base_scale
            half = draw_size / 2
            for y, x in zip(ys, xs):
                r, g, b, a = frame[y, x]
                cx = (x + 0.5) * cell_w
                cy = (y + 0.5) * cell_h
                ctx.set_source_rgba(r, g, b, a)
                ctx.rectangle(cx - half, cy - half, draw_size, draw_size)
                ctx.fill()

    def render_frame(self, frame: np.ndarray, source_width: int, source_height: int, 
                     use_fast_path: bool = True) -> np.ndarray:
        """
        Render a single NCA frame using fast pixel upscaling.
        Treats each cell as a square block of pixels.
        """
        out_h, out_w = self.config.output_height, self.config.output_width
        
        # 1. Resize (Upsample) using Nearest Neighbor to keep sharp pixels
        # frame is [H, W, 4] float 0-1
        resized = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        
        # 2. Apply Alpha Threshold
        mask = resized[..., 3] < self.config.alpha_threshold
        resized[mask, 3] = 0
        
        # 3. Convert to uint8 [0-255]
        resized_uint8 = (resized * 255).astype(np.uint8)
        
        # 4. Handle Background
        bg_color = self.config.background_color
        if bg_color[3] == 0:
            return resized_uint8
            
        # Composite over background if needed
        bg = np.zeros((out_h, out_w, 4), dtype=np.uint8)
        bg[:] = [c * 255 for c in bg_color] # RGBA
        
        # Simple alpha blending
        alpha = resized_uint8[..., 3:4].astype(np.float32) / 255.0
        out = np.zeros_like(resized_uint8)
        out[..., :3] = resized_uint8[..., :3] * alpha + bg[..., :3] * (1 - alpha)
        out[..., 3] = 255 # Opaque result
        
        return out
    
    def _calculate_frame_repeats(self, frame_idx: int) -> int:
        """
        Calculate how many times a frame should be repeated based on its index.
        Uses config settings for exponential decay.
        """
        repeats = int(self.config.initial_repeats * (self.config.decay_rate ** frame_idx))
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
        print(f"Applying frame persistence (initial={self.config.initial_repeats}, decay={self.config.decay_rate})...")
        for i, frame in enumerate(base_rendered_frames):
            repeats = self._calculate_frame_repeats(i)
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
