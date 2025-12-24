"""
NCA renderer using Cairo.
Upsamples low-res NCA frames to high-res with configurable cell shapes.

Future extensions: jitter, glow, spawn animations, texture fills.
"""

import cairo
import numpy as np
import imageio
import cv2
import multiprocessing
from tqdm import tqdm
from typing import Dict, Any, Tuple
from pathlib import Path

from config.render_config import NCARenderConfig
from .base import Renderer


def render_nca_frame_wrapper(args):
    config, frame, source_w, source_h = args
    renderer = NCARenderer(config)
    return renderer.render_frame(frame, source_w, source_h)


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
        if self.config.use_metaballs:
            base_scale *= self.config.metaball_blur_radius
            
        radius = (cell_size / 2) * base_scale
        
        h, w = frame.shape[:2]
        
        for y in range(h):
            for x in range(w):
                r, g, b, a = frame[y, x]
                
                if a < self.config.alpha_threshold:
                    continue
                
                cx = (x + 0.5) * cell_w
                cy = (y + 0.5) * cell_h
                
                # Breathing effect
                scale_factor = 1.0
                if self.config.use_breathing:
                    # Map alpha 0..1 to (1-factor)..1
                    scale_factor = 1.0 - self.config.breathing_factor * (1.0 - a)
                
                if self.config.cell_shape == "circle" or self.config.use_metaballs:
                    draw_radius = radius * scale_factor
                    self._draw_cell_circle(ctx, cx, cy, draw_radius, r, g, b, a)
                else:
                    draw_size = cell_size * base_scale * scale_factor
                    self._draw_cell_square(ctx, cx, cy, draw_size, r, g, b, a)

    def _apply_metaballs(self, image: np.ndarray) -> np.ndarray:
        """Apply metaball effect by blurring and thresholding alpha."""
        if not self.config.use_metaballs:
            return image
            
        # Separate channels
        bgr = image[..., :3]
        alpha = image[..., 3]
        
        # Blur alpha channel
        # Kernel size depends on resolution, heuristic: 1% of width
        ksize = int(self.config.output_width * 0.02) | 1
        blurred_alpha = cv2.GaussianBlur(alpha, (ksize, ksize), 0)
        
        # Threshold alpha
        _, thresh_alpha = cv2.threshold(blurred_alpha, self.config.metaball_threshold * 255, 255, cv2.THRESH_BINARY)
        
        # Smooth the edges of the thresholded alpha slightly for anti-aliasing
        final_alpha = cv2.GaussianBlur(thresh_alpha, (3, 3), 0)
        
        # Also blur color channels to blend neighbors
        blurred_bgr = cv2.GaussianBlur(bgr, (ksize, ksize), 0)
        
        # Recombine
        result = np.dstack((blurred_bgr, final_alpha))
        return result

    def _apply_fake_3d(self, image: np.ndarray) -> np.ndarray:
        """Apply fake 3D lighting based on alpha height map."""
        if not self.config.use_fake_3d:
            return image
            
        # Use alpha as height map
        height_map = image[..., 3].astype(np.float32) / 255.0
        
        # Compute gradients
        gx = cv2.Sobel(height_map, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(height_map, cv2.CV_32F, 0, 1, ksize=3)
        
        # Construct normal map (N = (-gx, -gy, 1))
        # Normalize
        ones = np.ones_like(height_map)
        norm = np.sqrt(gx**2 + gy**2 + ones**2)
        nx = -gx / norm
        ny = -gy / norm
        nz = ones / norm
        
        # Light vector
        lx, ly, lz = self.config.light_pos
        l_len = np.sqrt(lx**2 + ly**2 + lz**2)
        lx, ly, lz = lx/l_len, ly/l_len, lz/l_len
        
        # Diffuse lighting (N dot L)
        diffuse = np.maximum(0, nx*lx + ny*ly + nz*lz)
        
        # Specular lighting (Phong)
        # View vector is (0,0,1)
        # Reflection vector R = 2(N.L)N - L
        rx = 2 * diffuse * nx - lx
        ry = 2 * diffuse * ny - ly
        rz = 2 * diffuse * nz - lz
        
        # R dot V (where V is 0,0,1) => rz
        specular = np.maximum(0, rz) ** self.config.shininess
        specular = specular * self.config.specular_intensity
        
        # Apply lighting to RGB
        rgb = image[..., :3].astype(np.float32)
        
        # Ambient + Diffuse + Specular
        # Simple model: Color * (0.2 + 0.8 * Diffuse) + Specular * 255
        lighting = (0.2 + 0.8 * diffuse[..., np.newaxis])
        lit_rgb = rgb * lighting + (specular[..., np.newaxis] * 255)
        
        lit_rgb = np.clip(lit_rgb, 0, 255).astype(np.uint8)
        
        return np.dstack((lit_rgb, image[..., 3]))

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
        raw_image = self._surface_to_numpy(surface)
        
        # Apply post-processing effects
        processed = self._apply_metaballs(raw_image)
        processed = self._apply_fake_3d(processed)
        
        return processed
    
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
        
        if self.config.temporal_smoothing > 0:
            accumulated_frame = None
            for frame in tqdm(frames_data, desc="Rendering NCA frames"):
                rendered = self.render_frame(frame, source_w, source_h)
                
                # Temporal smoothing
                rendered_float = rendered.astype(np.float32)
                
                if accumulated_frame is None:
                    accumulated_frame = rendered_float
                else:
                    # Exponential moving average
                    # smoothing=0.9 means keep 90% of history, add 10% new
                    alpha = 1.0 - self.config.temporal_smoothing
                    accumulated_frame = accumulated_frame * (1.0 - alpha) + rendered_float * alpha
                
                final_frame = accumulated_frame.astype(np.uint8)
                rendered_frames.append(final_frame)
        else:
            # Parallel rendering
            num_cores = max(1, multiprocessing.cpu_count() - 1) # Leave one core free
            print(f"Rendering with {num_cores} cores...")
            
            task_args = [(self.config, frame, source_w, source_h) for frame in frames_data]
            
            with multiprocessing.Pool(processes=num_cores) as pool:
                rendered_frames = list(tqdm(pool.imap(render_nca_frame_wrapper, task_args), total=len(frames_data), desc="Rendering NCA frames (Parallel)"))
        
        imageio.mimsave(output_path, rendered_frames, fps=fps)
        print(f"Saved animation: {output_path}")
    
    def save_frame(self, frame: np.ndarray, source_width: int, source_height: int, 
                   output_path: str):
        """Render and save a single frame as PNG."""
        rendered = self.render_frame(frame, source_width, source_height)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(output_path, rendered)
