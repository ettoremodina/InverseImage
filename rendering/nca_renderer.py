"""
NCA renderer.

Upsamples low-res NCA frames to high-res by nearest-neighbour block scaling,
so each simulation cell becomes a solid square of pixels.
"""

import numpy as np
import imageio
import cv2
from tqdm import tqdm
from typing import Dict, Any
from pathlib import Path

from config.render_config import NCARenderConfig
from .base import Renderer
from .utils import get_time_dilated_indices, open_video_writer


class NCARenderer(Renderer):
    def __init__(self, config: NCARenderConfig = None):
        super().__init__(config or NCARenderConfig())

    def render_frame(self, frame: np.ndarray, source_width: int, source_height: int) -> np.ndarray:
        """
        Render a single NCA frame using fast pixel upscaling.
        Treats each cell as a square block of pixels.

        `source_width` / `source_height` are accepted for interface symmetry with
        the other renderers; the output size comes from the config.
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

    def render_animation(self, data: Dict[str, Any], output_path: str, fps: int = 30,
                         duration_seconds: float = None):
        """
        Render full NCA animation from loaded data.

        Frames are rendered and written one at a time, so peak memory does not
        grow with the length of the video.

        Args:
            data: Dict from load_nca_frames() with keys: frames, source_width, source_height
            output_path: Path for output video (mp4)
            fps: Frames per second
            duration_seconds: Target video length. When omitted, early frames are
                simply repeated (initial_repeats/decay_rate) and the length
                follows from the number of simulation steps.
        """
        frames_data = data["frames"]
        source_w = data["source_width"]
        source_h = data["source_height"]

        if duration_seconds is not None:
            # Resample onto a fixed frame budget, keeping the slow start.
            indices = get_time_dilated_indices(
                len(frames_data),
                max(1, int(round(duration_seconds * fps))),
                self.config.initial_repeats,
                self.config.decay_rate,
            )
            repeats_for = None
        else:
            indices = range(len(frames_data))
            repeats_for = self._calculate_frame_repeats

        accumulated_frame = None
        smoothing = self.config.temporal_smoothing
        total = 0

        print("Rendering NCA frames...")
        with open_video_writer(output_path, fps) as writer:
            for i, idx in enumerate(tqdm(indices, desc="Rendering NCA frames")):
                rendered = self.render_frame(frames_data[idx], source_w, source_h)

                if smoothing > 0:
                    frame_float = rendered.astype(np.float32)
                    if accumulated_frame is None:
                        accumulated_frame = frame_float
                    else:
                        accumulated_frame = (
                            accumulated_frame * smoothing + frame_float * (1.0 - smoothing)
                        )
                    rendered = accumulated_frame.astype(np.uint8)

                for _ in range(repeats_for(i) if repeats_for else 1):
                    writer.append_data(rendered)
                    total += 1

        print(f"Saved animation: {output_path} ({total} frames, {total / fps:.2f}s)")

    def save_frame(self, frame: np.ndarray, source_width: int, source_height: int, 
                   output_path: str):
        """Render and save a single frame as PNG."""
        rendered = self.render_frame(frame, source_width, source_height)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(output_path, rendered)
