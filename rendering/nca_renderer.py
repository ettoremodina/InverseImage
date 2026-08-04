"""
NCA renderer.

Two paths, chosen by `config.cells.enabled`:

- cells (default): every living cell is drawn as its own disc with Cairo, with a
  cellularity that closes into a surface as the tissue matures (PLAN a), then
  lit with a normal map derived from its alpha (PLAN c);
- blocks (legacy): nearest-neighbour upscaling, one square of pixels per cell.

The simulation is untouched either way -- this is rendering code.
"""

import numpy as np
import imageio
import cv2
from tqdm import tqdm
from typing import Dict, Any, Optional
from pathlib import Path

from config.render_config import NCARenderConfig
from .base import Renderer
from .cells import CellPainter, cellularity_at
from .lighting import apply_lighting
from .utils import get_time_dilated_indices, open_video_writer


class NCARenderer(Renderer):
    def __init__(self, config: NCARenderConfig = None):
        super().__init__(config or NCARenderConfig())
        self.cell_painter = CellPainter(self.config.cells)

    # ------------------------------------------------------------------ layer
    def render_layer(self, frame: np.ndarray, progress: float = 1.0) -> np.ndarray:
        """
        The cells on a transparent layer, lit, with straight alpha.

        This is what the timeline composites over the tree. `progress` is the
        NCA stage progress in [0, 1] and only drives the cellularity.
        """
        config = self.config
        out_h, out_w = config.output_height, config.output_width

        if config.cells.enabled:
            cellularity = cellularity_at(config.cells, progress)
            layer = self.cell_painter.render(
                frame, out_w, out_h, cellularity, config.antialiasing
            )
        else:
            resized = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            resized = resized.copy()
            resized[resized[..., 3] < config.alpha_threshold, 3] = 0
            layer = (np.clip(resized, 0, 1) * 255).astype(np.uint8)

        return apply_lighting(layer, config.lighting, scale=config.render_scale)

    def render_frame(self, frame: np.ndarray, source_width: int = None,
                     source_height: int = None, progress: float = 1.0) -> np.ndarray:
        """
        Render a single NCA frame, composited over the configured background.

        `source_width` / `source_height` are accepted for interface symmetry
        with the other renderers; the output size comes from the config.
        """
        layer = self.render_layer(frame, progress)

        bg_color = self.config.background_color
        if bg_color[3] == 0:
            return layer

        background = np.empty_like(layer)
        background[..., :3] = [int(c * 255) for c in bg_color[:3]]
        background[..., 3] = 255

        return composite(background, layer)

    def _calculate_frame_repeats(self, frame_idx: int) -> int:
        """
        Calculate how many times a frame should be repeated based on its index.
        Uses config settings for exponential decay.
        """
        repeats = int(self.config.initial_repeats * (self.config.decay_rate ** frame_idx))
        return max(1, repeats)

    def render_animation(self, data: Dict[str, Any], output_path: str, fps: int = 30,
                         duration_seconds: float = None,
                         resolve: Optional[callable] = None):
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
            resolve: optional callable applied to every finished frame, used to
                bring the supersampled canvas down to the video size and to
                apply the grading pass.
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
        count = len(indices)

        print("Rendering NCA frames...")
        with open_video_writer(output_path, fps) as writer:
            for i, idx in enumerate(tqdm(indices, desc="Rendering NCA frames")):
                progress = i / max(1, count - 1)
                rendered = self.render_frame(frames_data[idx], source_w, source_h, progress)

                if smoothing > 0:
                    frame_float = rendered.astype(np.float32)
                    if accumulated_frame is None:
                        accumulated_frame = frame_float
                    else:
                        accumulated_frame = (
                            accumulated_frame * smoothing + frame_float * (1.0 - smoothing)
                        )
                    rendered = accumulated_frame.astype(np.uint8)

                if resolve is not None:
                    rendered = resolve(rendered, i)

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


def composite(background: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """
    Alpha-composite a straight-alpha RGBA layer over an opaque background.

    Lives here rather than in each renderer because the timeline, the NCA
    renderer and the (future) swarm stage all need exactly this operation.
    """
    alpha = layer[..., 3:4].astype(np.float32) / 255.0

    out = background.astype(np.float32)
    out[..., :3] = layer[..., :3] * alpha + background[..., :3] * (1.0 - alpha)
    out[..., 3] = 255.0

    return out.astype(np.uint8)
