"""
Animated crop window (PLAN 3.5).

Slow push-in plus a drift, expressed as a rectangle on the internal canvas. The
camera never rasterises anything: it hands a rectangle to
`supersample.resolve`, which crops and downscales in one resampling.

Disabled, it returns the full canvas and costs exactly nothing.
"""

from typing import Tuple

import numpy as np

from config.camera_config import CameraConfig
from .easing import apply_easing

Rect = Tuple[int, int, int, int]


class Camera:
    """Produces the crop rectangle for a given point in the video."""

    def __init__(self, config: CameraConfig = None, canvas_size: int = 1024):
        self.config = config or CameraConfig()
        self.canvas_size = int(canvas_size)

    def crop(self, progress: float) -> Rect:
        """
        Crop rectangle at `progress` in [0, 1] over the whole video.

        The rectangle is always clamped inside the canvas, so the drift can only
        use the margin the push-in has opened up. At zoom 1.0 there is no margin
        and the drift is necessarily zero -- that is why the default
        `zoom_start` sits at 1.0 and the movement builds up rather than starting
        at full speed.
        """
        canvas = self.canvas_size

        if not self.config.enabled:
            return (0, 0, canvas, canvas)

        t = apply_easing(self.config.easing, progress)
        zoom = self.config.zoom_start + (self.config.zoom_end - self.config.zoom_start) * t
        zoom = float(np.clip(zoom, 0.05, 1.0))

        size = max(2, int(round(canvas * zoom)))
        margin = canvas - size

        # Drift is expressed as a fraction of the canvas and grows with t.
        offset_x = self.config.drift_x * canvas * t
        offset_y = self.config.drift_y * canvas * t

        x = int(round(margin * 0.5 + offset_x))
        y = int(round(margin * 0.5 + offset_y))
        x = int(np.clip(x, 0, margin))
        y = int(np.clip(y, 0, margin))

        return (x, y, size, size)
