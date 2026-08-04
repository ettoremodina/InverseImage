"""
Configuration for the animated crop window (PLAN 3.5).

The camera does not need a canvas of its own: it crops inside the supersampled
canvas the renderers already draw on, and the crop is resolved to the final
video size in the same area-average resampling that does the antialiasing.
That is the "one shared internal scale factor" the plan asks for -- camera and
supersampling never enlarge the canvas twice.

Consequence to keep in mind: at `zoom_end` the crop keeps
`render_supersample * zoom_end` pixels per output pixel, so with the default
supersample of 2 the push-in still ends up oversampled.
"""

from dataclasses import dataclass


@dataclass
class CameraConfig:
    enabled: bool = True

    # Fraction of the canvas visible at the start / at the end.
    # 1.0 = whole canvas. Going down = pushing in.
    zoom_start: float = 1.0
    zoom_end: float = 0.86

    # Slow drift of the crop centre, as a fraction of the canvas, over the whole
    # video. The crop is always clamped inside the canvas, so at zoom 1.0 the
    # drift is necessarily zero and only opens up as the push-in progresses.
    drift_x: float = 0.035
    drift_y: float = -0.02

    easing: str = 'ease_in_out'
