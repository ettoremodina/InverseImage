"""
Internal render scale and final resolve (PLAN b, 3.5).

Everything is drawn once on an internal canvas of `render_size * supersample`
pixels and reduced to the video size with an area average. That is real
antialiasing: the SCA's thin branches stop shimmering and the cell edges come
out soft. The NCA simulation stays at 128 and the video stays at 512, so the
file size does not move -- only the rasterisation time (~3-4x).

The camera crop happens in the same step: crop and downscale are a single
resampling, so the frame is never resized twice (PLAN 3.5, "one shared internal
scale factor").
"""

from typing import Optional, Tuple

import numpy as np

from utils.log import get_logger

logger = get_logger(__name__)

# Integer pixel rectangle on the internal canvas: (x, y, width, height).
Rect = Tuple[int, int, int, int]


def internal_size(output_size: int, supersample: int) -> int:
    """Side of the internal canvas for a given video size."""
    return int(output_size) * max(1, int(supersample))


def resolve(frame: np.ndarray, output_size: int, crop: Optional[Rect] = None) -> np.ndarray:
    """
    Bring one internal-resolution frame down to the video resolution.

    Args:
        frame: uint8 [H, W, C] rendered on the internal canvas.
        output_size: side of the final frame.
        crop: optional (x, y, w, h) window on the internal canvas. None = full frame.

    Returns:
        uint8 [output_size, output_size, C].
    """
    import cv2

    if crop is not None:
        x, y, w, h = crop
        frame = frame[y:y + h, x:x + w]

    if frame.shape[0] == output_size and frame.shape[1] == output_size:
        return np.ascontiguousarray(frame)

    # INTER_AREA is the average over the source footprint: exactly the box
    # filter supersampling wants. On upscales it degrades to bilinear, which is
    # the right behaviour for a heavy camera push-in.
    interpolation = cv2.INTER_AREA if frame.shape[0] >= output_size else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (output_size, output_size), interpolation=interpolation)
    return np.ascontiguousarray(resized)
