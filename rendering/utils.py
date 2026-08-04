"""
Rendering utility functions.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import cairo
import imageio
import numpy as np

# Every video in the pipeline goes through these settings, so clips can be
# concatenated later with a stream copy instead of being re-encoded.
VIDEO_CODEC = 'libx264'
VIDEO_QUALITY = 8
VIDEO_PIXELFORMAT = 'yuv420p'


def open_video_writer(output_path: str, fps: int):
    """
    Open a streaming H.264 writer.

    Streaming keeps memory flat: a 30 s 1024x1024 render is ~2.5 GB if the
    frames are collected in a list first, and nothing at all this way.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        output_path,
        fps=fps,
        codec=VIDEO_CODEC,
        quality=VIDEO_QUALITY,
        pixelformat=VIDEO_PIXELFORMAT,
        macro_block_size=1,
    )


def create_surface(width: int, height: int,
                   background=(0.0, 0.0, 0.0, 0.0),
                   antialias: bool = True) -> Tuple[cairo.ImageSurface, cairo.Context]:
    """
    A Cairo surface painted with `background`, plus its context.

    Shared by every renderer so surface creation, antialias policy and the
    background fill live in exactly one place.
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(width), int(height))
    ctx = cairo.Context(surface)

    # ANTIALIAS_GOOD rather than BEST: on dense line work BEST costs
    # noticeably more rasterisation time for no visible difference.
    if antialias:
        ctx.set_antialias(cairo.ANTIALIAS_GOOD)

    r, g, b, a = background
    ctx.set_source_rgba(r, g, b, a)
    ctx.set_operator(cairo.OPERATOR_SOURCE)
    ctx.paint()
    ctx.set_operator(cairo.OPERATOR_OVER)

    return surface, ctx


def surface_to_numpy(surface: cairo.ImageSurface, width: int, height: int,
                     unpremultiply: bool = False) -> np.ndarray:
    """
    Copy a Cairo ARGB32 surface into a uint8 RGBA array.

    Cairo stores ARGB32 *premultiplied*. On an opaque surface that is a no-op,
    but a transparent layer meant to be alpha-composited later must be
    unpremultiplied first, or the alpha ends up applied twice and the layer
    comes out dark at its soft edges.
    """
    surface.flush()
    buf = surface.get_data()
    arr = np.ndarray(shape=(int(height), int(width), 4), dtype=np.uint8, buffer=buf)

    rgba = np.empty_like(arr)
    rgba[:, :, 0] = arr[:, :, 2]  # R
    rgba[:, :, 1] = arr[:, :, 1]  # G
    rgba[:, :, 2] = arr[:, :, 0]  # B
    rgba[:, :, 3] = arr[:, :, 3]  # A

    if unpremultiply:
        alpha = rgba[:, :, 3:4].astype(np.float32)
        scale = np.where(alpha > 0, 255.0 / np.maximum(alpha, 1.0), 0.0)
        straight = np.clip(rgba[:, :, :3].astype(np.float32) * scale, 0, 255)
        rgba[:, :, :3] = straight.astype(np.uint8)

    return rgba


def load_rgb_image(path: str, background=(1.0, 1.0, 1.0)) -> np.ndarray:
    """
    Load an image as float32 RGB in [0, 1], flattening alpha onto `background`.

    Target images are usually RGBA cut-outs whose transparent pixels are stored
    as pure black. Reading them without the alpha channel turns 80% of the
    image into black, which the particle stage would then happily paint onto
    the canvas. Compositing first keeps the transparent area the same colour as
    the render background, so stray particles stay invisible.
    """
    import cv2  # local import: only the particle path needs OpenCV

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.astype(np.float32) / 255.0

    if image.shape[2] == 4:
        alpha = image[..., 3:4]
        canvas = np.asarray(background, dtype=np.float32).reshape(1, 1, 3)
        image = image[..., :3] * alpha + canvas * (1.0 - alpha)

    return np.ascontiguousarray(image)


def max_polyline_depth(polylines: List[Dict]) -> int:
    """Deepest point depth in an SCA polyline set (1 when the tree is empty)."""
    return max((p['depths'][-1] for p in polylines), default=1)


def draw_line(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: list):
    """Draw a line using Bresenham's algorithm."""
    h, w = img.shape[:2]
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    while True:
        if 0 <= x1 < w and 0 <= y1 < h:
            img[y1, x1] = color
        
        if x1 == x2 and y1 == y2:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy


def get_time_dilated_indices(num_source_frames: int, num_target_frames: int, 
                             initial_repeats: float, decay_rate: float) -> np.ndarray:
    """
    Calculate indices for resampling frames with time dilation (slow start).
    
    Args:
        num_source_frames: Number of available source frames
        num_target_frames: Number of frames in the output video
        initial_repeats: Relative weight of the first frame
        decay_rate: Decay rate for weight of subsequent frames
        
    Returns:
        Array of indices (integers) of length num_target_frames
    """
    if num_source_frames == 0:
        return np.zeros(num_target_frames, dtype=int)
        
    # Calculate weight (duration) for each source frame
    indices = np.arange(num_source_frames)
    weights = initial_repeats * (decay_rate ** indices)
    
    # Calculate cumulative weight (time)
    cumulative_weights = np.cumsum(weights)
    total_weight = cumulative_weights[-1]
    
    # Map target frames to source frames
    # Target times are evenly spaced from 0 to total_weight
    target_times = np.linspace(0, total_weight, num_target_frames)
    
    # Find which source frame corresponds to each target time
    # searchsorted finds the first index where cumulative_weight >= target_time
    resampled_indices = np.searchsorted(cumulative_weights, target_times)
    
    # Clip to valid range
    return np.clip(resampled_indices, 0, num_source_frames - 1)

