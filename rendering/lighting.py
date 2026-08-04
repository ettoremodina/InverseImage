"""
Light instead of pixels (PLAN c).

A normal map is derived from the gradient of the smoothed alpha, then the cell
layer gets a lambertian term, a rim light on the silhouette and a soft bloom on
the brightest cells. Pure 2-D numpy/OpenCV work applied *after* the cells are
rasterised, so it costs a few milliseconds and touches nothing upstream.

What it buys: the residual per-cell irregularity left by `cells.py` stops
reading as a grid of dots and starts reading as a surface with relief.

All the blur radii are expressed in *output* pixels. When the render runs
supersampled, pass `scale = supersample` so the look does not change with the
internal resolution.
"""

from typing import Optional

import numpy as np

from config.render_config import LightingConfig

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _normalized(vector) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float32)


def apply_lighting(layer: np.ndarray, config: LightingConfig = None,
                   scale: float = 1.0,
                   height_field: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Light one RGBA layer using its own alpha as a height field.

    Args:
        layer: uint8 [H, W, 4], straight (non-premultiplied) alpha.
        config: lighting parameters.
        scale: internal pixels per output pixel (the supersample factor).
        height_field: optional float [H, W] to use instead of the layer alpha,
            for when the relief should not follow coverage exactly.

    Returns:
        uint8 [H, W, 4]. Alpha is preserved, except where the bloom adds a halo.
    """
    import cv2

    config = config or LightingConfig()
    if not config.enabled:
        return layer

    alpha = layer[..., 3].astype(np.float32) / 255.0
    if float(alpha.max()) <= 0.0:
        return layer

    rgb = layer[..., :3].astype(np.float32) / 255.0

    # ---------------------------------------------------------- normal map
    source = alpha if height_field is None else height_field.astype(np.float32)
    sigma = max(0.1, config.height_blur * scale)
    height = cv2.GaussianBlur(source, (0, 0), sigmaX=sigma, sigmaY=sigma)

    gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)

    # Gradients shrink as the resolution grows, so the relief is scaled with it
    # to keep the same apparent bumpiness at any supersampling factor.
    relief = config.relief * scale
    nx = -gx * relief
    ny = -gy * relief
    nz = np.ones_like(nx)

    norm = np.sqrt(nx * nx + ny * ny + 1.0)
    nx /= norm
    ny /= norm
    nz /= norm

    # ------------------------------------------------------------- lambert
    light = _normalized(config.light_direction)
    lambert = np.clip(nx * light[0] + ny * light[1] + nz * light[2], 0.0, 1.0)
    shade = config.ambient + config.diffuse * lambert
    rgb = rgb * shade[..., None]

    # ---------------------------------------------------------------- rim
    if config.rim_enabled and config.rim_intensity > 0:
        # nz falls off exactly where the surface turns away from the viewer,
        # which on an alpha height field is the silhouette.
        facing = np.clip(1.0 - nz, 0.0, 1.0)
        rim = np.power(facing, config.rim_power) * config.rim_intensity * alpha
        rim_color = np.asarray(config.rim_color, dtype=np.float32).reshape(1, 1, 3)
        rgb = rgb + rim[..., None] * rim_color

    rgb = np.clip(rgb, 0.0, 1.0)

    # -------------------------------------------------------------- bloom
    if config.bloom_enabled and config.bloom_intensity > 0:
        luma = (rgb * _LUMA).sum(axis=-1) * alpha
        bright = np.clip(luma - config.bloom_threshold, 0.0, 1.0)
        if float(bright.max()) > 0.0:
            blur = max(0.1, config.bloom_blur * scale)
            glow = cv2.GaussianBlur(bright, (0, 0), sigmaX=blur, sigmaY=blur)
            glow *= config.bloom_intensity
            rgb = np.clip(rgb + glow[..., None], 0.0, 1.0)
            # The halo has to carry some coverage of its own, otherwise it is
            # multiplied away when the layer is composited.
            alpha = np.clip(alpha + glow, 0.0, 1.0)

    out = np.empty_like(layer)
    out[..., :3] = (rgb * 255.0 + 0.5).astype(np.uint8)
    out[..., 3] = (alpha * 255.0 + 0.5).astype(np.uint8)
    return out
