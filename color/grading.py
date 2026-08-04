"""
Shared grading pass (PLAN g).

The problem it solves: every stage currently leaves the pipeline through a
different code path, so the three pieces look like three different exports.
This is one function applied to *every* frame of *every* stage, right before it
is written -- tone curve, chromatic aberration, vignette, grain.

The grain is not decoration. It breaks up the pixel grid, so the low simulation
resolution reads as film grain instead of as a poor image.

Everything is numpy on uint8 RGB(A) frames; cost is a few milliseconds per
frame at 512x512.
"""

from typing import Optional, Tuple

import numpy as np

from config.grading_config import GradingConfig
from utils.log import get_logger

logger = get_logger(__name__)

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


class Grader:
    """
    Stateful grading pass.

    Stateful only for caching: the vignette mask and the fixed grain field
    depend on the frame size alone, so they are built once and reused for the
    whole render.
    """

    def __init__(self, config: GradingConfig = None):
        self.config = config or GradingConfig()
        self._vignette: Optional[np.ndarray] = None
        self._vignette_shape: Optional[Tuple[int, int]] = None
        self._static_grain: Optional[np.ndarray] = None
        self._aberration_maps = None
        self._rng = np.random.default_rng(self.config.seed)

    # ------------------------------------------------------------------ tone
    def _tone_curve(self, rgb: np.ndarray) -> np.ndarray:
        cfg = self.config

        rgb = rgb * cfg.gain + cfg.lift
        rgb = np.clip(rgb, 0.0, 1.0)

        if cfg.gamma != 1.0:
            rgb = np.power(rgb, cfg.gamma)

        if cfg.contrast != 1.0:
            rgb = np.clip((rgb - cfg.contrast_pivot) * cfg.contrast + cfg.contrast_pivot,
                          0.0, 1.0)

        if cfg.saturation != 1.0:
            luma = (rgb * _LUMA).sum(axis=-1, keepdims=True)
            rgb = np.clip(luma + (rgb - luma) * cfg.saturation, 0.0, 1.0)

        if cfg.shadow_tint is not None and cfg.shadow_tint_strength > 0:
            luma = (rgb * _LUMA).sum(axis=-1, keepdims=True)
            # Weight tends to 1 in the shadows and to 0 in the highlights.
            weight = (1.0 - luma) ** 2 * cfg.shadow_tint_strength
            tint = np.asarray(cfg.shadow_tint, dtype=np.float32).reshape(1, 1, 3)
            rgb = np.clip(rgb * (1.0 - weight) + tint * weight, 0.0, 1.0)

        return rgb

    # -------------------------------------------------------------- vignette
    def _vignette_mask(self, height: int, width: int) -> np.ndarray:
        if self._vignette is not None and self._vignette_shape == (height, width):
            return self._vignette

        cfg = self.config
        y = np.linspace(-1.0, 1.0, height, dtype=np.float32).reshape(-1, 1)
        x = np.linspace(-1.0, 1.0, width, dtype=np.float32).reshape(1, -1)
        radius = np.sqrt(x * x + y * y) / np.sqrt(2.0)

        falloff = np.clip((radius - cfg.vignette_radius) / max(1e-6, 1.0 - cfg.vignette_radius),
                          0.0, 1.0)
        mask = 1.0 - cfg.vignette_strength * np.power(falloff, cfg.vignette_softness)

        self._vignette = mask[..., None].astype(np.float32)
        self._vignette_shape = (height, width)
        return self._vignette

    # ----------------------------------------------------------------- grain
    def _grain_field(self, height: int, width: int, animated: bool) -> np.ndarray:
        """Noise field at frame resolution, generated smaller when grain_size > 1."""
        import cv2

        cfg = self.config
        small_h = max(1, int(height / max(1.0, cfg.grain_size)))
        small_w = max(1, int(width / max(1.0, cfg.grain_size)))

        if not animated:
            if self._static_grain is not None and self._static_grain.shape[:2] == (height, width):
                return self._static_grain
            rng = np.random.default_rng(cfg.seed)
        else:
            rng = self._rng

        noise = rng.standard_normal((small_h, small_w, 3)).astype(np.float32)

        # Push the noise towards luminance: coloured grain looks like sensor
        # noise, luminance grain looks like film.
        mono = noise.mean(axis=-1, keepdims=True)
        noise = mono * cfg.grain_luma_weight + noise * (1.0 - cfg.grain_luma_weight)

        if (small_h, small_w) != (height, width):
            noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_LINEAR)

        if not animated:
            self._static_grain = noise
        return noise

    def _apply_grain(self, rgb: np.ndarray) -> np.ndarray:
        cfg = self.config
        height, width = rgb.shape[:2]
        noise = self._grain_field(height, width, cfg.grain_animated)

        if cfg.grain_shadow_bias > 0:
            luma = (rgb * _LUMA).sum(axis=-1, keepdims=True)
            weight = 1.0 - cfg.grain_shadow_bias * luma
        else:
            weight = 1.0

        return np.clip(rgb + noise * cfg.grain_strength * weight, 0.0, 1.0)

    # ----------------------------------------------------- chromatic aberration
    def _aberration(self, rgb: np.ndarray) -> np.ndarray:
        """
        Radial channel separation: red pushed out, blue pulled in.

        Implemented as two whole-frame scalings rather than a remap, which is
        both cheaper and exactly what a lens does to first order.
        """
        import cv2

        strength = self.config.aberration_strength
        if strength <= 0:
            return rgb

        height, width = rgb.shape[:2]
        out = rgb.copy()

        for channel, scale in ((0, 1.0 + strength), (2, 1.0 - strength)):
            matrix = np.array([
                [scale, 0.0, (1.0 - scale) * width * 0.5],
                [0.0, scale, (1.0 - scale) * height * 0.5],
            ], dtype=np.float32)
            out[..., channel] = cv2.warpAffine(
                rgb[..., channel], matrix, (width, height),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            )

        return out

    # ------------------------------------------------------------------ main
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Grade one frame.

        Args:
            frame: uint8 RGB or RGBA. The alpha channel, if any, is passed
                through untouched -- grading is a look, not a coverage change.

        Returns:
            uint8 array of the same shape.
        """
        if not self.config.enabled:
            return frame

        alpha = frame[..., 3:] if frame.shape[-1] == 4 else None
        rgb = frame[..., :3].astype(np.float32) / 255.0

        if self.config.tone_enabled:
            rgb = self._tone_curve(rgb)
        if self.config.aberration_enabled:
            rgb = self._aberration(rgb)
        if self.config.vignette_enabled:
            rgb = np.clip(rgb * self._vignette_mask(*rgb.shape[:2]), 0.0, 1.0)
        if self.config.grain_enabled:
            rgb = self._apply_grain(rgb)

        graded = (rgb * 255.0 + 0.5).astype(np.uint8)

        if alpha is None:
            return graded
        return np.concatenate([graded, alpha], axis=-1)
