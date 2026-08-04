"""
Configuration for the shared grading pass (PLAN g).

One pass applied to every frame of every stage, immediately before the frame is
written. Each effect can be switched off on its own, and the whole pass can be
switched off with `enabled`.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GradingConfig:
    enabled: bool = True

    # ==================== TONE CURVE ====================
    tone_enabled: bool = True
    lift: float = 0.02        # blacks raised: pure black reads as dead on video
    gain: float = 1.02        # highlight multiplier
    gamma: float = 0.98       # < 1 brightens midtones
    contrast: float = 1.06    # S-curve strength around the pivot
    contrast_pivot: float = 0.45
    saturation: float = 1.04
    # Shadows tinted towards the background colour, so the three stages sit in
    # the same chromatic world. None = filled in from the palette at runtime.
    shadow_tint: Optional[Tuple[float, float, float]] = None
    shadow_tint_strength: float = 0.12

    # ==================== VIGNETTE ====================
    vignette_enabled: bool = True
    vignette_strength: float = 0.28   # 0 = none, 1 = heavy
    vignette_radius: float = 0.75     # where the falloff starts, in half-diagonals
    vignette_softness: float = 1.6    # falloff exponent

    # ==================== GRAIN ====================
    grain_enabled: bool = True
    grain_strength: float = 0.030     # std-dev in [0, 1] units
    grain_size: float = 1.4           # >1 renders the noise smaller and upscales it
    grain_animated: bool = True       # new noise every frame (PLAN default)
    grain_luma_weight: float = 0.7    # 1 = pure luminance grain, 0 = full RGB noise
    grain_shadow_bias: float = 0.5    # how much more grain lands in the shadows

    # ==================== CHROMATIC ABERRATION ====================
    aberration_enabled: bool = True
    aberration_strength: float = 0.0018  # radial channel offset, fraction of the frame

    # Seed for the fixed-grain mode and for the per-frame noise sequence.
    seed: int = 1234
