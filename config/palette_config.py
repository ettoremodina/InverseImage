"""
Configuration for the palette derived from the reference image (PLAN d, e).

The palette is the single source of the background colour and of the SCA tree
gradient. Every field here can be overridden by hand, and a manual override
always wins over the extracted value.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

BackgroundRule = Literal['dark_desaturated', 'complementary']

RGBA = Tuple[float, float, float, float]


@dataclass
class PaletteConfig:
    enabled: bool = True

    # k-means on the non-transparent pixels of the target image.
    num_colors: int = 5
    sample_pixels: int = 20000     # random subsample; k-means on the full image is pointless
    kmeans_iterations: int = 25
    alpha_threshold: float = 0.1   # pixels below this alpha are not part of the subject

    # How the background colour is derived from the palette. Both rules are
    # implemented: this is a taste call, so it is decided by looking at a few
    # images rather than on paper (PLAN, working principle).
    background_rule: BackgroundRule = 'dark_desaturated'
    background_value: float = 0.12      # target luminance of the background
    background_saturation: float = 0.35  # how much of the original saturation survives

    # SCA tree: same hue as the dominant colour, desaturated and darkened,
    # instead of the hardcoded brown -> green gradient.
    tree_from_palette: bool = True
    tree_base_value: float = 0.30       # value of the trunk end of the gradient
    tree_tip_value: float = 0.62        # value of the tip end
    tree_saturation: float = 0.30

    # ==================== MANUAL OVERRIDES ====================
    # None = derive from the image. Anything else wins.
    background_color: Optional[RGBA] = None
    branch_color: Optional[RGBA] = None
    branch_color_end: Optional[RGBA] = None
