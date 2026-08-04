"""
Palette extraction from the reference image (PLAN e).

k-means over the non-transparent pixels of the target, k ~ 5, sorted by
luminance. Everything the render needs to agree on chromatically -- the
background, the SCA gradient -- is derived from here instead of being hardcoded
in two different places.

The image is only ever read to *derive colours for the render*; no stage of the
pipeline samples it at inference time (see the project rule in PLAN, Fase 1).
"""

import colorsys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from config.palette_config import PaletteConfig
from utils.log import get_logger

logger = get_logger(__name__)

RGB = Tuple[float, float, float]
RGBA = Tuple[float, float, float, float]

# Rec. 709 luminance weights.
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass
class Palette:
    """Colours extracted from one image, sorted from darkest to lightest."""

    colors: List[RGB]
    weights: List[float]  # fraction of subject pixels belonging to each colour

    @property
    def dominant(self) -> RGB:
        """The colour that covers the most pixels."""
        return self.colors[int(np.argmax(self.weights))]

    @property
    def darkest(self) -> RGB:
        return self.colors[0]

    @property
    def lightest(self) -> RGB:
        return self.colors[-1]

    def luminance(self, color: RGB) -> float:
        return float(np.dot(np.asarray(color, dtype=np.float32), _LUMA))

    def describe(self) -> str:
        return ', '.join(
            f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x} ({w:.0%})'
            for (r, g, b), w in zip(self.colors, self.weights)
        )


def _load_subject_pixels(image_path: str, alpha_threshold: float) -> np.ndarray:
    """RGB float pixels of the subject only, shape [N, 3]."""
    from PIL import Image

    image = np.asarray(Image.open(image_path).convert('RGBA'), dtype=np.float32) / 255.0
    rgb = image[..., :3].reshape(-1, 3)
    alpha = image[..., 3].reshape(-1)

    subject = rgb[alpha > alpha_threshold]
    if len(subject) == 0:
        # Fully opaque-less image: fall back to everything rather than failing.
        logger.warning('Palette: no pixel above alpha %.2f, using the whole image',
                       alpha_threshold)
        subject = rgb
    return subject


def _kmeans(points: np.ndarray, k: int, iterations: int, seed: int = 0):
    """
    Plain Lloyd k-means with k-means++ seeding.

    Small enough to keep in-tree: a few thousand 3-D points and k = 5 converge
    in a handful of iterations, and it avoids adding scikit-learn as a
    dependency for one function.
    """
    rng = np.random.default_rng(seed)
    n = len(points)
    k = max(1, min(k, n))

    # k-means++ seeding: first centre at random, then favour distant points.
    centers = [points[rng.integers(n)]]
    for _ in range(k - 1):
        distances = np.min(
            ((points[:, None, :] - np.asarray(centers)[None, :, :]) ** 2).sum(-1),
            axis=1,
        )
        total = distances.sum()
        if total <= 0:
            centers.append(points[rng.integers(n)])
            continue
        centers.append(points[rng.choice(n, p=distances / total)])

    centers = np.asarray(centers, dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)

    for _ in range(iterations):
        distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(len(centers)):
            members = points[labels == j]
            if len(members):
                centers[j] = members.mean(axis=0)

    counts = np.bincount(labels, minlength=len(centers)).astype(np.float32)
    return centers, counts / max(1.0, counts.sum())


def extract_palette(image_path: str, config: PaletteConfig = None) -> Palette:
    """Extract the palette of one image, sorted by luminance."""
    config = config or PaletteConfig()

    pixels = _load_subject_pixels(image_path, config.alpha_threshold)

    if len(pixels) > config.sample_pixels:
        rng = np.random.default_rng(0)
        pixels = pixels[rng.choice(len(pixels), config.sample_pixels, replace=False)]

    centers, weights = _kmeans(pixels, config.num_colors, config.kmeans_iterations)

    order = np.argsort(centers @ _LUMA)
    palette = Palette(
        colors=[tuple(float(c) for c in centers[i]) for i in order],
        weights=[float(weights[i]) for i in order],
    )
    logger.info('Palette from %s: %s', image_path, palette.describe())
    return palette


# ==================== DERIVED COLOURS ====================

def _to_hsv(color: RGB):
    return colorsys.rgb_to_hsv(*color)


def _from_hsv(h: float, s: float, v: float) -> RGB:
    return colorsys.hsv_to_rgb(h, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))


def background_color(palette: Palette, config: PaletteConfig) -> RGBA:
    """
    Background colour derived from the palette (PLAN d).

    Two rules, both implemented so they can be compared on real images:

    - `dark_desaturated`: the dominant hue, drained of saturation and pushed
      down to `background_value`. Reads as the subject's own shadow.
    - `complementary`: the opposite hue at the same low value. More graphic,
      more separation between subject and ground.
    """
    if config.background_color is not None:
        return config.background_color

    h, s, _ = _to_hsv(palette.dominant)

    if config.background_rule == 'complementary':
        h = (h + 0.5) % 1.0

    r, g, b = _from_hsv(h, s * config.background_saturation, config.background_value)
    return (r, g, b, 1.0)


def tree_colors(palette: Palette, config: PaletteConfig) -> Tuple[RGBA, RGBA]:
    """
    Base and tip colour of the SCA gradient (PLAN e).

    Same hue as the dominant colour, desaturated; the gradient runs on value
    only, from a dark trunk to a lighter tip, instead of the hardcoded
    brown -> green ramp.
    """
    base_override = config.branch_color
    tip_override = config.branch_color_end
    if base_override is not None and tip_override is not None:
        return base_override, tip_override

    h, s, _ = _to_hsv(palette.dominant)
    s = s * config.tree_saturation

    base = (*_from_hsv(h, s, config.tree_base_value), 1.0)
    tip = (*_from_hsv(h, s * 0.8, config.tree_tip_value), 1.0)

    return (base_override or base), (tip_override or tip)


def resolve_palette(image_path: str, config: PaletteConfig) -> Optional[Palette]:
    """Extract the palette unless it is disabled; None means 'keep the defaults'."""
    if not config.enabled:
        return None
    return extract_palette(image_path, config)
