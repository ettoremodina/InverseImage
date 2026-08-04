"""
NCA cells drawn as cells (PLAN a).

The old renderer upscaled the simulation grid with nearest-neighbour, so every
cell was a hard square of pixels and the result read as a low-resolution image.
Here each living cell is drawn as its own disc with Cairo, at whatever internal
resolution the render runs at.

Zero impact on the NCA training by construction: this is rendering code, the
model never sees it.

The idea that makes it a surface rather than a colony: **the cellularity is a
function of time, not a fixed property.**

- a young cell (low alpha) gets a small radius and a visible gap: you read the
  individuals while they grow;
- a mature cell gets a radius past the grid step, so neighbours interpenetrate,
  the gaps close and the colony becomes a continuous surface.

At the end the cellularity does not reach zero: `cellularity_floor` leaves a
barely perceptible residue. A perfectly smooth surface is dead; the residue,
with the lighting of `lighting.py`, reads as skin.
"""

import math
from typing import Optional, Tuple

import numpy as np

from config.render_config import CellRenderConfig
from .utils import create_surface, surface_to_numpy

TAU = 2.0 * math.pi


def cellularity_at(config: CellRenderConfig, progress: float) -> float:
    """
    Cellularity at a given point of the growth.

    `progress` is the NCA stage progress in [0, 1]. The value falls from
    `cellularity` to `cellularity_floor`; `cellularity_curve` > 1 keeps the
    colony look alive longer before the surface closes.
    """
    floor = min(config.cellularity_floor, config.cellularity)
    t = float(np.clip(progress, 0.0, 1.0))
    decay = (1.0 - t) ** (1.0 / max(1e-3, config.cellularity_curve))
    return floor + (config.cellularity - floor) * decay


class CellPainter:
    """
    Draws one NCA frame as a field of cells.

    The per-cell jitter is deterministic (seeded once per grid size), so a cell
    keeps its own small irregularity for the whole animation instead of
    twitching from frame to frame.
    """

    def __init__(self, config: CellRenderConfig = None):
        self.config = config or CellRenderConfig()
        self._jitter: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self._jitter_shape: Optional[Tuple[int, int]] = None

    def _jitter_fields(self, height: int, width: int):
        if self._jitter is not None and self._jitter_shape == (height, width):
            return self._jitter

        rng = np.random.default_rng(self.config.jitter_seed)
        fields = (
            rng.uniform(-1.0, 1.0, (height, width)).astype(np.float32),  # radius
            rng.uniform(-1.0, 1.0, (height, width)).astype(np.float32),  # x
            rng.uniform(-1.0, 1.0, (height, width)).astype(np.float32),  # y
        )
        self._jitter = fields
        self._jitter_shape = (height, width)
        return fields

    def draw(self, ctx, frame: np.ndarray, out_width: int, out_height: int,
             cellularity: float):
        """
        Draw the living cells of `frame` onto an existing Cairo context.

        Args:
            ctx: Cairo context, already sized to (out_width, out_height).
            frame: [H, W, 4] float RGBA in [0, 1] straight from the simulation.
            cellularity: 0 = smooth surface, 1 = marked colony.
        """
        config = self.config
        height, width = frame.shape[:2]

        step_x = out_width / width
        step_y = out_height / height
        step = 0.5 * (step_x + step_y)

        alpha = frame[..., 3]
        alive = alpha > config.alpha_threshold
        if not alive.any():
            return

        span = max(1e-6, config.maturity_alpha - config.alpha_threshold)
        maturity = np.clip((alpha - config.alpha_threshold) / span, 0.0, 1.0)

        radius_jitter, jitter_x, jitter_y = self._jitter_fields(height, width)

        c = float(np.clip(cellularity, 0.0, 1.0))

        # gap = 1 when the cell is isolated, 0 when it overflows into its
        # neighbours. Only immature cells leave a gap, and only as much as the
        # current cellularity allows.
        gap = c * (1.0 - maturity)
        radius = step * (config.radius_mature - gap * (config.radius_mature - config.radius_young))
        radius = radius * (1.0 + config.radius_jitter * c * radius_jitter)

        rows, cols = np.nonzero(alive)

        # Painter's algorithm: the most mature cells go on top, so the surface
        # closes over the young ones instead of being pockmarked by them.
        order = np.argsort(alpha[rows, cols], kind='stable')
        rows, cols = rows[order], cols[order]

        centers_x = ((cols + 0.5) + config.position_jitter * c * jitter_x[rows, cols]) * step_x
        centers_y = ((rows + 0.5) + config.position_jitter * c * jitter_y[rows, cols]) * step_y

        colors = np.clip(frame[rows, cols, :3], 0.0, 1.0)
        alphas = np.clip(alpha[rows, cols] * config.alpha_gain, 0.0, 1.0)
        radii = np.maximum(radius[rows, cols], 0.25)

        # Python-level loop over ~10-20k cells: converting to lists first is
        # measurably faster than indexing numpy scalars inside the loop.
        xs = centers_x.tolist()
        ys = centers_y.tolist()
        rs = radii.tolist()
        reds = colors[:, 0].tolist()
        greens = colors[:, 1].tolist()
        blues = colors[:, 2].tolist()
        alphas = alphas.tolist()

        if config.shape == 'square':
            for x, y, r, cr, cg, cb, ca in zip(xs, ys, rs, reds, greens, blues, alphas):
                ctx.set_source_rgba(cr, cg, cb, ca)
                ctx.rectangle(x - r, y - r, r * 2.0, r * 2.0)
                ctx.fill()
        else:
            for x, y, r, cr, cg, cb, ca in zip(xs, ys, rs, reds, greens, blues, alphas):
                ctx.set_source_rgba(cr, cg, cb, ca)
                ctx.arc(x, y, r, 0.0, TAU)
                ctx.fill()

    def render(self, frame: np.ndarray, out_width: int, out_height: int,
               cellularity: float, antialias: bool = True) -> np.ndarray:
        """
        Draw one frame onto a fresh transparent layer.

        Returns uint8 RGBA with straight (non-premultiplied) colours, ready to
        be lit and composited over the tree.
        """
        surface, ctx = create_surface(out_width, out_height, (0.0, 0.0, 0.0, 0.0), antialias)
        self.draw(ctx, frame, out_width, out_height, cellularity)
        return surface_to_numpy(surface, out_width, out_height, unpremultiply=True)
