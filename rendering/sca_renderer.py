"""
SCA tree renderer using Cairo.
Demonstrates upsampling: low-res simulation data -> high-res rendered output.
"""

import cairo
import numpy as np
import imageio
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from pathlib import Path

from config.render_config import SCARenderConfig
from utils.log import get_logger
from .base import Renderer
from .tree_weights import infer_weights_from_polylines
from .utils import max_polyline_depth, open_video_writer

logger = get_logger(__name__)


def prune_tips(polylines: List[Dict], config: SCARenderConfig) -> List[Dict]:
    """
    Thin out the deepest tips (PLAN f).

    The last levels of the SCA are a dense scribble: hundreds of twigs a few
    points long, all at the same width, which is what turns the silhouette into
    fuzz. The candidates are exactly those -- deep, short, and terminal; long
    deep branches are real structure and are always kept.
    """
    if config.prune_tips <= 0 or not polylines:
        return polylines

    max_depth = max((p['depths'][-1] for p in polylines), default=1)
    threshold = config.prune_depth_start * max_depth

    rng = np.random.default_rng(config.prune_seed)
    keep_roll = rng.random(len(polylines))

    def is_scribble(polyline, i):
        return (polyline.get('is_tip', False)
                and polyline['depths'][-1] >= threshold
                and len(polyline['points']) <= config.prune_max_points
                and keep_roll[i] < config.prune_tips)

    kept = [p for i, p in enumerate(polylines) if not is_scribble(p, i)]

    logger.info('SCA: pruned %d of %d polylines (deep short tips past depth %.0f)',
                len(polylines) - len(kept), len(polylines), threshold)
    return kept


def catmull_rom(points: np.ndarray, subdivisions: int) -> np.ndarray:
    """
    Catmull-Rom interpolation of one polyline.

    SCA emits one segment per growth step, so what comes out is a chain of
    straight pieces. Interpolating it turns the chain into an actual curve --
    the difference between a plant and a wire diagram.

    Returns the resampled points; the endpoints are preserved exactly.
    """
    n = len(points)
    if n < 3 or subdivisions < 2:
        return points

    # Duplicate the ends so the first and last segments have control points.
    padded = np.vstack([points[0], points, points[-1]])

    p0 = padded[:-3]
    p1 = padded[1:-2]
    p2 = padded[2:-1]
    p3 = padded[3:]

    t = np.linspace(0.0, 1.0, subdivisions, endpoint=False).reshape(-1, 1, 1)
    t2 = t * t
    t3 = t2 * t

    # Uniform Catmull-Rom basis (tension 0.5).
    curve = 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )

    # curve is [subdivisions, segments, 2] -> interleave into one polyline.
    flat = curve.transpose(1, 0, 2).reshape(-1, 2)
    return np.vstack([flat, points[-1]])


class TreeGeometry:
    """
    Flattened, render-ready form of an SCA polyline set.

    Every point of every polyline lives in one contiguous array, so the
    per-frame sway is a single numpy expression rather than a Python loop.

    Each point also carries a `value` in [0, 1]: 0 at the trunk, 1 at the tips.
    It drives both the colour and the width, and where it comes from is what
    `branch_width_mode` selects:

    - 'depth'   -> value = depth / max_depth (the old behaviour);
    - 'subtree' -> value = 1 - Murray subtree weight, so width follows how much
                   tree the branch carries rather than how far it is from the
                   root (PLAN f).

    The polylines are pre-split into `color_steps` bands of that value, so every
    piece in a band shares one colour and one line width — the whole band then
    becomes a single Cairo path and a single stroke, which is what turns tens
    of thousands of strokes per frame into a few dozen.

    `depths` stays separate and always drives the growth reveal: how the tree is
    drawn changed, when each piece appears did not.
    """

    __slots__ = ('x', 'y', 'depths', 'values', 'begins', 'ends',
                 'max_depth', 'band_pieces', 'color_steps')

    def __init__(self, polylines: List[Dict], config: SCARenderConfig):
        polylines = prune_tips(polylines, config)

        self.color_steps = max(1, config.color_steps)
        max_depth = max((p['depths'][-1] for p in polylines), default=1) or 1
        weights = self._polyline_weights(polylines, config)

        subdivisions = config.smoothing_subdivisions if config.branch_smoothing else 1

        point_arrays, depth_arrays, value_arrays = [], [], []
        for polyline, weight in zip(polylines, weights):
            points = np.asarray(polyline['points'], dtype=np.float64)
            depths = np.asarray(polyline['depths'], dtype=np.float64)

            if subdivisions > 1 and len(points) >= 3:
                smoothed = catmull_rom(points, subdivisions)
                # Depth is resampled linearly: the reveal has to stay monotonic
                # along the curve, and it already is a smooth quantity.
                original_t = np.linspace(0.0, 1.0, len(points))
                new_t = np.linspace(0.0, 1.0, len(smoothed))
                depths = np.interp(new_t, original_t, depths)
                points = smoothed

            if config.branch_width_mode == 'subtree':
                # Constant along the polyline: by construction it contains no
                # branching, so it carries the same amount of tree throughout.
                spread = weight ** config.branch_width_gamma
                values = np.full(len(points), 1.0 - spread, dtype=np.float64)
            else:
                values = np.clip(depths / max_depth, 0.0, 1.0)

            point_arrays.append(points)
            depth_arrays.append(depths)
            value_arrays.append(values)

        counts = [len(p) for p in point_arrays]
        offsets = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        self.begins = offsets[:-1]
        self.ends = offsets[1:]

        if point_arrays:
            points = np.concatenate(point_arrays)
            self.depths = np.concatenate(depth_arrays)
            self.values = np.concatenate(value_arrays)
        else:
            points = np.zeros((0, 2))
            self.depths = np.zeros(0)
            self.values = np.zeros(0)

        self.x = points[:, 0]
        self.y = points[:, 1]
        self.max_depth = float(self.depths.max()) if len(self.depths) else 1.0

        self.band_pieces = self._split_into_bands()

    def _polyline_weights(self, polylines: List[Dict], config: SCARenderConfig) -> np.ndarray:
        """Murray subtree weight of every polyline, normalised to 1 at the trunk."""
        if not polylines or config.branch_width_mode != 'subtree':
            return np.ones(len(polylines))

        if all('weight' in p for p in polylines):
            weights = np.array([p['weight'] for p in polylines], dtype=np.float64)
        else:
            logger.info('SCA: render data carries no subtree weights, '
                        'reconstructing them from the geometry')
            weights = infer_weights_from_polylines(polylines)

        return weights / max(float(weights.max()), 1e-9)

    def _split_into_bands(self) -> Dict[int, List[Tuple[int, int, int]]]:
        """
        Cut every polyline where it crosses a value-band boundary.

        In 'subtree' mode the value is constant along a polyline, so no cut ever
        happens and each polyline lands whole in one band. In 'depth' mode the
        value climbs along the polyline and this is what preserves the gradient
        inside a long branch.
        """
        bands: Dict[int, List[Tuple[int, int, int]]] = {}
        top = self.color_steps - 1

        for i, (begin, end) in enumerate(zip(self.begins, self.ends)):
            if end - begin < 2:
                continue
            band_of = np.clip(
                (self.values[begin:end] * self.color_steps).astype(np.int64), 0, top)
            # Boundaries where the band changes; pieces overlap by one point so
            # consecutive bands stay visually joined.
            cuts = [0, *(np.flatnonzero(np.diff(band_of)) + 1), end - begin]
            for a, b in zip(cuts, cuts[1:]):
                piece_end = min(b + 1, end - begin)
                if piece_end - a < 2:
                    continue
                bands.setdefault(int(band_of[a]), []).append(
                    (i, int(begin + a), int(begin + piece_end)))

        return bands

    def band_value(self, band: int) -> float:
        """Centre of a band, in [0, 1] -- the colour/width interpolation factor."""
        return min(1.0, (band + 0.5) / self.color_steps)

    def __len__(self):
        return len(self.begins)


class SCARenderer(Renderer):
    def __init__(self, config: SCARenderConfig = None):
        super().__init__(config or SCARenderConfig())
        self._cached_polylines = None
        self._cached_geometry = None

    def get_geometry(self, polylines: List[Dict]) -> TreeGeometry:
        """Build (and cache) the flattened geometry for a polyline set."""
        if self._cached_polylines is not polylines:
            self._cached_polylines = polylines
            self._cached_geometry = TreeGeometry(polylines, self.config)
        return self._cached_geometry

    def _swayed_coords(self, geom: TreeGeometry, scale_x: float, scale_y: float,
                       time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Screen-space coordinates for every point, with the idle sway applied.

        Sway amplitude grows quadratically with relative depth, so the trunk
        stays anchored and the tips move most.
        """
        y = geom.y * scale_y

        if self.config.sway_magnitude <= 0:
            return geom.x * scale_x, y

        # Sway is expressed in source units and only becomes pixels through
        # scale_x below, so it is already independent of the internal render
        # scale -- unlike the line widths, which are set in output pixels.
        t = geom.depths / geom.max_depth
        amplitude = self.config.sway_magnitude * (t * t)
        phase = time * self.config.sway_frequency + geom.depths * 0.2 + geom.y * 0.05
        return (geom.x + np.sin(phase) * amplitude) * scale_x, y

    def _visible_extent(self, geom: TreeGeometry, max_depth_limit: int):
        """
        How much of each polyline is revealed at this point in the growth.

        Returns (ends, fractions): `ends[i]` is the index one past the last
        fully revealed point of polyline i, and `fractions[i]` how far the
        growth has advanced into the following segment (0 when there is none).
        Interpolating that fraction makes the tree grow smoothly instead of
        jumping a whole segment at a time.
        """
        if max_depth_limit is None:
            return geom.ends, np.zeros(len(geom))

        # A segment is revealed once the growth reaches the depth of its far
        # end, so the continuous cut sits at max_depth_limit + 1.
        cut = max_depth_limit + 1
        ends = np.empty(len(geom), dtype=np.int64)
        fractions = np.zeros(len(geom))

        for i in range(len(geom)):
            begin, end = geom.begins[i], geom.ends[i]
            local = np.searchsorted(geom.depths[begin:end], cut, side='right')
            ends[i] = begin + local
            if 0 < local < end - begin:
                previous = geom.depths[begin + local - 1]
                span = geom.depths[begin + local] - previous
                if span > 0:
                    fractions[i] = (cut - previous) / span

        return ends, fractions

    def _draw_polylines(self, ctx: cairo.Context, geom: TreeGeometry,
                        scale_x: float, scale_y: float,
                        max_depth_limit: int = None, time: float = 0.0,
                        opacity: float = 1.0):
        """
        Stroke the tree, one batched path per value band.

        `opacity` scales every band's alpha: it is how the scaffold dissolves
        under the flesh (PLAN 3.4, 'time' mode).
        """
        if len(geom) == 0 or opacity <= 0.0:
            return

        r1, g1, b1, a1 = self.config.branch_color
        r2, g2, b2, a2 = self.config.branch_color_end
        scale = self.config.render_scale
        base_w = self.config.branch_base_width * scale
        tip_w = self.config.branch_tip_width * scale

        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)

        x, y = self._swayed_coords(geom, scale_x, scale_y, time)
        visible_ends, fractions = self._visible_extent(geom, max_depth_limit)

        for band in sorted(geom.band_pieces):
            t = geom.band_value(band)

            ctx.set_source_rgba(
                r1 + (r2 - r1) * t,
                g1 + (g2 - g1) * t,
                b1 + (b2 - b1) * t,
                (a1 + (a2 - a1) * t) * opacity,
            )
            ctx.set_line_width(base_w + (tip_w - base_w) * t)

            drew = False
            for poly, begin, end in geom.band_pieces[band]:
                visible_end = visible_ends[poly]
                if begin >= visible_end:
                    continue

                if visible_end < end:
                    stop, fraction = visible_end, fractions[poly]
                else:
                    stop, fraction = end, 0.0

                if stop - begin < 2 and fraction <= 0:
                    continue

                ctx.move_to(x[begin], y[begin])
                for k in range(begin + 1, stop):
                    ctx.line_to(x[k], y[k])

                if fraction > 0:
                    last = stop - 1
                    ctx.line_to(
                        x[last] + (x[last + 1] - x[last]) * fraction,
                        y[last] + (y[last + 1] - y[last]) * fraction,
                    )

                drew = True

            if drew:
                ctx.stroke()
            else:
                ctx.new_path()

    def render_frame(self, data: Dict[str, Any], max_depth_limit: int = None,
                     time: float = 0.0) -> np.ndarray:
        surface, ctx = self._create_surface()

        scale_x, scale_y = self._compute_scale(data['source_width'], data['source_height'])
        geom = self.get_geometry(data['polylines'])

        self._draw_polylines(ctx, geom, scale_x, scale_y, max_depth_limit, time=time)

        return self._surface_to_numpy(surface)

    def save_frame(self, data: Dict[str, Any], output_path: str, resolve=None):
        frame = self.render_frame(data)
        if resolve is not None:
            frame = resolve(frame, 0)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(output_path, frame)

    def render_animation(self, data: Dict[str, Any], output_path: str,
                         fps: int = 30, duration_seconds: float = None,
                         resolve=None):
        """
        Render the growth animation by progressively revealing branches by depth.

        Args:
            data: SCA render data
            output_path: Output video path
            fps: Frames per second
            duration_seconds: Target video length. When omitted, one frame is
                emitted per depth level, so the length follows the tree depth.
            resolve: optional callable applied to each finished frame, used to
                bring the supersampled canvas down to the video size and to run
                the grading pass.
        """
        max_depth = max_polyline_depth(data['polylines'])

        if duration_seconds is not None:
            num_frames = max(1, int(round(duration_seconds * fps)))
            depths = np.linspace(0, max_depth, num_frames).astype(int)
        else:
            depths = np.arange(0, max_depth + 1)

        time = 0.0
        dt = 1.0 / fps

        logger.info('Rendering %d SCA frames...', len(depths))
        with open_video_writer(output_path, fps) as writer:
            for i, depth in enumerate(tqdm(depths, desc="Rendering SCA frames")):
                frame = self.render_frame(data, max_depth_limit=int(depth), time=time)
                writer.append_data(frame if resolve is None else resolve(frame, i))
                time += dt

        logger.info('  Saved animation: %s', output_path)
