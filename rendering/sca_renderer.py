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
from .base import Renderer
from .utils import max_polyline_depth, open_video_writer


class TreeGeometry:
    """
    Flattened, render-ready form of an SCA polyline set.

    Every point of every polyline lives in one contiguous array, so the
    per-frame sway is a single numpy expression rather than a Python loop.

    The polylines are also pre-split into `color_steps` depth bands. Colour and
    width vary continuously with depth, so quantising depth lets every piece in
    a band share one source colour and one line width — the whole band then
    becomes a single Cairo path and a single stroke, which is what turns tens
    of thousands of strokes per frame into a few dozen.
    """

    __slots__ = ('x', 'y', 'depths', 'begins', 'ends', 'max_depth', 'band_pieces')

    def __init__(self, polylines: List[Dict], color_steps: int):
        counts = [len(p['points']) for p in polylines]
        offsets = np.zeros(len(polylines) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        self.begins = offsets[:-1]
        self.ends = offsets[1:]

        if polylines:
            points = np.concatenate([np.asarray(p['points'], dtype=np.float64) for p in polylines])
            self.depths = np.concatenate([np.asarray(p['depths'], dtype=np.int64) for p in polylines])
        else:
            points = np.zeros((0, 2))
            self.depths = np.zeros(0, dtype=np.int64)

        self.x = points[:, 0]
        self.y = points[:, 1]
        self.max_depth = int(self.depths.max()) if len(self.depths) else 1

        self.band_pieces = self._split_into_bands(color_steps)

    def _split_into_bands(self, color_steps: int) -> Dict[int, List[Tuple[int, int, int]]]:
        """Cut every polyline where it crosses a depth-band boundary."""
        band_size = self.band_size(color_steps)
        bands: Dict[int, List[Tuple[int, int, int]]] = {}

        for i, (begin, end) in enumerate(zip(self.begins, self.ends)):
            if end - begin < 2:
                continue
            band_of = self.depths[begin:end] // band_size
            # Boundaries where the band changes; pieces overlap by one point so
            # consecutive bands stay visually joined.
            cuts = [0, *(np.flatnonzero(np.diff(band_of)) + 1), end - begin]
            for a, b in zip(cuts, cuts[1:]):
                piece_end = min(b + 1, end - begin)
                if piece_end - a < 2:
                    continue
                bands.setdefault(int(band_of[a]), []).append((i, begin + a, begin + piece_end))

        return bands

    def band_size(self, color_steps: int) -> int:
        return max(1, int(np.ceil((self.max_depth + 1) / max(1, color_steps))))

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
            self._cached_geometry = TreeGeometry(polylines, self.config.color_steps)
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
                        max_depth_limit: int = None, time: float = 0.0):
        """Stroke the tree, one batched path per depth band."""
        if len(geom) == 0:
            return

        r1, g1, b1, a1 = self.config.branch_color
        r2, g2, b2, a2 = self.config.branch_color_end
        base_w = self.config.branch_base_width
        tip_w = self.config.branch_tip_width

        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)

        x, y = self._swayed_coords(geom, scale_x, scale_y, time)
        visible_ends, fractions = self._visible_extent(geom, max_depth_limit)

        band_size = geom.band_size(self.config.color_steps)
        inv_max_depth = 1.0 / geom.max_depth if geom.max_depth > 0 else 0.0

        for band in sorted(geom.band_pieces):
            t = min(1.0, (band + 0.5) * band_size * inv_max_depth)

            ctx.set_source_rgba(
                r1 + (r2 - r1) * t,
                g1 + (g2 - g1) * t,
                b1 + (b2 - b1) * t,
                a1 + (a2 - a1) * t,
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

    def save_frame(self, data: Dict[str, Any], output_path: str):
        frame = self.render_frame(data)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(output_path, frame)

    def render_animation(self, data: Dict[str, Any], output_path: str,
                         fps: int = 30, duration_seconds: float = None):
        """
        Render the growth animation by progressively revealing branches by depth.

        Args:
            data: SCA render data
            output_path: Output video path
            fps: Frames per second
            duration_seconds: Target video length. When omitted, one frame is
                emitted per depth level, so the length follows the tree depth.
        """
        max_depth = max_polyline_depth(data['polylines'])

        if duration_seconds is not None:
            num_frames = max(1, int(round(duration_seconds * fps)))
            depths = np.linspace(0, max_depth, num_frames).astype(int)
        else:
            depths = np.arange(0, max_depth + 1)

        time = 0.0
        dt = 1.0 / fps

        print(f"Rendering {len(depths)} SCA frames...")
        with open_video_writer(output_path, fps) as writer:
            for depth in tqdm(depths, desc="Rendering SCA frames"):
                writer.append_data(self.render_frame(data, max_depth_limit=int(depth), time=time))
                time += dt

        print(f"  Saved animation: {output_path}")
