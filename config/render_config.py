"""
Configuration for rendering module.

Note on `background_color`: it appears in both render configs because both
renderers can run standalone, but it is *not* set independently -- the pipeline
propagates one value (derived from the palette, see palette_config) into both.
Never edit the two by hand.
"""

from dataclasses import dataclass, field
from typing import Literal, Tuple

BranchWidthMode = Literal['depth', 'subtree']
CellShape = Literal['circle', 'square']


@dataclass
class CellRenderConfig:
    """
    NCA cells drawn as cells rather than as square pixels (PLAN a).

    The cellularity is a function of time, not a fixed property: a young cell is
    a small disc with a visible gap around it, a mature cell overflows the grid
    step and merges with its neighbours into a continuous surface. You see the
    colony while it grows and the tissue at the end.

    It never goes all the way to zero: at `cellularity_floor` a residue of
    per-cell irregularity survives, which -- together with the lighting -- makes
    the surface read as skin instead of as dead plastic.
    """

    enabled: bool = True
    shape: CellShape = 'circle'

    # 0 = smooth surface, 1 = marked colony. This is the value at the start of
    # the growth; it decays towards `cellularity_floor` following the curve.
    cellularity: float = 1.0
    cellularity_floor: float = 0.1
    cellularity_curve: float = 1.6   # exponent of the decay: >1 stays cellular longer

    # Cell radius as a fraction of the grid step.
    # `radius_mature` > 0.5 makes neighbours overlap, which is what closes the
    # gaps into a surface.
    radius_mature: float = 0.78
    radius_young: float = 0.34

    # Alpha at which a cell counts as fully mature.
    maturity_alpha: float = 0.85
    # Cells below this alpha are not drawn at all (also used as the maturity floor).
    alpha_threshold: float = 0.1
    # Opacity multiplier of the deposited colour.
    alpha_gain: float = 1.0

    # Deterministic per-cell irregularity, scaled by the current cellularity.
    radius_jitter: float = 0.18
    position_jitter: float = 0.10
    jitter_seed: int = 7


@dataclass
class LightingConfig:
    """
    Light instead of pixels (PLAN c).

    A normal map is built from the gradient of the smoothed alpha, then a
    lambertian term, a rim light and a soft bloom are applied. All 2-D numpy
    work on the rendered cell layer -- the NCA model never sees any of this.
    """

    enabled: bool = True

    # Normal map
    height_blur: float = 1.6      # sigma, in output pixels, of the alpha smoothing
    relief: float = 3.0           # how much the alpha gradient tilts the normals

    # Lambert
    light_direction: Tuple[float, float, float] = (-0.5, -0.7, 0.6)
    ambient: float = 0.62
    diffuse: float = 0.55

    # Rim light on the silhouette
    rim_enabled: bool = True
    rim_intensity: float = 0.45
    rim_power: float = 2.2
    rim_color: Tuple[float, float, float] = (0.75, 0.85, 1.0)

    # Bloom on the brightest cells
    bloom_enabled: bool = True
    bloom_threshold: float = 0.72
    bloom_intensity: float = 0.35
    bloom_blur: float = 9.0       # sigma in output pixels


@dataclass
class ScaffoldFadeConfig:
    """
    The scaffold disappears (PLAN 3.4).

    When the flesh covers a branch, that branch fades out. Both implementations
    the plan asks for are here, behind `mode`:

    - 'alpha' (default, the better looking one): the tree's opacity is driven by
      the NCA alpha sampled where the branch is, so a branch only disappears
      where it is actually covered;
    - 'time': the whole tree fades out on the schedule alone, regardless of what
      grew on top of it. Trivial, and useful as a comparison.
    """

    enabled: bool = True
    mode: Literal['alpha', 'time'] = 'alpha'

    strength: float = 0.95   # how far the fade goes: 1 = the branch vanishes entirely
    blur: float = 6.0        # blur of the coverage mask, in output pixels ('alpha' mode)
    coverage_gain: float = 1.6  # coverage is multiplied by this before being used


@dataclass
class SCARenderConfig:
    output_width: int = 512
    output_height: int = 512
    background_color: Tuple[float, float, float, float] = (0.08, 0.08, 0.10, 1.0)

    branch_color: Tuple[float, float, float, float] = (0.35, 0.20, 0.10, 1.0)
    branch_color_end: Tuple[float, float, float, float] = (0.10, 0.60, 0.30, 1.0)
    branch_base_width: float = 4.5
    branch_tip_width: float = 0.5

    # Number of depth bands the colour/width gradient is quantised into.
    # Every band is drawn as one batched Cairo stroke, so this trades a little
    # gradient smoothness for render speed. Above ~64 the banding is invisible.
    color_steps: int = 64

    # ==================== LINE QUALITY (PLAN f) ====================
    # 'depth'   : width follows the distance from the root (the old behaviour).
    # 'subtree' : width follows how much tree the branch carries, using Murray's
    #             law (w_parent^2 = sum w_child^2) -- the rule Leonardo observed.
    #             Two branches at the same depth no longer look identical when
    #             one holds half the crown and the other a dead twig.
    # In 'subtree' mode the colour gradient follows the same driver, so a thick
    # branch is also a trunk-coloured one.
    branch_width_mode: BranchWidthMode = 'subtree'

    # Visual remap of the subtree weight, not a change to Murray's law.
    # A tree with a few thousand leaves has a trunk weight of sqrt(leaves), so
    # the raw weights pile up near zero and almost every branch would come out
    # at tip width. The gamma spreads them back out; 1.0 = raw Murray.
    branch_width_gamma: float = 0.45

    # Catmull-Rom smoothing of the polylines: SCA emits one segment per growth
    # step, so the curves are polygonal. 0 or 1 = off.
    branch_smoothing: bool = True
    smoothing_subdivisions: int = 3

    # Thinning of the deepest levels, which today are a dense scribble.
    # A polyline is a candidate when it is a tip, ends past `prune_depth_start`
    # of the maximum depth, and is shorter than `prune_max_points` -- that is
    # the actual scribble; long deep branches are tentacles and are kept.
    # `prune_tips` is the fraction of candidates dropped. Deterministic.
    prune_tips: float = 0.5
    prune_depth_start: float = 0.5
    prune_max_points: int = 8
    prune_seed: int = 11

    sway_magnitude: float = 3
    sway_frequency: float = 5.0

    antialiasing: bool = True

    # Internal pixels per output pixel (the supersample factor). Set by the
    # pipeline, not by hand. Quantities expressed directly in output pixels --
    # line widths -- are multiplied by it; quantities expressed in source units
    # -- the sway -- already scale through the source->canvas mapping.
    render_scale: float = 1.0


@dataclass
class NCARenderConfig:
    output_width: int = 512
    output_height: int = 512
    background_color: Tuple[float, float, float, float] = (0.08, 0.08, 0.10, 1.0)

    alpha_threshold: float = 0.1  # cells below this alpha are not drawn

    # 0.0 = no smoothing, 0.9 = heavy. A little smoothing removes the flicker
    # of the noisy phase without smearing the growth.
    temporal_smoothing: float = 0.25

    # Frame persistence / Time dilation
    initial_repeats: int = 5  # Relative duration of first frame
    decay_rate: float = 0.99     # < 1.0 means early frames last longer (slow start)

    antialiasing: bool = True

    # Internal pixels per output pixel; set by the pipeline. The lighting radii
    # are expressed in output pixels and are scaled by it.
    render_scale: float = 1.0

    cells: CellRenderConfig = field(default_factory=CellRenderConfig)
    lighting: LightingConfig = field(default_factory=LightingConfig)
