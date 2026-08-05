"""
Rendering module for high-resolution skinning of SCA/NCA simulations.
Uses Cairo for resolution-independent vector graphics.
"""

from config.render_config import (
    SCARenderConfig,
    NCARenderConfig,
    CellRenderConfig,
    LightingConfig,
    ScaffoldFadeConfig,
)
from .sca_renderer import SCARenderer
from .nca_renderer import NCARenderer, composite
from .timeline import TimelineRenderer, StageLayer, window_progress
from .cells import CellPainter, cellularity_at
from .lighting import apply_lighting
from .camera import Camera
from .supersample import resolve, internal_size
from .easing import apply_easing, get_easing
from .tree_weights import murray_weights, infer_weights_from_polylines
from .exporters import (
    export_sca_data,
    load_sca_data,
    export_nca_frames,
    load_nca_frames,
    build_polylines
)
from .animation import (
    frame_to_rgb,
    save_combined_animation,
    save_frames_as_gif,
    save_frame_as_image,
    render_seeds_image
)
from .utils import (
    draw_line,
    load_rgb_image,
    max_polyline_depth,
    open_video_writer,
    create_surface,
    surface_to_numpy,
)
