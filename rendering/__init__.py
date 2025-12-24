"""
Rendering module for high-resolution skinning of SCA/NCA simulations.
Uses Cairo for resolution-independent vector graphics.
"""

from config.render_config import RenderConfig, NCARenderConfig
from .sca_renderer import SCARenderer
from .nca_renderer import NCARenderer
from .combined_renderer import CombinedRenderer
from .exporters import (
    export_sca_data, 
    load_sca_data,
    export_nca_frames,
    load_nca_frames
)
from .animation import (
    frame_to_rgb,
    save_combined_animation,
    save_frames_as_gif,
    save_frame_as_image,
    render_seeds_image
)
from .utils import draw_line
