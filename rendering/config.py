"""
Configuration for rendering module.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class RenderConfig:
    output_width: int = 1024
    output_height: int = 1024
    background_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    
    branch_color: Tuple[float, float, float, float] = (0.35, 0.20, 0.10, 1.0)
    branch_color_end: Tuple[float, float, float, float] = (0.10, 0.60, 0.30, 1.0)
    branch_base_width: float = 5.0
    branch_tip_width: float = 0.5
    
    sway_magnitude: float = 2.5
    sway_frequency: float = 2.0
    
    antialiasing: bool = True


@dataclass
class NCARenderConfig:
    output_width: int = 512
    output_height: int = 512
    background_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    
    cell_shape: str = "circle"  # "circle", "square", "hexagon"
    cell_scale: float = 1.0  # multiplier for cell size (1.0 = cells touch)
    alpha_threshold: float = 0.1  # cells below this alpha are not drawn
    
    # Aesthetic refinements
    use_metaballs: bool = True
    metaball_threshold: float = 0.5
    metaball_blur_radius: float = 1.5  # Multiplier for cell radius
    
    use_breathing: bool = True
    breathing_factor: float = 0.5  # How much alpha affects size
    
    use_fake_3d: bool = True
    specular_intensity: float = 0.8
    shininess: float = 10.0
    light_pos: Tuple[float, float, float] = (1.0, 1.0, 2.0)
    
    temporal_smoothing: float = 0.5  # 0.0 = no smoothing, 0.9 = heavy smoothing
    
    antialiasing: bool = True
