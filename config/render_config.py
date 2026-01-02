"""
Configuration for rendering module.
"""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class SCARenderConfig:
    output_width: int = 512
    output_height: int = 512
    background_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    
    branch_color: Tuple[float, float, float, float] = (0.35, 0.20, 0.10, 1.0)
    branch_color_end: Tuple[float, float, float, float] = (0.10, 0.60, 0.30, 1.0)
    branch_base_width: float = 4.5
    branch_tip_width: float = 0.5
    
    sway_magnitude: float = 3
    sway_frequency: float = 5.0
    
    antialiasing: bool = True


@dataclass
class NCARenderConfig:
    output_width: int = 512
    output_height: int = 512
    background_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    
    cell_shape: str = "square"  # "circle", "square", "hexagon"
    cell_scale: float = 1.0  # multiplier for cell size (1.0 = cells touch)
    alpha_threshold: float = 0.1  # cells below this alpha are not drawn
    
    temporal_smoothing: float = 0  # 0.0 = no smoothing, 0.9 = heavy smoothing
    
    # Frame persistence / Time dilation
    initial_repeats: int = 5  # Relative duration of first frame
    decay_rate: float = 0.99     # < 1.0 means early frames last longer (slow start)
    
    antialiasing: bool = True
