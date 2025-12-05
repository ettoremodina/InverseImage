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
    branch_base_width: float = 3.0
    branch_tip_width: float = 1.0
    
    antialiasing: bool = True
