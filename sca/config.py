"""
Configuration for Space Colonization Algorithm.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class SCAConfig:
    mask_image_path: str = 'images/ragnopiccolo.png'
    
    # For 512x512 image - tuned for spider shape:
    num_attractors: int = 5000
    influence_radius: float = 200.0  # Large enough to reach attractors from root
    kill_distance: float = 5.0       # Consume attractors when close
    growth_step: float = 3.0         # Branch segment length
    
    root_pos: Optional[Tuple[float, float]] = None
    max_iterations: int = 2000
    
    animate: bool = False
    show_attractors: bool = True
    
    output_dir: str = 'outputs/sca'
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
