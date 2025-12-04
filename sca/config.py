"""
Configuration for Space Colonization Algorithm.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class SCAConfig:
    mask_image_path: str = 'images/ragnopiccolo.png'
    
    num_attractors: int = 4000        # More attractors = denser result
    influence_radius: float = 15.0   # Slightly smaller for more local growth
    kill_distance: float = 2.0        # Smaller = branches get closer to attractors
    growth_step: float = 1.0          # Smaller = finer detail
    
    # Branching: lower threshold = more branching
    # 0.3 = ~70 degrees, 0.5 = ~60 degrees, 0.7 = ~45 degrees
    branch_angle_threshold: float = 0.1
    min_attractors_per_branch: int = 2
    
    root_pos: Optional[Tuple[float, float]] = None
    max_iterations: int = 300
    stagnation_limit: int = 100  # Stop if no attractors die for this many iterations
    
    animate: bool = False
    show_attractors: bool = True
    
    output_dir: str = 'outputs/sca'
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
