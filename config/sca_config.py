"""
Configuration for Space Colonization Algorithm.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Literal
import numpy as np

AttractorPlacement = Literal['random', 'edge']

@dataclass
class SCAConfig:
    mask_image_path: Optional[str] = None

    num_attractors: int = 10000
    attractor_placement: AttractorPlacement = 'edge'
    edge_threshold: float = 0.1  # Lower values = more edges detected (0-1 scale)
    influence_radius: float = 50.0
    kill_distance: float = 1.0
    growth_step: float = 1.0

    branch_angle_threshold: float = 0.1
    min_attractors_per_branch: int = 2

    root_pos: Optional[Tuple[float, float]] = None
    max_iterations: int = 2000  # Increased for sparse attractor cases
    stagnation_limit: int = 100
    
    # Adaptive growth for sparse attractors
    adaptive_influence: bool = True  # Increase influence radius when stagnating
    influence_growth_rate: float = 1.1  # Multiply influence radius by this on stagnation
    max_influence_radius: float = 200.0  # Maximum influence radius
    
    # Stopping criteria based on attractor coverage
    min_attractor_kill_ratio: float = 0.95  # Stop if 95% of attractors are killed
    enable_attractor_stopping: bool = True  # Use attractor ratio as stopping criterion
    
    # Seed extraction
    seed_mode: str = 'tips'  # 'tips' or 'all'
    max_seeds: int = 1000

    animate: bool = False
    show_attractors: bool = True

    output_dir: str = 'outputs/sca'
    random_seed: Optional[int] = None

    def __post_init__(self):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

    @classmethod
    def from_pipeline(cls, pipeline_config) -> 'SCAConfig':
        """Create SCA Config from PipelineConfig."""
        return cls(
            mask_image_path=pipeline_config.target_image,
            num_attractors=pipeline_config.num_attractors,
            attractor_placement=pipeline_config.attractor_placement,
            influence_radius=pipeline_config.influence_radius,
            kill_distance=pipeline_config.kill_distance,
            growth_step=pipeline_config.growth_step,
            branch_angle_threshold=pipeline_config.branch_angle_threshold,
            min_attractors_per_branch=pipeline_config.min_attractors_per_branch,
            max_iterations=pipeline_config.max_iterations,
            stagnation_limit=pipeline_config.stagnation_limit,
            output_dir=str(pipeline_config.sca_output_dir),
            random_seed=pipeline_config.random_seed,
        )
