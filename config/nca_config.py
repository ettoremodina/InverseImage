"""
Configuration for Neural Cellular Automata.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from .common import ResolutionStage, get_device

@dataclass
class NCAConfig:
    channel_n: int = 16
    hidden_size: int = 128
    update_rate: float = 0.5

    image_path: Optional[str] = None
    target_size: int = 128
    target_padding: int = 16

    nca_steps: int = 64  # Steps for inference/demo

    # Training
    n_epochs: int = 2000
    batch_size: int = 8
    steps_per_epoch: int = 50
    steps_variance: float = 0.0  # Standard deviation for Gaussian distribution of steps
    
    # Persistence (Pool)
    use_pattern_pool: bool = True
    pool_size: int = 1024

    lr: float = 2e-3
    lr_gamma: float = 0.9999
    betas: tuple = (0.5, 0.5)

    progressive_stages: List[ResolutionStage] = field(default_factory=list)
    use_mixed_precision: bool = True

    seed_positions: Optional[List[Tuple[int, int]]] = None

    output_dir: str = 'outputs'
    save_gif: bool = True
    animation_steps: int = 100

    checkpoint_interval: int = 0
    device: str = None
    log_interval: int = 100

    def __post_init__(self):
        if self.device is None:
            self.device = get_device()


# Alias for backward compatibility with pickled models
Config = NCAConfig
