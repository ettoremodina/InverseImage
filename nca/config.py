"""
Configuration module for Neural Cellular Automata.
Contains all hyperparameters and settings.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import torch


def get_device():
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


@dataclass
class ResolutionStage:
    """Configuration for one resolution stage in progressive training."""
    size: int
    epochs: int
    batch_size: int
    accumulation_steps: int = 1  # Gradient accumulation steps
    
    @property
    def effective_batch_size(self):
        return self.batch_size * self.accumulation_steps


@dataclass
class Config:
    # Model architecture
    channel_n: int = 16
    hidden_size: int = 128
    update_rate: float = 0.5
    
    # Image settings
    image_path: str = 'images/ragnopiccolo.png'
    target_size: int = 128  # Used for single-resolution training
    target_padding: int = 16
    
    # Training settings
    n_epochs: int = 2000
    batch_size: int = 8
    steps_per_epoch: int = 50
    
    # Optimizer settings
    lr: float = 2e-3
    lr_gamma: float = 0.9999
    betas: tuple = (0.5, 0.5)
    
    # Progressive training stages: (size, epochs, batch_size, accumulation_steps)
    # At higher resolutions, we use smaller batches but accumulate gradients
    # to maintain effective batch size while fitting in memory
    progressive_stages: List[ResolutionStage] = field(default_factory=lambda: [
        ResolutionStage(size=40, epochs=800, batch_size=8, accumulation_steps=1),
        ResolutionStage(size=128, epochs=800, batch_size=2, accumulation_steps=4),
        # ResolutionStage(size=256, epochs=200, batch_size=1, accumulation_steps=8),
    ])
    
    # Mixed precision training (reduces memory by ~50%)
    use_mixed_precision: bool = True
    
    # Output settings
    output_dir: str = 'outputs'
    save_gif: bool = True
    animation_steps: int = steps_per_epoch * 2
    
    # Misc
    device: str = None
    log_interval: int = 100
    
    def __post_init__(self):
        if self.device is None:
            self.device = get_device()
