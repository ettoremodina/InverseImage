"""
Configuration module for Neural Cellular Automata.
Contains all hyperparameters and settings.

Note: For pipeline usage, prefer using config.PipelineConfig which provides
unified configuration for NCA, SCA, and rendering.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
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
    accumulation_steps: int = 1

    @property
    def effective_batch_size(self):
        return self.batch_size * self.accumulation_steps


@dataclass
class Config:
    channel_n: int = 16
    hidden_size: int = 128
    update_rate: float = 0.5

    image_path: str = 'images/Brini.png'
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

    @classmethod
    def from_pipeline(cls, pipeline_config) -> 'Config':
        """Create NCA Config from PipelineConfig."""
        from config.pipeline import ResolutionStage as PipelineStage
        
        stages = []
        for s in pipeline_config.progressive_stages:
            stages.append(ResolutionStage(
                size=s.size, epochs=s.epochs,
                batch_size=s.batch_size, accumulation_steps=s.accumulation_steps
            ))
        
        return cls(
            channel_n=pipeline_config.channel_n,
            hidden_size=pipeline_config.hidden_size,
            update_rate=pipeline_config.update_rate,
            image_path=pipeline_config.target_image,
            target_size=pipeline_config.target_size,
            target_padding=pipeline_config.target_padding,
            n_epochs=pipeline_config.n_epochs,
            batch_size=pipeline_config.batch_size,
            steps_per_epoch=pipeline_config.steps_per_epoch,
            lr=pipeline_config.lr,
            lr_gamma=pipeline_config.lr_gamma,
            progressive_stages=stages,
            use_mixed_precision=pipeline_config.use_mixed_precision,
            seed_positions=pipeline_config.load_seed_positions(),
            output_dir=str(pipeline_config.nca_output_dir),
            animation_steps=pipeline_config.animation_steps,
            device=pipeline_config.device,
            log_interval=pipeline_config.log_interval,
        )
