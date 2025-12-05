"""
Unified configuration for the NCA-SCA pipeline.

All settings are derived from the target image path.
This is the single source of truth for the entire pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path
import json

import torch
import numpy as np


def get_device():
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


@dataclass
class ResolutionStage:
    size: int
    epochs: int
    batch_size: int
    accumulation_steps: int = 1

    @property
    def effective_batch_size(self):
        return self.batch_size * self.accumulation_steps


@dataclass
class PipelineConfig:
    """
    Unified configuration for the NCA-SCA pipeline.
    All output paths are derived from target_image.
    """
    
    # ==================== MAIN SETTING ====================
    target_image: str = 'images/Brini.png'
    
    # ==================== OUTPUT SETTINGS ====================
    output_base: str = 'outputs'
    
    # ==================== NCA SETTINGS ====================
    # Model architecture
    channel_n: int = 16
    hidden_size: int = 128
    update_rate: float = 0.5
    
    # Training
    target_size: int = 128
    target_padding: int = 16
    n_epochs: int = 2000
    batch_size: int = 8
    steps_per_epoch: int = 50
    
    # Optimizer
    lr: float = 2e-3
    lr_gamma: float = 0.9999
    betas: tuple = (0.5, 0.5)
    
    # Progressive training (empty = single resolution)
    progressive_stages: List[ResolutionStage] = field(default_factory=list)
    use_mixed_precision: bool = True
    
    # Seed positions from SCA (None = center seed, path = load from json)
    seed_positions_path: Optional[str] = None
    
    # Animation
    animation_steps: int = 100
    animation_fps: int = 20
    
    # ==================== SCA SETTINGS ====================
    num_attractors: int = 2000
    influence_radius: float = 15.0
    kill_distance: float = 2.0
    growth_step: float = 1.0
    branch_angle_threshold: float = 0.1
    min_attractors_per_branch: int = 2
    max_iterations: int = 800
    stagnation_limit: int = 100
    
    # Seed extraction
    seed_mode: str = 'tips'  # 'tips' or 'all'
    max_seeds: int = 100
    
    # ==================== RENDERING SETTINGS ====================
    render_size: int = 256
    render_fps: int = 20
    
    # ==================== MISC ====================
    device: str = None
    log_interval: int = 100
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        if self.device is None:
            self.device = get_device()
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
    
    # ==================== DERIVED PATHS ====================
    @property
    def image_name(self) -> str:
        return Path(self.target_image).stem
    
    @property
    def nca_output_dir(self) -> Path:
        return Path(self.output_base) / 'nca'
    
    @property
    def sca_output_dir(self) -> Path:
        return Path(self.output_base) / 'sca'
    
    @property
    def render_output_dir(self) -> Path:
        return Path(self.output_base) / 'rendering'
    
    # NCA paths
    @property
    def nca_model_path(self) -> Path:
        return self.nca_output_dir / f'{self.image_name}_model.pt'
    
    @property
    def nca_metadata_path(self) -> Path:
        return self.nca_output_dir / f'{self.image_name}_metadata.json'
    
    @property
    def nca_loss_path(self) -> Path:
        return self.nca_output_dir / f'{self.image_name}_loss.png'
    
    @property
    def nca_animation_path(self) -> Path:
        return self.nca_output_dir / f'{self.image_name}_animation.gif'
    
    # SCA paths
    @property
    def sca_render_data_path(self) -> Path:
        return self.sca_output_dir / f'{self.image_name}_render_data.json'
    
    @property
    def sca_metadata_path(self) -> Path:
        return self.sca_output_dir / f'{self.image_name}_metadata.json'
    
    @property
    def sca_tree_path(self) -> Path:
        return self.sca_output_dir / f'{self.image_name}_tree.png'
    
    @property
    def sca_seeds_path(self) -> Path:
        return self.sca_output_dir / f'{self.image_name}_seeds.json'
    
    # Rendering paths
    @property
    def render_sca_gif_path(self) -> Path:
        return self.render_output_dir / f'{self.image_name}_sca.gif'
    
    @property
    def render_nca_gif_path(self) -> Path:
        return self.render_output_dir / f'{self.image_name}_nca.gif'
    
    @property
    def render_combined_gif_path(self) -> Path:
        return self.render_output_dir / f'{self.image_name}_combined.gif'
    
    # ==================== SEED POSITIONS ====================
    def load_seed_positions(self) -> Optional[List[Tuple[int, int]]]:
        """Load seed positions from file if configured."""
        if self.seed_positions_path is None:
            return None
        
        path = Path(self.seed_positions_path)
        if not path.exists():
            print(f"Warning: seed positions file not found: {path}")
            return None
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return [tuple(p) for p in data]
        elif 'positions' in data:
            return [tuple(p) for p in data['positions']]
        
        return None
    
    def save_seed_positions(self, positions: List[Tuple[int, int]]):
        """Save seed positions to the default seeds file."""
        self.sca_output_dir.mkdir(parents=True, exist_ok=True)
        data = {
            'target_size': self.target_size,
            'positions': positions
        }
        with open(self.sca_seeds_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(positions)} seed positions to {self.sca_seeds_path}")
    
    # ==================== DIRECTORY CREATION ====================
    def create_output_dirs(self):
        """Create all output directories."""
        self.nca_output_dir.mkdir(parents=True, exist_ok=True)
        self.sca_output_dir.mkdir(parents=True, exist_ok=True)
        self.render_output_dir.mkdir(parents=True, exist_ok=True)


def load_config(path: str = 'config/pipeline.json') -> PipelineConfig:
    """Load config from JSON file, with defaults for missing fields."""
    config_path = Path(path)
    if not config_path.exists():
        return PipelineConfig()
    
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    if 'progressive_stages' in data:
        data['progressive_stages'] = [
            ResolutionStage(**stage) for stage in data['progressive_stages']
        ]
    
    return PipelineConfig(**data)


def save_config(config: PipelineConfig, path: str = 'config/pipeline.json'):
    """Save config to JSON file."""
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'target_image': config.target_image,
        'output_base': config.output_base,
        'channel_n': config.channel_n,
        'hidden_size': config.hidden_size,
        'update_rate': config.update_rate,
        'target_size': config.target_size,
        'target_padding': config.target_padding,
        'n_epochs': config.n_epochs,
        'batch_size': config.batch_size,
        'steps_per_epoch': config.steps_per_epoch,
        'lr': config.lr,
        'lr_gamma': config.lr_gamma,
        'use_mixed_precision': config.use_mixed_precision,
        'seed_positions_path': config.seed_positions_path,
        'animation_steps': config.animation_steps,
        'animation_fps': config.animation_fps,
        'num_attractors': config.num_attractors,
        'influence_radius': config.influence_radius,
        'kill_distance': config.kill_distance,
        'growth_step': config.growth_step,
        'branch_angle_threshold': config.branch_angle_threshold,
        'max_iterations': config.max_iterations,
        'stagnation_limit': config.stagnation_limit,
        'seed_mode': config.seed_mode,
        'max_seeds': config.max_seeds,
        'render_size': config.render_size,
        'render_fps': config.render_fps,
        'random_seed': config.random_seed,
    }
    
    if config.progressive_stages:
        data['progressive_stages'] = [
            {'size': s.size, 'epochs': s.epochs, 'batch_size': s.batch_size, 
             'accumulation_steps': s.accumulation_steps}
            for s in config.progressive_stages
        ]
    
    with open(config_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved config to {config_path}")
