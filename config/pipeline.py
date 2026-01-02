"""
Unified configuration for the NCA-SCA pipeline.

All settings are derived from the target image path.
This is the single source of truth for the entire pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path
import json
import numpy as np

from .common import ResolutionStage, get_device
from .nca_config import NCAConfig
from .sca_config import SCAConfig
from .render_config import SCARenderConfig, NCARenderConfig
from .particle_config import ParticleConfig

@dataclass
class PipelineConfig:
    """
    Unified configuration for the NCA-SCA pipeline.
    All output paths are derived from target_image.
    """
    
    # ==================== MAIN SETTING ====================
    target_image: str = 'images/Brini.png'
    output_base: str = 'outputs'
    
    # ==================== SUB-CONFIGS ====================
    nca: NCAConfig = field(default_factory=NCAConfig)
    sca: SCAConfig = field(default_factory=SCAConfig)
    sca_render: SCARenderConfig = field(default_factory=SCARenderConfig)
    nca_render: NCARenderConfig = field(default_factory=NCARenderConfig)
    particles: ParticleConfig = field(default_factory=ParticleConfig)
    
    # ==================== PIPELINE SPECIFIC ====================
    # Seed positions from SCA (None = center seed, path = load from json)
    seed_positions_path: Optional[str] = "outputs/sca/Brini_seeds.json"
    
    # Combined animation settings
    total_video_duration_seconds: float = 20.0
    sca_percentage: float = 0.4  # 40% of video for SCA growth
    nca_percentage: float = 0.6  # 60% of video for NCA growth
    
    # Animation
    animation_steps: int = 100
    animation_fps: int = 20
    render_size: int = 512
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
            
        # Propagate shared settings to sub-configs
        self.nca.image_path = self.target_image
        self.nca.output_dir = str(self.nca_output_dir)
        self.nca.device = self.device
        self.nca.animation_steps = self.animation_steps
        
        self.sca.mask_image_path = self.target_image
        self.sca.output_dir = str(self.sca_output_dir)
        self.sca.random_seed = self.random_seed
        
        self.nca_render.output_width = self.render_size
        self.nca_render.output_height = self.render_size
        
        self.sca_render.output_width = self.render_size 
        self.sca_render.output_height = self.render_size
    
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
            'target_size': self.nca.target_size,
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
    """
    Load config. 
    Returns the default configuration.
    """
    return PipelineConfig()


def save_config(config: PipelineConfig, path: str = 'config/pipeline.json'):
    """Save config to JSON file (Deprecated but kept for compatibility if needed)."""
    # We don't really need to save the full config anymore since it's code-based,
    # but we can save a summary.
    pass
