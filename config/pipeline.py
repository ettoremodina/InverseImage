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

from .common import get_device
from .camera_config import CameraConfig
from .grading_config import GradingConfig
from .nca_config import NCAConfig
from .palette_config import PaletteConfig
from .sca_config import SCAConfig
from .render_config import SCARenderConfig, NCARenderConfig, ScaffoldFadeConfig
from .particle_config import ParticleConfig
from .seeding_config import ProgressiveSeedingConfig
from .timing_config import TimingConfig

@dataclass
class PipelineConfig:
    """
    Unified configuration for the NCA-SCA pipeline.
    All output paths are derived from target_image.
    """

    # ==================== MAIN SETTING ====================
    target_image: str = 'images/jellyfish.png'
    output_base: str = 'outputs'

    # ==================== SUB-CONFIGS ====================
    nca: NCAConfig = field(default_factory=NCAConfig)
    sca: SCAConfig = field(default_factory=SCAConfig)
    sca_render: SCARenderConfig = field(default_factory=SCARenderConfig)
    nca_render: NCARenderConfig = field(default_factory=NCARenderConfig)
    particles: ParticleConfig = field(default_factory=ParticleConfig)

    palette: PaletteConfig = field(default_factory=PaletteConfig)
    grading: GradingConfig = field(default_factory=GradingConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    scaffold: ScaffoldFadeConfig = field(default_factory=ScaffoldFadeConfig)
    seeding: ProgressiveSeedingConfig = field(default_factory=ProgressiveSeedingConfig)

    # ==================== PIPELINE SPECIFIC ====================
    # Seed positions from SCA (None = center seed, path = load from json)
    seed_positions_path: Optional[str] = None

    # Legacy combined animation settings (mode 'combined').
    # The timeline mode ignores these and uses `timing` instead.
    total_video_duration_seconds: float = 20.0
    sca_percentage: float = 0.4  # 40% of video for SCA growth
    nca_percentage: float = 0.6  # 60% of video for NCA growth

    # Animation
    animation_steps: int = 100
    animation_fps: int = 20
    render_size: int = 512
    render_fps: int = 20

    # Everything is drawn on a canvas `render_supersample` times the video size
    # and reduced with an area average: real antialiasing, thin SCA branches
    # that stop shimmering, soft cell edges. 1 = disabled.
    # The camera crop shares this same canvas -- there is never a second
    # enlargement (PLAN b, 3.5).
    render_supersample: int = 2

    # ==================== MISC ====================
    device: str = None
    log_interval: int = 100
    random_seed: Optional[int] = None

    def __post_init__(self):
        if not Path(self.target_image).exists():
            raise FileNotFoundError(
                f"target_image not found: {self.target_image}\n"
                f"Set PipelineConfig.target_image in config/pipeline.py to an existing file."
            )

        if self.device is None:
            self.device = get_device()
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
        # Automate seed positions path if not provided
        if self.seed_positions_path is None:
            self.seed_positions_path = str(self.sca_seeds_path)

        # Propagate shared settings to sub-configs
        self.nca.image_path = self.target_image
        self.nca.output_dir = str(self.nca_output_dir)
        self.nca.device = self.device
        self.nca.animation_steps = self.animation_steps
        
        self.sca.mask_image_path = self.target_image
        self.sca.output_dir = str(self.sca_output_dir)
        self.sca.random_seed = self.random_seed

        # Both renderers draw on the shared internal canvas; the reduction to
        # `render_size` happens once, at the end of the frame.
        canvas = self.canvas_size
        for render_config in (self.nca_render, self.sca_render):
            render_config.output_width = canvas
            render_config.output_height = canvas
            render_config.render_scale = float(max(1, self.render_supersample))

        self.apply_palette()

    # ==================== DERIVED RENDER GEOMETRY ====================
    @property
    def canvas_size(self) -> int:
        """Side of the internal canvas every renderer draws on."""
        return self.render_size * max(1, self.render_supersample)

    def apply_palette(self):
        """
        Push the palette derived from the target image into the render configs.

        This is the one place where the background colour is decided, so the two
        hardcoded backgrounds of the old render_config can no longer drift apart
        (PLAN d). Manual overrides in `palette` always win.
        """
        if not self.palette.enabled:
            return

        # Imported here: config must stay importable without the render stack.
        from color.palette import background_color, resolve_palette, tree_colors

        palette = resolve_palette(self.target_image, self.palette)
        if palette is None:
            return

        background = background_color(palette, self.palette)
        self.sca_render.background_color = background
        self.nca_render.background_color = background

        if self.palette.tree_from_palette:
            base, tip = tree_colors(palette, self.palette)
            self.sca_render.branch_color = base
            self.sca_render.branch_color_end = tip

        if self.grading.shadow_tint is None:
            self.grading.shadow_tint = background[:3]


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


def load_config() -> PipelineConfig:
    """
    Return the pipeline configuration.

    The config is code-based: edit the dataclass defaults in this file.
    All scripts go through this function so they always agree on paths.
    """
    return PipelineConfig()
