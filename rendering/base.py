"""
Base renderer class defining the interface for all renderers.
"""

import cairo
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Union
from config.render_config import SCARenderConfig, NCARenderConfig

from .utils import create_surface, surface_to_numpy


class Renderer(ABC):
    def __init__(self, config: Union[SCARenderConfig, NCARenderConfig]):
        self.config = config

    def _create_surface(self, background=None) -> Tuple[cairo.ImageSurface, cairo.Context]:
        return create_surface(
            self.config.output_width,
            self.config.output_height,
            self.config.background_color if background is None else background,
            getattr(self.config, 'antialiasing', True),
        )

    def _surface_to_numpy(self, surface: cairo.ImageSurface,
                          unpremultiply: bool = False) -> np.ndarray:
        return surface_to_numpy(
            surface, self.config.output_width, self.config.output_height, unpremultiply
        )

    def _compute_scale(self, source_width: int, source_height: int) -> Tuple[float, float]:
        scale_x = self.config.output_width / source_width
        scale_y = self.config.output_height / source_height
        return scale_x, scale_y

    @abstractmethod
    def render_frame(self, *args, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def render_animation(self, *args, **kwargs):
        pass
