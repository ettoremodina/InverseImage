"""
Base renderer class defining the interface for all renderers.
"""

import cairo
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Any
from .config import RenderConfig, NCARenderConfig


class Renderer(ABC):
    def __init__(self, config: Union[RenderConfig, NCARenderConfig]):
        self.config = config
    
    def _create_surface(self) -> Tuple[cairo.ImageSurface, cairo.Context]:
        surface = cairo.ImageSurface(
            cairo.FORMAT_ARGB32,
            self.config.output_width,
            self.config.output_height
        )
        ctx = cairo.Context(surface)
        
        # Check if antialiasing is in config (both RenderConfig and NCARenderConfig have it now)
        if getattr(self.config, 'antialiasing', True):
            ctx.set_antialias(cairo.ANTIALIAS_BEST)
        
        r, g, b, a = self.config.background_color
        ctx.set_source_rgba(r, g, b, a)
        ctx.paint()
        
        return surface, ctx

    def _surface_to_numpy(self, surface: cairo.ImageSurface) -> np.ndarray:
        buf = surface.get_data()
        arr = np.ndarray(
            shape=(self.config.output_height, self.config.output_width, 4),
            dtype=np.uint8,
            buffer=buf
        )
        arr_copy = arr.copy()
        arr_rgba = np.zeros_like(arr_copy)
        arr_rgba[:, :, 0] = arr_copy[:, :, 2]  # R
        arr_rgba[:, :, 1] = arr_copy[:, :, 1]  # G
        arr_rgba[:, :, 2] = arr_copy[:, :, 0]  # B
        arr_rgba[:, :, 3] = arr_copy[:, :, 3]  # A
        return arr_rgba

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
