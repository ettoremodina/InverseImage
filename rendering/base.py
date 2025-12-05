"""
Base renderer class defining the interface for all renderers.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np

from .config import RenderConfig


class Renderer(ABC):
    def __init__(self, config: RenderConfig):
        self.config = config
    
    @abstractmethod
    def render_frame(self, frame_data: dict) -> np.ndarray:
        pass
    
    @abstractmethod
    def render_animation(self, frames_data: List[dict], output_path: str):
        pass
