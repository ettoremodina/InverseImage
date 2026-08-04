"""
Colour: palette extraction from the reference image and the shared grading pass.

Kept apart from `rendering/` because neither piece draws anything -- they decide
what colours the renderers use (`palette`) and how every finished frame is
finally treated (`grading`).
"""

from .palette import (
    Palette,
    extract_palette,
    resolve_palette,
    background_color,
    tree_colors,
)
from .grading import Grader

__all__ = [
    'Palette',
    'extract_palette',
    'resolve_palette',
    'background_color',
    'tree_colors',
    'Grader',
]
