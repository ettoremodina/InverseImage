"""
Mask loading and sampling utilities.
"""

import numpy as np
from PIL import Image
from typing import List, Tuple
from .vector import Vector2D


def load_mask(image_path: str) -> np.ndarray:
    """Load an image as a binary mask. Returns True where foreground exists."""
    img = Image.open(image_path).convert('RGBA')
    arr = np.array(img)
    
    alpha = arr[:, :, 3]
    mask = alpha > 0
    
    return mask


def get_valid_positions(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Get all (x, y) coordinates where the mask is True."""
    ys, xs = np.where(mask)
    return list(zip(xs.astype(float), ys.astype(float)))


def sample_attractors(mask: np.ndarray, num_attractors: int) -> List[Vector2D]:
    """Sample random positions from within the mask."""
    valid_positions = get_valid_positions(mask)
    
    if len(valid_positions) < num_attractors:
        num_attractors = len(valid_positions)
        print(f"Warning: Only {len(valid_positions)} valid positions available")
    
    indices = np.random.choice(len(valid_positions), size=num_attractors, replace=False)
    positions = [valid_positions[i] for i in indices]
    
    return [Vector2D(x, y) for x, y in positions]


def find_bottom_center(mask: np.ndarray) -> Vector2D:
    """Find the bottom-center point of the mask for root placement."""
    ys, xs = np.where(mask)
    
    max_y = np.max(ys)
    bottom_xs = xs[ys == max_y]
    center_x = np.mean(bottom_xs)
    
    return Vector2D(center_x, max_y)


def get_mask_dimensions(mask: np.ndarray) -> Tuple[int, int]:
    """Return (width, height) of the mask."""
    return mask.shape[1], mask.shape[0]
