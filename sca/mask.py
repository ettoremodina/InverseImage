"""
Mask loading and sampling utilities for attractor placement.

Provides two placement methods:
- random: Uniform random sampling within the mask
- edge: Edge detection-based sampling using Sobel operator on color gradients
"""

import numpy as np
from PIL import Image
from typing import List, Tuple, Literal
from scipy import ndimage
from .vector import Vector2D


AttractorPlacement = Literal['random', 'edge']


def load_mask(image_path: str) -> np.ndarray:
    """Load an image as a binary mask. Returns True where foreground exists."""
    img = Image.open(image_path).convert('RGBA')
    arr = np.array(img)
    
    alpha = arr[:, :, 3]
    mask = alpha > 0
    
    return mask


def load_image_grayscale(image_path: str) -> np.ndarray:
    """Load an image as grayscale for edge detection."""
    img = Image.open(image_path).convert('L')
    return np.array(img, dtype=np.float32) / 255.0


def get_valid_positions(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Get all (x, y) coordinates where the mask is True."""
    ys, xs = np.where(mask)
    return list(zip(xs.astype(float), ys.astype(float)))


def detect_edges_from_image(image_path: str, threshold: float = 0.1) -> np.ndarray:
    """
    Detect edges based on color gradients in the image using Sobel operator.
    
    Args:
        image_path: Path to the image
        threshold: Minimum gradient magnitude to consider as edge (0-1 scale)
    """
    gray = load_image_grayscale(image_path)
    
    sobel_x = ndimage.sobel(gray, axis=1)
    sobel_y = ndimage.sobel(gray, axis=0)
    
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_magnitude = edge_magnitude / edge_magnitude.max() if edge_magnitude.max() > 0 else edge_magnitude
    
    edges = edge_magnitude > threshold
    
    return edges, edge_magnitude


def save_edge_visualization(image_path: str, save_path: str, threshold: float = 0.1):
    """Save edge detection result as an image for visual inspection."""
    _, edge_magnitude = detect_edges_from_image(image_path, threshold)
    edge_img = (edge_magnitude * 255).astype(np.uint8)
    Image.fromarray(edge_img, mode='L').save(save_path)
    print(f"Saved edge detection to: {save_path}")


def get_edge_positions(image_path: str, mask: np.ndarray, threshold: float = 0.1) -> List[Tuple[int, int]]:
    """Get all (x, y) coordinates on the edges within the mask."""
    edges, _ = detect_edges_from_image(image_path, threshold)
    edges_in_mask = edges & mask
    ys, xs = np.where(edges_in_mask)
    return list(zip(xs.astype(float), ys.astype(float)))


def sample_attractors_random(mask: np.ndarray, num_attractors: int) -> List[Vector2D]:
    """Sample random positions uniformly from within the mask."""
    valid_positions = get_valid_positions(mask)
    
    if len(valid_positions) < num_attractors:
        num_attractors = len(valid_positions)
        print(f"Warning: Only {len(valid_positions)} valid positions available")
    
    indices = np.random.choice(len(valid_positions), size=num_attractors, replace=False)
    positions = [valid_positions[i] for i in indices]
    
    return [Vector2D(x, y) for x, y in positions]


def sample_attractors_edge(
    image_path: str, 
    mask: np.ndarray, 
    num_attractors: int,
    threshold: float = 0.1
) -> List[Vector2D]:
    """Sample positions from color gradient edges using Sobel edge detection."""
    edge_positions = get_edge_positions(image_path, mask, threshold)
    
    if len(edge_positions) == 0:
        print("Warning: No edges detected, falling back to random sampling")
        return sample_attractors_random(mask, num_attractors)
    
    if len(edge_positions) < num_attractors:
        num_attractors = len(edge_positions)
        print(f"Warning: Only {len(edge_positions)} edge positions available")
    
    indices = np.random.choice(len(edge_positions), size=num_attractors, replace=False)
    positions = [edge_positions[i] for i in indices]
    
    return [Vector2D(x, y) for x, y in positions]


def sample_attractors(
    mask: np.ndarray, 
    num_attractors: int, 
    method: AttractorPlacement = 'random',
    image_path: str = None,
    edge_threshold: float = 0.1
) -> List[Vector2D]:
    """
    Sample attractor positions from the mask.
    
    Args:
        mask: Binary mask where True indicates valid positions
        num_attractors: Number of attractors to sample
        method: Placement method - 'random' for uniform sampling, 'edge' for edge-based
        image_path: Path to image (required for edge method)
        edge_threshold: Minimum gradient magnitude for edge detection (0-1)
    """
    if method == 'edge':
        if image_path is None:
            raise ValueError("image_path is required for edge placement method")
        return sample_attractors_edge(image_path, mask, num_attractors, edge_threshold)
    return sample_attractors_random(mask, num_attractors)


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
