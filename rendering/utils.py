"""
Rendering utility functions.
"""

import numpy as np


def draw_line(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: list):
    """Draw a line using Bresenham's algorithm."""
    h, w = img.shape[:2]
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    while True:
        if 0 <= x1 < w and 0 <= y1 < h:
            img[y1, x1] = color
        
        if x1 == x2 and y1 == y2:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy


def get_time_dilated_indices(num_source_frames: int, num_target_frames: int, 
                             initial_repeats: float, decay_rate: float) -> np.ndarray:
    """
    Calculate indices for resampling frames with time dilation (slow start).
    
    Args:
        num_source_frames: Number of available source frames
        num_target_frames: Number of frames in the output video
        initial_repeats: Relative weight of the first frame
        decay_rate: Decay rate for weight of subsequent frames
        
    Returns:
        Array of indices (integers) of length num_target_frames
    """
    if num_source_frames == 0:
        return np.zeros(num_target_frames, dtype=int)
        
    # Calculate weight (duration) for each source frame
    indices = np.arange(num_source_frames)
    weights = initial_repeats * (decay_rate ** indices)
    
    # Calculate cumulative weight (time)
    cumulative_weights = np.cumsum(weights)
    total_weight = cumulative_weights[-1]
    
    # Map target frames to source frames
    # Target times are evenly spaced from 0 to total_weight
    target_times = np.linspace(0, total_weight, num_target_frames)
    
    # Find which source frame corresponds to each target time
    # searchsorted finds the first index where cumulative_weight >= target_time
    resampled_indices = np.searchsorted(cumulative_weights, target_times)
    
    # Clip to valid range
    return np.clip(resampled_indices, 0, num_source_frames - 1)

