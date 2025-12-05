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
