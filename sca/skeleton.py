"""
Skeleton/seed extraction from SCA trees for NCA initialization.
"""

import numpy as np

from .tree import Tree
from .config import SCAConfig


def extract_seed_positions(tree: Tree, target_size: int, mode: str = 'tips', max_seeds: int = 50):
    """
    Extract positions from SCA tree and scale to NCA grid size.
    
    Args:
        tree: Grown SCA tree
        target_size: Target NCA grid size
        mode: 'tips' for branch tips only, 'all' for all branch endpoints
        max_seeds: Maximum number of seeds to return
    """
    mask_w, mask_h = tree.mask_width, tree.mask_height
    scale_x = target_size / mask_w
    scale_y = target_size / mask_h
    
    if mode == 'tips':
        positions = [(b.end_pos.x, b.end_pos.y) for b in tree.branch_tips]
    else:
        positions = [(b.end_pos.x, b.end_pos.y) for b in tree.branches]
    
    scaled = []
    for x, y in positions:
        nx = int(x * scale_x)
        ny = int(y * scale_y)
        nx = max(0, min(target_size - 1, nx))
        ny = max(0, min(target_size - 1, ny))
        scaled.append((nx, ny))
    
    unique = list(dict.fromkeys(scaled))
    
    if len(unique) > max_seeds:
        step = len(unique) // max_seeds
        unique = unique[::step][:max_seeds]
    
    return unique


def grow_sca_with_frames(config: SCAConfig, target_size: int, frame_skip: int = 2):
    """
    Grow SCA tree and collect frames scaled to target_size.
    
    Args:
        config: SCA configuration
        target_size: Size to scale frames to
        frame_skip: Collect frame every N iterations
        
    Returns:
        Tuple of (tree, frames) where frames is list of RGBA numpy arrays
    """
    from rendering.utils import draw_line
    
    tree = Tree(config)
    frames = []
    
    scale_x = target_size / tree.mask_width
    scale_y = target_size / tree.mask_height
    
    def render_frame():
        img = np.zeros((target_size, target_size, 4), dtype=np.float32)
        
        for branch in tree.branches:
            x1 = int(branch.start_pos.x * scale_x)
            y1 = int(branch.start_pos.y * scale_y)
            x2 = int(branch.end_pos.x * scale_x)
            y2 = int(branch.end_pos.y * scale_y)
            draw_line(img, x1, y1, x2, y2, color=[0.55, 0.27, 0.07, 1.0])
        
        return img
    
    frames.append(render_frame())
    
    while tree.grow_step():
        if tree.iteration % frame_skip == 0:
            frames.append(render_frame())
        if tree.iteration >= config.max_iterations:
            break
    
    frames.append(render_frame())
    
    return tree, frames
