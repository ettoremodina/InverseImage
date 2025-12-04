"""
Spatial partitioning for efficient nearest-neighbor queries.
Uses scipy's KDTree for O(log n) lookups instead of O(n) brute force.
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import List, Tuple
from .branch import Branch


class BranchSpatialIndex:
    """KD-Tree based spatial index for branch tips."""
    
    def __init__(self):
        self._tips: List[Branch] = []
        self._tree: cKDTree = None
        self._positions: np.ndarray = None
    
    def rebuild(self, branches: List[Branch]):
        self._tips = [b for b in branches if b.is_tip]
        
        if not self._tips:
            self._tree = None
            self._positions = None
            return
        
        self._positions = np.array([[b.end_pos.x, b.end_pos.y] for b in self._tips])
        self._tree = cKDTree(self._positions)
    
    def find_nearest(self, x: float, y: float) -> Tuple[Branch, float]:
        if self._tree is None:
            return None, float('inf')
        
        dist, idx = self._tree.query([x, y])
        return self._tips[idx], dist
    
    def find_within_radius(self, x: float, y: float, radius: float) -> List[Tuple[Branch, float]]:
        if self._tree is None:
            return []
        
        indices = self._tree.query_ball_point([x, y], radius)
        results = []
        for idx in indices:
            branch = self._tips[idx]
            dist = np.sqrt((x - branch.end_pos.x)**2 + (y - branch.end_pos.y)**2)
            results.append((branch, dist))
        
        return results
    
    @property
    def tips(self) -> List[Branch]:
        return self._tips
