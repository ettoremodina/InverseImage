"""
Spatial partitioning for efficient nearest-neighbor queries.
Uses scipy's KDTree for O(log n) lookups instead of O(n) brute force.
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import List, Tuple, Optional
from .branch import Branch
from .profiling import profile


class BranchSpatialIndex:
    """KD-Tree based spatial index for branch tips."""
    
    def __init__(self):
        self._tips: List[Branch] = []
        self._tree: cKDTree = None
        self._positions: np.ndarray = None
    
    @profile
    def rebuild(self, branches: List[Branch]):
        self._tips = [b for b in branches if b.is_tip]
        
        if not self._tips:
            self._tree = None
            self._positions = None
            return
        
        self._positions = np.array([[b.end_pos.x, b.end_pos.y] for b in self._tips])
        self._tree = cKDTree(self._positions)
    
    def query_batch(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query nearest tip for multiple positions at once.
        Returns (distances, indices) arrays.
        """
        if self._tree is None or len(positions) == 0:
            return np.array([]), np.array([])
        return self._tree.query(positions)
    
    def find_nearest(self, x: float, y: float) -> Tuple[Optional[Branch], float]:
        if self._tree is None:
            return None, float('inf')
        
        dist, idx = self._tree.query([x, y])
        return self._tips[idx], dist
    
    @property
    def tips(self) -> List[Branch]:
        return self._tips
    
    @property 
    def tree(self) -> Optional[cKDTree]:
        return self._tree
