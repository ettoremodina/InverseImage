"""
Space Colonization Algorithm (SCA) for 2D tree-like branching structures.

Based on: "Modeling Trees with a Space Colonization Algorithm" 
by Runions, Lane, and Prusinkiewicz (2007).
"""

from .attractor import Attractor
from .branch import Branch
from .tree import Tree
from .config import SCAConfig
from .visualization import visualize_tree, animate_growth
from .skeleton import extract_seed_positions, grow_sca_with_frames

__all__ = [
    'Attractor',
    'Branch', 
    'Tree',
    'SCAConfig',
    'visualize_tree',
    'animate_growth',
    'extract_seed_positions',
    'grow_sca_with_frames'
]
