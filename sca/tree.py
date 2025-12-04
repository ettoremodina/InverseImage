"""
Tree class - manages the entire Space Colonization Algorithm system.

The SCA works by having attractors "claim" the nearest branch tip.
Each branch tip grows toward the average direction of its claimed attractors.
Branching occurs naturally when different tips are claimed by different attractor groups.
"""

from typing import List, Dict, Optional, Callable, Set
import numpy as np

from .config import SCAConfig
from .vector import Vector2D
from .attractor import Attractor
from .branch import Branch
from .spatial import BranchSpatialIndex
from .mask import load_mask, sample_attractors, find_bottom_center, get_mask_dimensions


class Tree:
    def __init__(self, config: SCAConfig):
        self.config = config
        self.attractors: List[Attractor] = []
        self.branches: List[Branch] = []
        self.spatial_index = BranchSpatialIndex()
        self.iteration = 0
        self.mask: np.ndarray = None
        self.mask_width: int = 0
        self.mask_height: int = 0
        
        self._initialize()
    
    def _initialize(self):
        self.mask = load_mask(self.config.mask_image_path)
        self.mask_width, self.mask_height = get_mask_dimensions(self.mask)
        
        attractor_positions = sample_attractors(self.mask, self.config.num_attractors)
        self.attractors = [Attractor(pos) for pos in attractor_positions]
        
        if self.config.root_pos is None:
            root_pos = find_bottom_center(self.mask)
        else:
            root_pos = Vector2D(*self.config.root_pos)
        
        initial_direction = Vector2D(0, -1)
        root_end = root_pos + initial_direction * self.config.growth_step
        root_branch = Branch(root_pos, root_end, parent=None)
        self.branches.append(root_branch)
        
        self.spatial_index.rebuild(self.branches)
        
        print(f"Initialized Tree:")
        print(f"  Mask size: {self.mask_width}x{self.mask_height}")
        print(f"  Attractors: {len(self.attractors)}")
        print(f"  Root position: {root_pos}")
    
    def _is_inside_mask(self, pos: Vector2D) -> bool:
        x, y = int(pos.x), int(pos.y)
        if x < 0 or x >= self.mask_width or y < 0 or y >= self.mask_height:
            return False
        return self.mask[y, x]
    
    def _associate_attractors(self) -> Dict[Branch, List[Attractor]]:
        """
        For each attractor within influence radius of ANY tip, find its CLOSEST tip.
        That attractor then influences only that one tip.
        Returns mapping of branch tips to their list of influencing attractors.
        """
        influence_radius = self.config.influence_radius
        kill_distance = self.config.kill_distance
        
        branch_influences: Dict[Branch, List[Attractor]] = {}
        attractors_to_kill: Set[Attractor] = set()
        
        tips = self.spatial_index.tips
        if not tips:
            return {}
        
        for attractor in self.attractors:
            if not attractor.alive:
                continue
            
            min_dist = float('inf')
            closest_tip = None
            
            for tip in tips:
                dist = attractor.position.distance_to(tip.end_pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_tip = tip
            
            if closest_tip is None:
                continue
            
            if min_dist < kill_distance:
                attractors_to_kill.add(attractor)
                continue
            
            if min_dist < influence_radius:
                if closest_tip not in branch_influences:
                    branch_influences[closest_tip] = []
                branch_influences[closest_tip].append(attractor)
        
        for attractor in attractors_to_kill:
            attractor.kill()
        
        return branch_influences
    
    def _grow_branches(self, branch_influences: Dict[Branch, List[Attractor]]) -> List[Branch]:
        """Create new branches based on attractor influences."""
        new_branches = []
        
        for branch, attractors in branch_influences.items():
            if not attractors:
                continue
            
            avg_direction = Vector2D(0, 0)
            for attractor in attractors:
                direction = (attractor.position - branch.end_pos).normalize()
                avg_direction = avg_direction + direction
            
            avg_direction = avg_direction.normalize()
            
            new_branch = branch.create_child(avg_direction, self.config.growth_step)
            new_branches.append(new_branch)
        
        return new_branches
    
    def _cleanup_attractors(self):
        """Remove dead attractors from the list."""
        self.attractors = [a for a in self.attractors if a.alive]
    
    def _reset_branch_counts(self):
        """Reset the influence count on all branches."""
        for branch in self.branches:
            branch.reset_count()
    
    def grow_step(self) -> bool:
        """
        Perform one growth iteration.
        Returns True if growth occurred, False if complete.
        """
        if not self.attractors:
            return False
        
        self._reset_branch_counts()
        
        branch_influences = self._associate_attractors()
        
        if not branch_influences:
            return False
        
        new_branches = self._grow_branches(branch_influences)
        
        self.branches.extend(new_branches)
        
        self._cleanup_attractors()
        
        self.spatial_index.rebuild(self.branches)
        
        self.iteration += 1
        
        return True
    
    def grow(self, callback: Optional[Callable[['Tree', int], None]] = None) -> int:
        """
        Run the full growth loop until completion.
        Optional callback is called after each iteration with (tree, iteration).
        Returns the total number of iterations.
        """
        print(f"Starting growth with {len(self.attractors)} attractors...")
        
        while self.iteration < self.config.max_iterations:
            if not self.grow_step():
                break
            
            if callback:
                callback(self, self.iteration)
            
            if self.iteration % 50 == 0:
                print(f"  Iteration {self.iteration}: {len(self.branches)} branches, "
                      f"{len(self.attractors)} attractors remaining")
        
        print(f"Growth complete after {self.iteration} iterations")
        print(f"  Final branches: {len(self.branches)}")
        print(f"  Remaining attractors: {len(self.attractors)}")
        
        return self.iteration
    
    @property
    def living_attractors(self) -> List[Attractor]:
        return [a for a in self.attractors if a.alive]
    
    @property
    def branch_tips(self) -> List[Branch]:
        return [b for b in self.branches if b.is_tip]
    
    def get_branch_segments(self) -> List[tuple]:
        """Return all branch segments as ((x1,y1), (x2,y2)) tuples for drawing."""
        return [
            (b.start_pos.to_tuple(), b.end_pos.to_tuple())
            for b in self.branches
        ]
