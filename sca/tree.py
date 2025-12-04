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
from .profiling import profile, profile_block


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
        self._stagnation_counter: int = 0
        
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
    
    @profile
    def _associate_attractors(self) -> Dict[Branch, List[Attractor]]:
        """
        Use KDTree batch query for O(n log m) instead of O(n*m).
        """
        influence_radius = self.config.influence_radius
        kill_distance = self.config.kill_distance
        
        tips = self.spatial_index.tips
        if not tips or self.spatial_index.tree is None:
            return {}
        
        living_attractors = [a for a in self.attractors if a.alive]
        if not living_attractors:
            return {}
        
        attractor_positions = np.array([[a.position.x, a.position.y] for a in living_attractors])
        
        min_distances, closest_tip_indices = self.spatial_index.query_batch(attractor_positions)
        
        kill_mask = min_distances < kill_distance
        influence_mask = (min_distances >= kill_distance) & (min_distances < influence_radius)
        
        for i in np.where(kill_mask)[0]:
            living_attractors[i].kill()
        
        branch_influences: Dict[Branch, List[Attractor]] = {}
        for i in np.where(influence_mask)[0]:
            tip = tips[closest_tip_indices[i]]
            if tip not in branch_influences:
                branch_influences[tip] = []
            branch_influences[tip].append(living_attractors[i])
        
        return branch_influences
    
    @profile
    def _cluster_directions(self, branch: Branch, attractors: List[Attractor]) -> List[Vector2D]:
        """
        Vectorized clustering of attractor directions.
        """
        if not attractors:
            return []
        
        angle_threshold = self.config.branch_angle_threshold
        min_per_branch = self.config.min_attractors_per_branch
        
        branch_pos = np.array([branch.end_pos.x, branch.end_pos.y])
        attractor_positions = np.array([[a.position.x, a.position.y] for a in attractors])
        
        diff = attractor_positions - branch_pos
        norms = np.linalg.norm(diff, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        directions = diff / norms
        
        n = len(directions)
        if n == 1:
            return [Vector2D(directions[0, 0], directions[0, 1])]
        
        cluster_labels = np.full(n, -1, dtype=int)
        cluster_sums = []
        cluster_counts = []
        
        for i in range(n):
            if cluster_labels[i] >= 0:
                continue
            
            cluster_labels[i] = len(cluster_sums)
            cluster_sums.append(directions[i].copy())
            cluster_counts.append(1)
            cluster_idx = cluster_labels[i]
            
            for j in range(i + 1, n):
                if cluster_labels[j] >= 0:
                    continue
                
                cluster_avg = cluster_sums[cluster_idx] / np.linalg.norm(cluster_sums[cluster_idx])
                dot = np.dot(directions[j], cluster_avg)
                
                if dot > angle_threshold:
                    cluster_labels[j] = cluster_idx
                    cluster_sums[cluster_idx] += directions[j]
                    cluster_counts[cluster_idx] += 1
        
        result = []
        for idx, (csum, count) in enumerate(zip(cluster_sums, cluster_counts)):
            if count >= min_per_branch:
                norm = np.linalg.norm(csum)
                if norm > 1e-10:
                    normalized = csum / norm
                    result.append(Vector2D(normalized[0], normalized[1]))
        
        if not result and n > 0:
            total = np.sum(directions, axis=0)
            norm = np.linalg.norm(total)
            if norm > 1e-10:
                normalized = total / norm
                result.append(Vector2D(normalized[0], normalized[1]))
        
        return result
    
    @profile
    def _grow_branches(self, branch_influences: Dict[Branch, List[Attractor]]) -> List[Branch]:
        """Create new branches based on attractor influences, with directional clustering."""
        new_branches = []
        
        for branch, attractors in branch_influences.items():
            if not attractors:
                continue
            
            growth_directions = self._cluster_directions(branch, attractors)
            
            for direction in growth_directions:
                new_end = branch.end_pos + direction * self.config.growth_step
                
                ### note: it could be removed
                if not self._is_inside_mask(new_end):
                    continue
                
                new_branch = branch.create_child(direction, self.config.growth_step)
                new_branches.append(new_branch)
        
        return new_branches
    
    @profile
    def _cleanup_attractors(self):
        """Remove dead attractors from the list."""
        self.attractors = [a for a in self.attractors if a.alive]
    
    @profile
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
        
        if self._stagnation_counter >= self.config.stagnation_limit:
            return False
        
        self._reset_branch_counts()
        
        attractor_count_before = len(self.attractors)
        branch_influences = self._associate_attractors()
        
        if not branch_influences:
            return False
        
        new_branches = self._grow_branches(branch_influences)
        
        self.branches.extend(new_branches)
        
        self._cleanup_attractors()
        
        attractor_count_after = len(self.attractors)
        if attractor_count_after == attractor_count_before:
            self._stagnation_counter += 1
        else:
            self._stagnation_counter = 0
        
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
        
        if self._stagnation_counter >= self.config.stagnation_limit:
            print(f"Growth stopped due to stagnation (no attractors died for {self.config.stagnation_limit} iterations)")
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
