"""
Branch class - represents a single segment of the growing tree structure.
"""

from typing import Optional
from .vector import Vector2D


class Branch:
    __slots__ = ('start_pos', 'end_pos', 'parent', 'count', '_direction', '_children')
    
    def __init__(self, start_pos: Vector2D, end_pos: Vector2D, parent: Optional['Branch'] = None):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.parent = parent
        self.count = 0  # Number of attractors influencing this branch tip
        self._direction = (end_pos - start_pos).normalize()
        self._children: list['Branch'] = []
        
        if parent is not None:
            parent._children.append(self)
    
    @property
    def direction(self) -> Vector2D:
        return self._direction
    
    @property
    def children(self) -> list['Branch']:
        return self._children
    
    @property
    def is_tip(self) -> bool:
        return len(self._children) == 0
    
    @property
    def length(self) -> float:
        return self.start_pos.distance_to(self.end_pos)
    
    def reset_count(self):
        self.count = 0
    
    def create_child(self, direction: Vector2D, growth_step: float) -> 'Branch':
        new_end = self.end_pos + direction * growth_step
        return Branch(self.end_pos.copy(), new_end, parent=self)
    
    def __repr__(self) -> str:
        return f"Branch({self.start_pos} -> {self.end_pos})"
