"""
Attractor class - represents growth hormone sources that guide branch development.
"""

from .vector import Vector2D


class Attractor:
    __slots__ = ('position', 'alive')
    
    def __init__(self, position: Vector2D):
        self.position = position
        self.alive = True
    
    def kill(self):
        self.alive = False
    
    def __repr__(self) -> str:
        status = "alive" if self.alive else "dead"
        return f"Attractor({self.position}, {status})"
