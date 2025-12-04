"""
Simple 2D Vector class for Space Colonization Algorithm.
"""

import numpy as np
from typing import Union


class Vector2D:
    __slots__ = ('x', 'y')
    
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = float(x)
        self.y = float(y)
    
    def __add__(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Vector2D':
        return Vector2D(self.x * scalar, self.y * scalar)
    
    def __rmul__(self, scalar: float) -> 'Vector2D':
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: float) -> 'Vector2D':
        return Vector2D(self.x / scalar, self.y / scalar)
    
    def __repr__(self) -> str:
        return f"Vector2D({self.x:.2f}, {self.y:.2f})"
    
    def __eq__(self, other: 'Vector2D') -> bool:
        return np.isclose(self.x, other.x) and np.isclose(self.y, other.y)
    
    @property
    def magnitude(self) -> float:
        return np.sqrt(self.x ** 2 + self.y ** 2)
    
    @property
    def magnitude_squared(self) -> float:
        return self.x ** 2 + self.y ** 2
    
    def normalize(self) -> 'Vector2D':
        mag = self.magnitude
        if mag < 1e-10:
            return Vector2D(0, 0)
        return self / mag
    
    def distance_to(self, other: 'Vector2D') -> float:
        return (self - other).magnitude
    
    def distance_squared_to(self, other: 'Vector2D') -> float:
        return (self - other).magnitude_squared
    
    def to_tuple(self) -> tuple:
        return (self.x, self.y)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])
    
    @classmethod
    def from_tuple(cls, t: tuple) -> 'Vector2D':
        return cls(t[0], t[1])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Vector2D':
        return cls(arr[0], arr[1])
    
    def copy(self) -> 'Vector2D':
        return Vector2D(self.x, self.y)
