"""
Configuration for Particle Advection Refinement.
"""

from dataclasses import dataclass

@dataclass
class ParticleConfig:
    particle_count: int = 10000
    particle_speed: float = 10.0
    particle_duration_seconds: float = 5.0
    particle_trail_fade: float = 1.0
    particle_stretch_factor: float = 2.0    
    particle_radius: int = 1