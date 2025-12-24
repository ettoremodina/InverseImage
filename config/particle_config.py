"""
Configuration for Particle Advection Refinement.
"""

from dataclasses import dataclass

@dataclass
class ParticleConfig:
    particle_count: int = 2000
    particle_speed: float = 4.0
    particle_duration_seconds: float = 5.0
    particle_trail_fade: float = 0.92
    particle_stretch_factor: float = 2.0
