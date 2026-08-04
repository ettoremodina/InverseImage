"""
Configuration for Particle Advection Refinement.
"""

from dataclasses import dataclass

@dataclass
class ParticleConfig:
    particle_count: int = 2000
    particle_speed: float = 10.0
    particle_duration_seconds: float = 10.0
    particle_trail_fade: float = 1.0
    particle_stretch_factor: float = 2.0
    particle_radius: int = 1

    # Extra radius drawn behind each particle head. Keep at 0 unless the heads
    # need separating from a busy background: the outline area grows with the
    # square of the radius, so even 1 px covers 4x more pixels than a
    # radius-1 head and turns the final frame into a dark mass.
    particle_outline_width: int = 0
    particle_outline_color: tuple = (0.0, 0.0, 0.0, 1.0)