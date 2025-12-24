"""
Particle Advection Refinement Module.

This module implements a particle system that refines low-resolution NCA outputs
by simulating particles flowing along the contours of the image.
"""

import numpy as np
import cv2
from tqdm import tqdm
import torch

class ParticleRefiner:
    """
    Refines a low-res NCA image using particle advection.
    Flow is derived from NCA (structure), Color is derived from Target Image (detail).
    """
    def __init__(self, nca_image, target_image, width, height, num_particles=20000, speed=1.0, trail_fade=0.92, stretch_factor=2.0):
        """
        Args:
            nca_image: Numpy array (H, W, C) float32 [0, 1]. Source of flow field.
            target_image: Numpy array (H, W, C) float32 [0, 1] or uint8 [0, 255]. Source of color.
            width: Target render width.
            height: Target render height.
            num_particles: Number of particles to simulate.
            speed: Movement speed of particles.
            trail_fade: Decay factor for trails (0.0-1.0). Lower = shorter trails.
            stretch_factor: How much to stretch particles based on velocity.
        """
        self.width = width
        self.height = height
        self.num_particles = num_particles
        self.speed = speed
        self.trail_fade = trail_fade
        self.stretch_factor = stretch_factor
        
        # 1. Prepare Flow Source (NCA)
        # Resize to target resolution for smooth gradients
        # Ensure input is float32
        if nca_image.dtype != np.float32:
            nca_image = nca_image.astype(np.float32)
            
        self.field_image = cv2.resize(nca_image, (width, height), interpolation=cv2.INTER_CUBIC)
        self.field_image = np.clip(self.field_image, 0, 1)
        
        # 2. Prepare Color Source (Target)
        # Resize and normalize
        self.target_image = cv2.resize(target_image, (width, height), interpolation=cv2.INTER_AREA)
        if self.target_image.dtype == np.uint8:
            self.target_image = self.target_image.astype(np.float32) / 255.0
            
        # Ensure channel consistency for rendering
        self.channels = self.target_image.shape[2] if len(self.target_image.shape) > 2 else 1

        # 3. Compute Flow Field from NCA Luminance
        # Convert to grayscale to find intensity gradients
        if self.field_image.shape[2] == 4:
            gray = cv2.cvtColor(self.field_image, cv2.COLOR_RGBA2GRAY)
        elif self.field_image.shape[2] == 3:
            gray = cv2.cvtColor(self.field_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = self.field_image
            
        # Compute gradients (Sobel derivatives)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        
        # 3. Create Vector Field
        # We want particles to flow ALONG edges, not across them.
        # The gradient points across the edge (uphill). 
        # The perpendicular vector (-dy, dx) points along the contour.
        self.flow_x = -grad_y
        self.flow_y = grad_x
        
        # Normalize vectors
        magnitude = np.sqrt(self.flow_x**2 + self.flow_y**2)
        
        # Create speed map from magnitude (normalized 0-1)
        # Particles will move faster in high-gradient areas
        self.speed_map = magnitude / (np.max(magnitude) + 1e-6)
        
        # Avoid division by zero
        magnitude[magnitude < 1e-5] = 1e-5 
        self.flow_x /= magnitude
        self.flow_y /= magnitude
        
        # 4. Initialize Particles
        # Random positions
        self.px = np.random.rand(num_particles) * width
        self.py = np.random.rand(num_particles) * height
        
        # Initialize velocities
        self.vx = np.zeros(num_particles)
        self.vy = np.zeros(num_particles)
        
        # Random lifetimes for fading in/out
        self.ages = np.random.rand(num_particles) * 100
        
        # 5. Initialize Canvas (Background)
        # We start with the upscaled NCA image so particles draw ON TOP of it.
        # We need to ensure channel consistency with target_image.
        
        # Check channels of upscaled NCA image (self.field_image)
        nca_channels = self.field_image.shape[2] if len(self.field_image.shape) > 2 else 1
        
        if self.channels == 3 and nca_channels == 4:
            # Convert RGBA to RGB with WHITE background blending
            # Extract alpha channel
            alpha = self.field_image[:, :, 3]
            rgb = self.field_image[:, :, :3]
            
            # Create white background
            white_bg = np.ones_like(rgb)
            
            # Blend: alpha * rgb + (1 - alpha) * white
            # Ensure alpha is broadcastable
            alpha = alpha[:, :, np.newaxis]
            
            self.canvas = alpha * rgb + (1 - alpha) * white_bg
            
        elif self.channels == 1 and nca_channels > 1:
             self.canvas = cv2.cvtColor(self.field_image, cv2.COLOR_RGBA2GRAY)
        elif self.channels == 3 and nca_channels == 1:
             self.canvas = cv2.cvtColor(self.field_image, cv2.COLOR_GRAY2RGB)
        else:
            self.canvas = self.field_image.copy()
            
        # Ensure canvas is float32
        if self.canvas.dtype != np.float32:
            self.canvas = self.canvas.astype(np.float32)
        
    def get_colors_at_positions(self, x, y):
        """Sample colors from the TARGET image at particle positions."""
        # Clip coordinates
        xi = np.clip(x.astype(int), 0, self.width - 1)
        yi = np.clip(y.astype(int), 0, self.height - 1)
        return self.target_image[yi, xi]

    def step(self):
        """Update particle positions based on flow field."""
        # 1. Sample flow at current positions
        xi = np.clip(self.px.astype(int), 0, self.width - 1)
        yi = np.clip(self.py.astype(int), 0, self.height - 1)
        
        # Variable speed based on gradient magnitude
        # Particles move faster in high-gradient areas (strong edges)
        # Base speed + modulation
        current_speed = self.speed * (0.5 + 2.0 * self.speed_map[yi, xi])
        
        dx = self.flow_x[yi, xi] * current_speed
        dy = self.flow_y[yi, xi] * current_speed
        
        # 2. Advect
        self.px += dx
        self.py += dy
        
        # Store velocity for rendering stretch
        self.vx = dx
        self.vy = dy
        
        # 3. Boundary wrapping (optional, or respawn)
        # Here we just clamp, but respawning looks better for flow
        mask = (self.px < 0) | (self.px >= self.width) | (self.py < 0) | (self.py >= self.height)
        
        # Respawn out of bounds particles
        self.px[mask] = np.random.rand(np.sum(mask)) * self.width
        self.py[mask] = np.random.rand(np.sum(mask)) * self.height
        
        self.ages += 1.0

    def render_frame(self):
        """Render the current state of particles to an image."""
        # We draw onto the persistent canvas
        
        # Apply trail fading
        self.canvas *= self.trail_fade
        
        # Vectorized drawing with stretch
        # Draw multiple points along the velocity vector to simulate a streak
        num_samples = 5
        t = np.linspace(0, 1, num_samples)
        
        # Expand dims for broadcasting: (N, 1)
        px_exp = self.px[:, np.newaxis]
        py_exp = self.py[:, np.newaxis]
        vx_exp = self.vx[:, np.newaxis]
        vy_exp = self.vy[:, np.newaxis]
        
        # Calculate sample points: (N, num_samples)
        # Draw backwards from current position to create a tail
        xs = px_exp - vx_exp * self.stretch_factor * t
        ys = py_exp - vy_exp * self.stretch_factor * t
        
        # Flatten to list of points
        xs = xs.flatten()
        ys = ys.flatten()
        
        # Clip coordinates
        xi = np.clip(xs.astype(int), 0, self.width - 1)
        yi = np.clip(ys.astype(int), 0, self.height - 1)
        
        # Get colors at these positions
        colors = self.get_colors_at_positions(xs, ys)
        
        # Draw to canvas
        # Note: Multiple particles might land on same pixel. 
        # Last one wins with this method.
        self.canvas[yi, xi] = colors
        
        return (self.canvas * 255).astype(np.uint8)

def generate_particle_animation(nca_final_frame, target_image, steps, width, height, output_path, fps=30, num_particles=20000, speed=1.0, trail_fade=0.92, stretch_factor=2.0):
    """Main driver to generate and save the particle animation."""
    print(f"Initializing Particle Refiner ({width}x{height})...")
    
    # Ensure nca_frame is numpy
    if isinstance(nca_final_frame, torch.Tensor):
        nca_final_frame = nca_final_frame.detach().cpu().numpy()
    
    # If batch dim exists, take first
    if len(nca_final_frame.shape) == 4:
        nca_final_frame = nca_final_frame[0]
        
    # Channels last
    if nca_final_frame.shape[0] in [3, 4]:
        nca_final_frame = np.transpose(nca_final_frame, (1, 2, 0))
        
    refiner = ParticleRefiner(nca_final_frame, target_image, width, height, num_particles, speed, trail_fade, stretch_factor)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Rendering {steps} particle frames...")
    for _ in tqdm(range(steps)):
        refiner.step()
        frame = refiner.render_frame()
        
        # Convert RGBA to BGR for OpenCV
        if frame.shape[2] == 4:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
        out.write(frame_bgr)
        
    out.release()
    print(f"Particle animation saved to {output_path}")
