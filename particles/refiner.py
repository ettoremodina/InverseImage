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
    def __init__(self, nca_image, target_image, width, height, num_particles=20000, speed=1.0, trail_fade=0.92, stretch_factor=2.0, device='cuda'):
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
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
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
        
        # Move maps to GPU
        self.flow_x = torch.from_numpy(self.flow_x.astype(np.float32)).to(self.device)
        self.flow_y = torch.from_numpy(self.flow_y.astype(np.float32)).to(self.device)
        self.speed_map = torch.from_numpy(self.speed_map.astype(np.float32)).to(self.device)
        self.target_image = torch.from_numpy(self.target_image.astype(np.float32)).to(self.device)

        # 4. Initialize Particles
        # Random positions
        self.px = torch.rand(num_particles, device=self.device) * width
        self.py = torch.rand(num_particles, device=self.device) * height
        
        # Initialize velocities
        self.vx = torch.zeros(num_particles, device=self.device)
        self.vy = torch.zeros(num_particles, device=self.device)
        
        # Random lifetimes for fading in/out
        self.ages = torch.rand(num_particles, device=self.device) * 100
        
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
            
        self.canvas = torch.from_numpy(self.canvas).to(self.device)
        
    def get_colors_at_positions(self, x, y):
        """Sample colors from the TARGET image at particle positions."""
        # Clip coordinates
        xi = torch.clamp(x.long(), 0, self.width - 1)
        yi = torch.clamp(y.long(), 0, self.height - 1)
        return self.target_image[yi, xi]

    def step(self):
        """Update particle positions based on flow field."""
        # 1. Sample flow at current positions
        xi = torch.clamp(self.px.long(), 0, self.width - 1)
        yi = torch.clamp(self.py.long(), 0, self.height - 1)
        
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
        num_respawn = mask.sum()
        if num_respawn > 0:
            self.px[mask] = torch.rand(num_respawn, device=self.device) * self.width
            self.py[mask] = torch.rand(num_respawn, device=self.device) * self.height
        
        self.ages += 1.0

    def render_frame(self):
        """Render the current state of particles to an image."""
        # We draw onto the persistent canvas
        
        # Apply trail fading
        self.canvas *= self.trail_fade
        
        # Vectorized drawing with stretch
        # Draw multiple points along the velocity vector to simulate a streak
        num_samples = 5
        t = torch.linspace(0, 1, num_samples, device=self.device)
        
        # Expand dims for broadcasting: (N, 1)
        px_exp = self.px.unsqueeze(1)
        py_exp = self.py.unsqueeze(1)
        vx_exp = self.vx.unsqueeze(1)
        vy_exp = self.vy.unsqueeze(1)
        
        # Calculate sample points: (N, num_samples)
        # Draw backwards from current position to create a tail
        xs = px_exp - vx_exp * self.stretch_factor * t
        ys = py_exp - vy_exp * self.stretch_factor * t
        
        # Flatten to list of points
        xs = xs.flatten()
        ys = ys.flatten()
        
        # Clip coordinates
        xi = torch.clamp(xs.long(), 0, self.width - 1)
        yi = torch.clamp(ys.long(), 0, self.height - 1)
        
        # Get colors at these positions
        colors = self.get_colors_at_positions(xs, ys)
        
        # Draw to canvas
        # Note: Multiple particles might land on same pixel. 
        # Last one wins with this method.
        self.canvas[yi, xi] = colors
        
        return (self.canvas * 255).byte().cpu().numpy()

def generate_particle_animation(nca_final_frame, target_image, steps, width, height, output_path, fps=30, num_particles=20000, speed=1.0, trail_fade=0.92, stretch_factor=2.0, device='cuda'):
    """Main driver to generate and save the particle animation."""
    print(f"Initializing Particle Refiner ({width}x{height}) on {device}...")
    
    # Ensure nca_frame is numpy
    if isinstance(nca_final_frame, torch.Tensor):
        nca_final_frame = nca_final_frame.detach().cpu().numpy()
    
    # If batch dim exists, take first
    if len(nca_final_frame.shape) == 4:
        nca_final_frame = nca_final_frame[0]
        
    # Channels last
    if nca_final_frame.shape[0] in [3, 4]:
        nca_final_frame = np.transpose(nca_final_frame, (1, 2, 0))
        
    refiner = ParticleRefiner(nca_final_frame, target_image, width, height, num_particles, speed, trail_fade, stretch_factor, device=device)
    
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
