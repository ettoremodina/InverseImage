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
    def __init__(self, nca_image, target_image, width, height, num_particles=20000, speed=1.0, trail_fade=0.92, stretch_factor=2.0, radius=2, spawn_duration=0, device='cuda', background_image=None):
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
            radius: Radius of the particle head.
            spawn_duration: Number of frames over which to spawn particles. 0 = all at once.
            background_image: Optional numpy array (H, W, C) to use as initial canvas.
        """
        self.width = width
        self.height = height
        self.num_particles = num_particles
        self.speed = speed
        self.trail_fade = trail_fade
        self.stretch_factor = stretch_factor
        self.radius = radius
        self.spawn_duration = spawn_duration
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

        # 3.1 Prepare Spawn Mask (Alive Cells)
        # We only want to spawn particles where the NCA cells are "alive" (alpha > 0.1)
        if self.field_image.shape[2] == 4:
            # Use alpha channel
            spawn_mask = self.field_image[:, :, 3] > 0.1
        else:
            # Fallback: use luminance > 0.1 if no alpha
            if self.field_image.shape[2] == 3:
                gray = cv2.cvtColor(self.field_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = self.field_image
            spawn_mask = gray > 0.1
            
        # Get valid coordinates for spawning
        valid_y, valid_x = np.where(spawn_mask)
        
        if len(valid_x) == 0:
            print("Warning: No valid spawn points found (no alive cells). Spawning everywhere.")
            self.valid_x = None
            self.valid_y = None
        else:
            self.valid_x = torch.from_numpy(valid_x).to(self.device)
            self.valid_y = torch.from_numpy(valid_y).to(self.device)

        # 4. Initialize Particles
        # Initialize off-screen
        self.px = torch.ones(num_particles, device=self.device) * -1000
        self.py = torch.ones(num_particles, device=self.device) * -1000
        
        # Initialize velocities
        self.vx = torch.zeros(num_particles, device=self.device)
        self.vy = torch.zeros(num_particles, device=self.device)
        
        # Initialize ages for sequential spawning
        # Negative age means "waiting to spawn"
        if self.spawn_duration > 0:
            # Exponential ramp up: start slow, end fast
            # We want the number of spawned particles to grow exponentially with time.
            # x is the cumulative fraction of particles spawned (0 to 1)
            # t is the time (0 to spawn_duration)
            # Relationship: x = (exp(k * t / T) - 1) / (exp(k) - 1)
            # Inverse: t = T * ln(x * (exp(k) - 1) + 1) / k
            
            k = 4.0 # Steepness factor
            x = torch.linspace(0, 1, num_particles, device=self.device)
            
            # Calculate spawn times
            term = x * (np.exp(k) - 1) + 1
            spawn_times = self.spawn_duration * torch.log(term) / k
            
            self.ages = -spawn_times
        else:
            # Spawn all immediately
            self.ages = torch.zeros(num_particles, device=self.device)
            # Set initial positions immediately
            new_x, new_y = self._get_random_spawn_positions(num_particles)
            self.px = new_x
            self.py = new_y
        
        # 5. Initialize Canvas (Background)
        if background_image is not None:
            # Use provided background image
            bg = cv2.resize(background_image, (width, height), interpolation=cv2.INTER_AREA)
            if bg.dtype == np.uint8:
                bg = bg.astype(np.float32) / 255.0
            
            # Ensure channels match target_image (self.channels)
            bg_channels = bg.shape[2] if len(bg.shape) > 2 else 1
            
            if self.channels == 3 and bg_channels == 4:
                # Blend with white if background has alpha
                alpha = bg[:, :, 3:4]
                rgb = bg[:, :, :3]
                white_bg = np.ones_like(rgb)
                self.canvas = alpha * rgb + (1 - alpha) * white_bg
            elif self.channels == 3 and bg_channels == 1:
                self.canvas = cv2.cvtColor(bg, cv2.COLOR_GRAY2RGB)
            elif self.channels == 1 and bg_channels > 1:
                if bg_channels == 4:
                    bg = cv2.cvtColor(bg, cv2.COLOR_RGBA2RGB)
                self.canvas = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)
            else:
                self.canvas = bg.copy()
        else:
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
        
    def _get_random_spawn_positions(self, count):
        """Get random positions within the valid spawn mask."""
        if self.valid_x is None:
            # Fallback: random everywhere
            x = torch.rand(count, device=self.device) * self.width
            y = torch.rand(count, device=self.device) * self.height
            return x, y
            
        # Sample random indices from valid coordinates
        indices = torch.randint(0, len(self.valid_x), (count,), device=self.device)
        
        # Add small random jitter to avoid grid artifacts
        jitter_x = torch.rand(count, device=self.device) - 0.5
        jitter_y = torch.rand(count, device=self.device) - 0.5
        
        x = self.valid_x[indices].float() + jitter_x
        y = self.valid_y[indices].float() + jitter_y
        
        return x, y

    def get_colors_at_positions(self, x, y):
        """Sample colors from the TARGET image at particle positions."""
        # Clip coordinates
        xi = torch.clamp(x.long(), 0, self.width - 1)
        yi = torch.clamp(y.long(), 0, self.height - 1)
        return self.target_image[yi, xi]

    def step(self):
        """Update particle positions based on flow field."""
        # Increment ages
        self.ages += 1.0
        
        # Identify particles that just became active (crossed 0)
        # We use a small window to catch them exactly once
        just_spawned = (self.ages >= 0) & (self.ages < 1.0)
        num_spawn = just_spawned.sum()
        
        if num_spawn > 0:
            new_x, new_y = self._get_random_spawn_positions(num_spawn)
            self.px[just_spawned] = new_x
            self.py[just_spawned] = new_y
            
        # Only move active particles (age >= 0)
        active = self.ages >= 0
        
        if not active.any():
            return

        # 1. Sample flow at current positions
        # Only for active particles to save compute? 
        # Vectorized is usually faster if we just do all, but masking prevents off-screen artifacts affecting flow
        
        # We'll just clamp everything, but off-screen particles (-1000) will clamp to (0,0)
        # This is fine as long as we don't draw them or let them accumulate velocity weirdly.
        # But better to mask.
        
        xi = torch.clamp(self.px.long(), 0, self.width - 1)
        yi = torch.clamp(self.py.long(), 0, self.height - 1)
        
        # Variable speed based on gradient magnitude
        # Particles move faster in high-gradient areas (strong edges)
        # Base speed + modulation
        current_speed = self.speed * (0.5 + 2.0 * self.speed_map[yi, xi])
        
        dx = self.flow_x[yi, xi] * current_speed
        dy = self.flow_y[yi, xi] * current_speed
        
        # Zero out movement for inactive particles
        dx[~active] = 0
        dy[~active] = 0
        
        # 2. Advect
        self.px += dx
        self.py += dy
        
        # Store velocity for rendering stretch
        self.vx = dx
        self.vy = dy
        
        # 3. Boundary wrapping (optional, or respawn)
        # Here we just clamp, but respawning looks better for flow
        # Only check active particles
        mask = active & ((self.px < 0) | (self.px >= self.width) | (self.py < 0) | (self.py >= self.height))
        
        # Respawn out of bounds particles
        num_respawn = mask.sum()
        if num_respawn > 0:
            new_x, new_y = self._get_random_spawn_positions(num_respawn)
            self.px[mask] = new_x
            self.py[mask] = new_y
        
        # self.ages += 1.0 # Moved to top

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
        
        # Filter out inactive particles
        # We need to expand the active mask to match the flattened samples
        active = self.ages >= 0
        active_exp = active.unsqueeze(1).expand(-1, num_samples).flatten()
        
        xs = xs[active_exp]
        ys = ys[active_exp]
        
        if len(xs) > 0:
            # Clip coordinates
            xi = torch.clamp(xs.long(), 0, self.width - 1)
            yi = torch.clamp(ys.long(), 0, self.height - 1)
            
            # Get colors at these positions
            colors = self.get_colors_at_positions(xs, ys)
            
            # Draw to canvas
            # Note: Multiple particles might land on same pixel. 
            # Last one wins with this method.
            self.canvas[yi, xi] = colors
        
        # Convert to numpy for output and overlay drawing
        frame = (self.canvas * 255).byte().cpu().numpy()
        
        # Draw particle heads (circles) with outline
        # We need px, py on CPU
        # Only draw active heads
        active_cpu = active.cpu().numpy()
        px_cpu = self.px.cpu().numpy().astype(np.int32)[active_cpu]
        py_cpu = self.py.cpu().numpy().astype(np.int32)[active_cpu]
        
        # Get colors for heads
        # We need to filter px/py before calling get_colors to avoid sampling off-screen
        # But get_colors clamps anyway.
        # Let's just use the filtered cpu arrays to get colors
        if len(px_cpu) > 0:
            # We need to pass tensors to get_colors_at_positions
            px_active = self.px[active]
            py_active = self.py[active]
            head_colors = self.get_colors_at_positions(px_active, py_active).cpu().numpy() * 255
            
            # Draw circles
            # Note: This loop might be slow for very large particle counts.
            # For 20k particles, it should be acceptable for offline rendering.
            for i in range(len(px_cpu)):
                x, y = px_cpu[i], py_cpu[i]
                color = tuple(map(int, head_colors[i]))
                
                # Draw outline (black)
                cv2.circle(frame, (x, y), self.radius + 1, (0, 0, 0), -1)
                # Draw inner (color)
                cv2.circle(frame, (x, y), self.radius, color, -1)
            
        return frame

def generate_particle_animation(nca_final_frame, target_image, steps, width, height, output_path, fps=30, num_particles=20000, speed=1.0, trail_fade=0.92, stretch_factor=2.0, radius=2, device='cuda', background_image=None):
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
        
    # Spawn particles over the first 50% of the animation
    spawn_duration = int(steps * 0.5)
    
    refiner = ParticleRefiner(nca_final_frame, target_image, width, height, num_particles, speed, trail_fade, stretch_factor, radius=radius, spawn_duration=spawn_duration, device=device, background_image=background_image)
    
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
