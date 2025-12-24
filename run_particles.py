"""
Standalone script to run Particle Refinement on existing NCA outputs.
Useful for tweaking particle parameters without re-running the whole pipeline.
"""

import cv2
import torch
import numpy as np
from pathlib import Path

from config import load_config
from particles import generate_particle_animation
from rendering import load_nca_frames

def main():
    pipeline = load_config()
    
    print(f"Running Particle Refinement for: {pipeline.image_name}")
    
    # 1. Load NCA Frames
    nca_npz_path = pipeline.render_output_dir / f'{pipeline.image_name}_nca_frames.npz'
    if not nca_npz_path.exists():
        print(f"Error: NCA frames not found at {nca_npz_path}")
        print("Please run render.py first to generate NCA frames.")
        return

    print(f"Loading NCA frames from {nca_npz_path}...")
    nca_data = load_nca_frames(str(nca_npz_path))
    
    # Extract final frame (assuming nca_data is list of frames or similar structure)
    # The load_nca_frames usually returns a list of dicts or objects for the renderer
    # We might need to load the raw tensor data if we want the raw values
    # Let's check how export_nca_frames saves it. It uses np.savez_compressed.
    
    # Re-loading raw numpy array for precision
    with np.load(nca_npz_path) as data:
        # Assuming 'arr_0' or similar key, or we can just use the loaded nca_data if it preserves float values
        # Let's look at export_nca_frames implementation if needed, but usually it's 'frames'
        if 'frames' in data:
            raw_frames = data['frames']
        else:
            # Fallback to first key
            raw_frames = data[data.files[0]]
            
    final_nca_frame = raw_frames[-1]
    print(f"Loaded final frame with shape {final_nca_frame.shape}")

    # 2. Load Target Image
    target_image_path = str(pipeline.target_image)
    print(f"Loading target image: {target_image_path}")
    target_img = cv2.imread(target_image_path)
    if target_img is None:
        print(f"Error: Target image not found at {target_image_path}")
        return
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

    # 3. Run Particles
    output_path = str(pipeline.render_output_dir / f'{pipeline.image_name}_particles_standalone.mp4')
    
    particle_steps = int(pipeline.particles.particle_duration_seconds * pipeline.render_fps)
    print(f"Generating {particle_steps} particle frames ({pipeline.particles.particle_duration_seconds}s at {pipeline.render_fps} fps)")
    
    generate_particle_animation(
        nca_final_frame=final_nca_frame,
        target_image=target_img,
        steps=particle_steps,
        width=pipeline.render_size,
        height=pipeline.render_size,
        output_path=output_path,
        fps=pipeline.render_fps,
        num_particles=pipeline.particles.particle_count,
        speed=pipeline.particles.particle_speed,
        trail_fade=pipeline.particles.particle_trail_fade,
        stretch_factor=pipeline.particles.particle_stretch_factor
    )

if __name__ == '__main__':
    main()
