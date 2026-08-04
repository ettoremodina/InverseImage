"""
Standalone script to run Particle Refinement on existing NCA outputs.
Useful for tweaking particle parameters without re-running the whole pipeline.
"""

from config import load_config
from particles import generate_particle_animation
from rendering import load_nca_frames, load_rgb_image

def main():
    pipeline = load_config()

    print(f"Running Particle Refinement for: {pipeline.image_name}")

    # 1. Load NCA Frames
    nca_npz_path = pipeline.render_output_dir / f'{pipeline.image_name}_nca_frames.npz'
    if not nca_npz_path.exists():
        print(f"Error: NCA frames not found at {nca_npz_path}")
        print("Please run train_nca.py or render.py first to generate NCA frames.")
        return

    print(f"Loading NCA frames from {nca_npz_path}...")
    final_nca_frame = load_nca_frames(str(nca_npz_path))["frames"][-1]
    print(f"Loaded final frame with shape {final_nca_frame.shape}")

    # 2. Load Target Image (transparency flattened onto the render background)
    print(f"Loading target image: {pipeline.target_image}")
    target_img = load_rgb_image(
        pipeline.target_image,
        background=pipeline.sca_render.background_color[:3]
    )

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
        stretch_factor=pipeline.particles.particle_stretch_factor,
        radius=pipeline.particles.particle_radius,
        device=pipeline.device,
        outline_width=pipeline.particles.particle_outline_width,
        outline_color=pipeline.particles.particle_outline_color
    )

if __name__ == '__main__':
    main()
