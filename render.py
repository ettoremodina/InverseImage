"""
Rendering Script

Generates high-quality animations using Cairo-based renderers.
Supports NCA, SCA, Particles, and combined SCA→NCA→Particles animations.

Configuration is loaded from config/pipeline.json.
All paths are derived from the target image name.

Modes:
    nca       - Render NCA growth animation from trained model (single seed)
    sca       - Render SCA growth animation from metadata
    particles - Render Particle refinement animation (requires NCA output)
    combined  - Render full pipeline: SCA -> NCA (seeded by SCA) -> Particles
"""

import argparse
import json
import sys
from pathlib import Path

# Patch for backward compatibility with pickled models
import config.nca_config
sys.modules['nca.config'] = config.nca_config

import cv2
import numpy as np
import torch

from config import load_config
from nca import CAModel, Config as NCAConfig
from nca.data import create_seed
from particles import generate_particle_animation
from rendering import (
    export_nca_frames,
    load_sca_data,
    load_nca_frames,
    SCARenderer,
    NCARenderer,
    NCARenderConfig,
    SCARenderConfig,
    CombinedRenderer
)


import os

def remove_if_exists(path: str):
    """Remove file if it exists to ensure fresh write."""
    p = Path(path)
    if p.exists():
        try:
            os.remove(p)
            print(f"Removed existing file: {path}")
        except OSError as e:
            print(f"Error removing {path}: {e}")

def load_nca_model(model_path: str, device: str = 'cuda'):
    """Load the trained NCA model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = CAModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config


def render_nca(pipeline):
    """Render high-quality NCA growth animation using Cairo (Single Seed)."""
    print(f"Loading NCA model from {pipeline.nca_model_path}...")
    model, config = load_nca_model(str(pipeline.nca_model_path), pipeline.device)

    print(f"Generating {pipeline.animation_steps} frames...")
    seed = create_seed(config)
    with torch.no_grad():
        frames = model.generate_frames(seed, steps=pipeline.animation_steps)

    npz_path = str(pipeline.render_output_dir / f'{pipeline.image_name}_nca_frames.npz')
    export_nca_frames(frames, npz_path)

    print(f"Rendering at {pipeline.render_size}x{pipeline.render_size} with Cairo...")
    nca_render_config = NCARenderConfig(
        output_width=pipeline.render_size,
        output_height=pipeline.render_size,
        cell_shape="circle",
        cell_scale=1.0
    )
    renderer = NCARenderer(nca_render_config)
    
    data = load_nca_frames(npz_path)
    output_path = str(pipeline.render_nca_gif_path.with_suffix('.mp4'))
    remove_if_exists(output_path)
    renderer.render_animation(data, output_path, fps=pipeline.render_fps)

    print(f"Saved animation to {output_path}")
    return frames


def render_sca(pipeline):
    """Render high-quality SCA growth animation using Cairo."""
    if not pipeline.sca_render_data_path.exists():
        raise FileNotFoundError(
            f"SCA render data not found at {pipeline.sca_render_data_path}. "
            f"Please run train_sca.py first to generate the SCA data."
        )
    
    print(f"Loading SCA data from {pipeline.sca_render_data_path}...")
    sca_data = load_sca_data(str(pipeline.sca_render_data_path))

    print(f"Rendering at {pipeline.render_size}x{pipeline.render_size} with Cairo...")
    sca_render_config = SCARenderConfig(
        output_width=pipeline.render_size,
        output_height=pipeline.render_size
    )
    renderer = SCARenderer(sca_render_config)
    
    output_path = str(pipeline.render_sca_gif_path.with_suffix('.mp4'))
    remove_if_exists(output_path)
    renderer.render_animation(sca_data, output_path, fps=pipeline.render_fps)
    
    final_frame_path = str(pipeline.render_output_dir / f'{pipeline.image_name}_sca_final.png')
    renderer.save_frame(sca_data, final_frame_path)

    print(f"Saved animation to {output_path}")
    return sca_data


def render_particles(pipeline):
    """Render particle refinement animation based on the last NCA frame."""
    nca_npz_path = pipeline.render_output_dir / f'{pipeline.image_name}_nca_frames.npz'
    
    if not nca_npz_path.exists():
        raise FileNotFoundError(
            f"NCA frames not found at {nca_npz_path}. "
            f"Please run NCA generation (nca or combined mode) first."
        )

    print(f"Loading NCA frames from {nca_npz_path}...")
    nca_data = load_nca_frames(str(nca_npz_path))
    
    # nca_data is expected to be the frames array
    final_nca_frame = nca_data['frames'][-1]

    # Load Target Image for color sampling
    target_image_path = str(pipeline.target_image)
    print(f"Loading target image for coloring: {target_image_path}")
    target_img = cv2.imread(target_image_path)
    if target_img is None:
        raise FileNotFoundError(f"Target image not found at {target_image_path}")

    # Convert BGR to RGB
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    
    particle_output_path = str(pipeline.render_output_dir / f'{pipeline.image_name}_particles.mp4')
    remove_if_exists(particle_output_path)
    
    particle_steps = int(pipeline.particles.particle_duration_seconds * pipeline.render_fps)
    print(f"Generating {particle_steps} particle frames ({pipeline.particles.particle_duration_seconds}s at {pipeline.render_fps} fps)")
    
    generate_particle_animation(
        nca_final_frame=final_nca_frame,
        target_image=target_img,
        steps=particle_steps,
        width=pipeline.render_size,
        height=pipeline.render_size,
        output_path=particle_output_path,
        fps=pipeline.render_fps,
        num_particles=pipeline.particles.particle_count,
        speed=pipeline.particles.particle_speed,
        device=pipeline.device
    )
    
    return particle_output_path


def merge_videos(pipeline, video_paths: list, output_path: str):
    """Merge multiple video files into one."""
    print(f"Merging {len(video_paths)} videos into {output_path}...")
    
    try:
        caps = [cv2.VideoCapture(p) for p in video_paths]
        
        if not all(cap.isOpened() for cap in caps):
            print("Error opening one or more video files for merging.")
            return

        # Get properties from first video
        width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = caps[0].get(cv2.CAP_PROP_FPS)
        
        remove_if_exists(output_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for i, cap in enumerate(caps):
            print(f"  Appending video {i+1}/{len(caps)}...")
            while True:
                ret, frame = cap.read()
                if not ret: break
                out.write(frame)
            cap.release()
            
        out.release()
        print(f"Full video saved to {output_path}")
            
    except Exception as e:
        print(f"Failed to merge videos: {e}")


def render_combined(pipeline):
    """Render combined SCA→NCA→Particles animation."""
    # 1. Check prerequisites
    if not pipeline.sca_render_data_path.exists():
        raise FileNotFoundError(
            f"SCA render data not found at {pipeline.sca_render_data_path}. "
            f"Please run train_sca.py first."
        )
    
    if not pipeline.sca_seeds_path.exists():
        raise FileNotFoundError(
            f"SCA seed positions not found at {pipeline.sca_seeds_path}. "
            f"Please run train_sca.py first."
        )

    # 2. Load Data
    print("1. Loading SCA data...")
    sca_data = load_sca_data(str(pipeline.sca_render_data_path))
    
    print("2. Loading seed positions...")
    with open(pipeline.sca_seeds_path, 'r') as f:
        seed_data = json.load(f)
        seed_positions = seed_data['positions']

    print(f"3. Loading NCA model from {pipeline.nca_model_path}...")
    model, nca_config = load_nca_model(str(pipeline.nca_model_path), pipeline.device)

    # 3. Calculate Timing
    total_frames = int(pipeline.total_video_duration_seconds * pipeline.render_fps)
    sca_frames = int(total_frames * pipeline.sca_percentage)
    nca_frames = int(total_frames * pipeline.nca_percentage)
    nca_steps_needed = max(nca_frames, pipeline.animation_steps)
    
    print(f"4. Video configuration:")
    print(f"   Total duration: {pipeline.total_video_duration_seconds}s")
    print(f"   SCA phase: {sca_frames} frames")
    print(f"   NCA phase: {nca_frames} frames")
    
    # 4. Generate NCA Frames (Seeded by SCA)
    print(f"\n5. Generating {nca_steps_needed} NCA frames...")
    seed = create_seed(nca_config, positions=seed_positions)
    with torch.no_grad():
        nca_frames_data = model.generate_frames(seed, steps=nca_steps_needed)

    nca_npz_path = str(pipeline.render_output_dir / f'{pipeline.image_name}_nca_frames.npz')
    export_nca_frames(nca_frames_data, nca_npz_path)

    # 5. Render Combined SCA -> NCA Video
    print("\n6. Rendering combined SCA→NCA animation...")
    sca_render_config = SCARenderConfig(
        output_width=pipeline.render_size,
        output_height=pipeline.render_size
    )
    nca_render_config = NCARenderConfig(
        output_width=pipeline.render_size,
        output_height=pipeline.render_size,
        cell_shape="square",
        cell_scale=1.0
    )
    
    combined_renderer = CombinedRenderer(sca_render_config, nca_render_config)
    nca_data = load_nca_frames(nca_npz_path)
    
    combined_output = str(pipeline.render_combined_gif_path.with_suffix('.mp4'))
    remove_if_exists(combined_output)
    combined_renderer.render_animation(
        sca_data, 
        nca_data, 
        combined_output, 
        fps=pipeline.render_fps,
        sca_frames=sca_frames,
        nca_frames=nca_frames
    )

    # 6. Render Particles
    print("\n7. Running Particle Refinement...")
    particle_output = render_particles(pipeline)

    # 7. Merge Videos
    print("\n8. Merging videos...")
    final_video_path = str(pipeline.render_output_dir / f'{pipeline.image_name}_full_pipeline.mp4')
    merge_videos(pipeline, [combined_output, particle_output], final_video_path)


def main():
    parser = argparse.ArgumentParser(description="Render animations for the NCA-SCA pipeline.")
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['nca', 'sca', 'particles', 'combined'], 
        default='combined',
        help='Rendering mode: nca, sca, particles, or combined (default: combined)'
    )
    args = parser.parse_args()

    pipeline = load_config()
    pipeline.create_output_dirs()
    
    print(f"Rendering for: {pipeline.target_image}")
    print(f"Output: {pipeline.render_output_dir}")
    print(f"Mode: {args.mode}")
    print()

    if args.mode == 'nca':
        render_nca(pipeline)
    elif args.mode == 'sca':
        render_sca(pipeline)
    elif args.mode == 'particles':
        render_particles(pipeline)
    elif args.mode == 'combined':
        render_combined(pipeline)


if __name__ == '__main__':
    main()
