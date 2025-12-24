"""
Rendering Script

Generates high-quality animations using Cairo-based renderers.
Supports NCA, SCA, and combined SCA→NCA animations.

Configuration is loaded from config/pipeline.json.
All paths are derived from the target image name.

Modes:
    nca      - Render NCA growth animation from trained model
    sca      - Render SCA growth animation from metadata  
    combined - Render SCA→NCA combined animation
"""

import json
from pathlib import Path

import torch
import numpy as np

from config import load_config
from nca import CAModel, Config as NCAConfig
from nca.data import create_seed
from rendering import (
    export_nca_frames,
    load_sca_data,
    load_nca_frames,
    SCARenderer,
    NCARenderer,
    NCARenderConfig,
    RenderConfig,
    CombinedRenderer
)


def load_nca_model(model_path: str, device: str = 'cuda'):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = CAModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config


def render_nca(pipeline):
    """Render high-quality NCA growth animation using Cairo."""
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
    sca_render_config = RenderConfig(
        output_width=pipeline.render_size,
        output_height=pipeline.render_size
    )
    renderer = SCARenderer(sca_render_config)
    
    output_path = str(pipeline.render_sca_gif_path.with_suffix('.mp4'))
    renderer.render_animation(sca_data, output_path, fps=pipeline.render_fps)
    
    final_frame_path = str(pipeline.render_output_dir / f'{pipeline.image_name}_sca_final.png')
    renderer.save_frame(sca_data, final_frame_path)

    print(f"Saved animation to {output_path}")
    return sca_data


def render_combined(pipeline):
    """Render combined SCA→NCA animation using Cairo renderers."""
    if not pipeline.sca_render_data_path.exists():
        raise FileNotFoundError(
            f"SCA render data not found at {pipeline.sca_render_data_path}. "
            f"Please run train_sca.py first to generate the SCA data."
        )
    
    if not pipeline.sca_seeds_path.exists():
        raise FileNotFoundError(
            f"SCA seed positions not found at {pipeline.sca_seeds_path}. "
            f"Please run train_sca.py first to generate the seed positions."
        )

    print("1. Loading SCA data...")
    sca_data = load_sca_data(str(pipeline.sca_render_data_path))
    print(f"   Loaded {len(sca_data['branches'])} branches")

    print("2. Loading seed positions...")
    with open(pipeline.sca_seeds_path, 'r') as f:
        seed_data = json.load(f)
        seed_positions = seed_data['positions']
    print(f"   Loaded {len(seed_positions)} seed positions")

    print(f"3. Loading NCA model from {pipeline.nca_model_path}...")
    model, nca_config = load_nca_model(str(pipeline.nca_model_path), pipeline.device)

    # Calculate frame counts based on configured timing
    total_frames = int(pipeline.total_video_duration_seconds * pipeline.render_fps)
    sca_frames = int(total_frames * pipeline.sca_percentage)
    nca_frames = int(total_frames * pipeline.nca_percentage)
    
    # Ensure we generate enough NCA frames
    nca_steps_needed = max(nca_frames, pipeline.animation_steps)
    
    print(f"4. Video configuration:")
    print(f"   Total duration: {pipeline.total_video_duration_seconds}s at {pipeline.render_fps} fps")
    print(f"   Total frames: {total_frames}")
    print(f"   SCA phase: {sca_frames} frames ({pipeline.sca_percentage * 100:.1f}%)")
    print(f"   NCA phase: {nca_frames} frames ({pipeline.nca_percentage * 100:.1f}%)")
    
    print(f"\n5. Generating {nca_steps_needed} NCA frames...")
    seed = create_seed(nca_config, positions=seed_positions)
    with torch.no_grad():
        nca_frames_data = model.generate_frames(seed, steps=nca_steps_needed)
    print(f"   Generated {len(nca_frames_data)} NCA frames")

    nca_npz_path = str(pipeline.render_output_dir / f'{pipeline.image_name}_nca_frames.npz')
    export_nca_frames(nca_frames_data, nca_npz_path)

    print("\n6. Rendering combined SCA→NCA animation with Cairo...")
    sca_render_config = RenderConfig(
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
    combined_renderer.render_animation(
        sca_data, 
        nca_data, 
        combined_output, 
        fps=pipeline.render_fps,
        sca_frames=sca_frames,
        nca_frames=nca_frames
    )

    print(f"\nRendering complete! Output saved to {combined_output}")


def main():
    pipeline = load_config()
    pipeline.create_output_dirs()
    
    print(f"Rendering for: {pipeline.target_image}")
    print(f"Output: {pipeline.render_output_dir}")
    print()
    
    mode = 'sca'
    
    if pipeline.nca_model_path.exists() and pipeline.sca_metadata_path.exists():
        mode = 'combined'
    elif pipeline.nca_model_path.exists():
        mode = 'nca'
    else:
        mode = 'sca'
    
    print(f"Mode: {mode}")
    print()

    if mode == 'nca':
        render_nca(pipeline)
    elif mode == 'sca':
        render_sca(pipeline)
    else:
        render_combined(pipeline)


if __name__ == '__main__':
    main()
