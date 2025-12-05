"""
Rendering Script

Loads trained models/metadata and generates high-quality animations.
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
from sca import SCAConfig, grow_sca_with_frames, extract_seed_positions
from rendering import (
    export_sca_data,
    export_nca_frames,
    save_frames_as_gif,
    save_frame_as_image,
    save_combined_animation,
    render_seeds_image
)


def load_nca_model(model_path: str, device: str = 'cuda'):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = CAModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config


def render_nca(pipeline):
    """Render NCA growth animation from trained model."""
    print(f"Loading NCA model from {pipeline.nca_model_path}...")
    model, config = load_nca_model(str(pipeline.nca_model_path), pipeline.device)

    print(f"Generating {pipeline.animation_steps} frames...")
    seed = create_seed(config)
    with torch.no_grad():
        frames = model.generate_frames(seed, steps=pipeline.animation_steps)

    save_frames_as_gif(frames, str(pipeline.render_nca_gif_path), fps=pipeline.render_fps)
    export_nca_frames(frames, str(pipeline.render_output_dir / f'{pipeline.image_name}_nca_frames.npz'))

    print(f"Saved animation to {pipeline.render_nca_gif_path}")
    return frames


def render_sca(pipeline):
    """Render SCA growth animation."""
    sca_config = SCAConfig.from_pipeline(pipeline)

    print(f"Running SCA at {pipeline.render_size}x{pipeline.render_size}...")
    tree, frames = grow_sca_with_frames(sca_config, pipeline.render_size, frame_skip=1)
    print(f"Generated {len(frames)} frames, {len(tree.branches)} branches")

    save_frames_as_gif(frames, str(pipeline.render_sca_gif_path), fps=pipeline.render_fps)
    save_frame_as_image(frames[-1], str(pipeline.render_output_dir / f'{pipeline.image_name}_sca_final.png'))
    export_sca_data(tree, str(pipeline.render_output_dir / f'{pipeline.image_name}_sca_render.json'))

    print(f"Saved animation to {pipeline.render_sca_gif_path}")
    return tree, frames


def render_combined(pipeline):
    """Render combined SCA→NCA animation."""
    sca_config = SCAConfig.from_pipeline(pipeline)

    print("1. Running SCA and collecting frames...")
    tree, sca_frames = grow_sca_with_frames(sca_config, pipeline.target_size, frame_skip=1)
    print(f"   Generated {len(sca_frames)} SCA frames")

    # save_frame_as_image(sca_frames[-1], str(pipeline.render_output_dir / f'{pipeline.image_name}_sca_final.png'))
    # save_frames_as_gif(sca_frames, str(pipeline.render_sca_gif_path), fps=pipeline.render_fps)

    print("2. Extracting seed positions...")
    seed_positions = extract_seed_positions(tree, pipeline.target_size, 
                                             mode=pipeline.seed_mode, max_seeds=pipeline.max_seeds)
    print(f"   Found {len(seed_positions)} seed positions")

    seeds_img = render_seeds_image(seed_positions, pipeline.target_size, sca_frames[-1])
    save_frame_as_image(seeds_img, str(pipeline.render_output_dir / f'{pipeline.image_name}_seeds.png'))

    print(f"3. Loading NCA model from {pipeline.nca_model_path}...")
    model, nca_config = load_nca_model(str(pipeline.nca_model_path), pipeline.device)

    print(f"4. Generating {pipeline.animation_steps} NCA frames...")
    seed = create_seed(nca_config, positions=seed_positions)
    with torch.no_grad():
        nca_frames = model.generate_frames(seed, steps=pipeline.animation_steps)
    print(f"   Generated {len(nca_frames)} NCA frames")

    save_frames_as_gif(nca_frames, str(pipeline.render_nca_gif_path), fps=pipeline.render_fps)

    print("5. Creating combined animation...")
    save_combined_animation(sca_frames, nca_frames, str(pipeline.render_combined_gif_path), 
                            fps=pipeline.render_fps)

    print("6. Exporting render data...")
    export_sca_data(tree, str(pipeline.render_output_dir / f'{pipeline.image_name}_sca_render.json'))
    export_nca_frames(nca_frames, str(pipeline.render_output_dir / f'{pipeline.image_name}_nca_frames.npz'))

    print(f"\nRendering complete! Outputs saved to {pipeline.render_output_dir}")


def main():
    pipeline = load_config()
    pipeline.create_output_dirs()
    
    print(f"Rendering for: {pipeline.target_image}")
    print(f"Output: {pipeline.render_output_dir}")
    print()
    
    mode = 'combined'
    
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
