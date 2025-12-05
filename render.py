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
from sca import SCAConfig, grow_sca_with_frames, extract_seed_positions
from rendering import (
    export_sca_data,
    export_nca_frames,
    load_sca_data,
    load_nca_frames,
    SCARenderer,
    NCARenderer,
    NCARenderConfig,
    RenderConfig,
    save_frame_as_image,
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
    if pipeline.sca_render_data_path.exists():
        print(f"Loading SCA data from {pipeline.sca_render_data_path}...")
        sca_data = load_sca_data(str(pipeline.sca_render_data_path))
    else:
        print("No SCA render data found, running SCA first...")
        sca_config = SCAConfig.from_pipeline(pipeline)
        tree, _ = grow_sca_with_frames(sca_config, pipeline.render_size, frame_skip=1)
        export_sca_data(tree, str(pipeline.sca_render_data_path))
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
    sca_config = SCAConfig.from_pipeline(pipeline)

    print("1. Running SCA and collecting frames...")
    tree, sca_frames = grow_sca_with_frames(sca_config, pipeline.target_size, frame_skip=1)
    print(f"   Generated {len(sca_frames)} SCA frames")

    sca_json_path = str(pipeline.render_output_dir / f'{pipeline.image_name}_sca_render.json')
    export_sca_data(tree, sca_json_path)

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

    nca_npz_path = str(pipeline.render_output_dir / f'{pipeline.image_name}_nca_frames.npz')
    export_nca_frames(nca_frames, nca_npz_path)

    print("5. Rendering SCA animation with Cairo...")
    sca_render_config = RenderConfig(
        output_width=pipeline.render_size,
        output_height=pipeline.render_size
    )
    sca_renderer = SCARenderer(sca_render_config)
    sca_data = load_sca_data(sca_json_path)
    sca_output = str(pipeline.render_sca_gif_path.with_suffix('.mp4'))
    sca_renderer.render_animation(sca_data, sca_output, fps=pipeline.render_fps)

    print("6. Rendering NCA animation with Cairo...")
    nca_render_config = NCARenderConfig(
        output_width=pipeline.render_size,
        output_height=pipeline.render_size,
        cell_shape="circle",
        cell_scale=1.0
    )
    nca_renderer = NCARenderer(nca_render_config)
    nca_data = load_nca_frames(nca_npz_path)
    nca_output = str(pipeline.render_nca_gif_path.with_suffix('.mp4'))
    nca_renderer.render_animation(nca_data, nca_output, fps=pipeline.render_fps)

    print(f"\nRendering complete! Outputs saved to {pipeline.render_output_dir}")
    print(f"  SCA: {sca_output}")
    print(f"  NCA: {nca_output}")


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
