"""
Inference script for Hybrid SCA-NCA.
Loads a trained model and generates all animations (SCA growth, NCA growth, combined).
Saves outputs in both GIF and MP4 formats.
"""

import json
from pathlib import Path

import torch

from sca import SCAConfig
from nca import CAModel, Config as NCAConfig
from nca.data import create_seed

from hybrid import (
    grow_sca_with_frames,
    extract_seed_positions,
    save_combined_animation,
    save_frames_as_gif,
    save_frame_as_image,
    render_seeds_image
)


def load_model(model_path: str, device: str = 'cuda'):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = CAModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config


def main():
    model_path = 'outputs/hybrid/Brini_model.pt'
    metadata_path = Path(model_path).with_suffix('.json').with_name(
        Path(model_path).stem.replace('_model', '_metadata') + '.json'
    )
    
    output_dir = Path('outputs/hybrid')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading metadata from {metadata_path}...")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    image_path = metadata['image_path']
    target_size = metadata['target_size']
    seed_positions = [tuple(p) for p in metadata['seed_positions']]
    sca_params = metadata['sca_config']
    image_name = Path(image_path).stem
    
    print(f"   Image: {image_path}")
    print(f"   Target size: {target_size}")
    print(f"   Seed positions: {len(seed_positions)}")
    
    print(f"\nLoading model from {model_path}...")
    model, nca_config = load_model(model_path)
    print("   Model loaded successfully")
    
    print("\n1. Running SCA and collecting frames...")
    sca_config = SCAConfig(
        mask_image_path=image_path,
        num_attractors=sca_params['num_attractors'],
        max_iterations=sca_params['max_iterations'],
        influence_radius=sca_params['influence_radius'],
        kill_distance=sca_params['kill_distance'],
        growth_step=sca_params['growth_step']
    )
    tree, sca_frames = grow_sca_with_frames(sca_config, target_size, frame_skip=1)
    print(f"   Collected {len(sca_frames)} SCA frames")
    
    save_frame_as_image(sca_frames[-1], str(output_dir / f'{image_name}_sca_final.png'))
    save_frames_as_gif(sca_frames, str(output_dir / f'{image_name}_sca_growth.gif'), fps=20)
    
    seeds_img = render_seeds_image(seed_positions, target_size, sca_frames[-1])
    save_frame_as_image(seeds_img, str(output_dir / f'{image_name}_seed_positions.png'))
    
    print("\n2. Generating NCA frames...")
    seed = create_seed(nca_config, positions=seed_positions)
    with torch.no_grad():
        nca_frames = model.generate_frames(seed, steps=100)
    print(f"   Generated {len(nca_frames)} NCA frames")
    
    save_frames_as_gif(nca_frames, str(output_dir / f'{image_name}_nca_growth.gif'), fps=20)
    
    print("\n3. Creating combined animation...")
    combined_path = str(output_dir / f'{image_name}_sca_nca_combined.gif')
    save_combined_animation(sca_frames, nca_frames, combined_path, fps=20)
    
    print(f"\nInference complete! Outputs saved to {output_dir}")


if __name__ == '__main__':
    main()
