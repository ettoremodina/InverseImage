"""
Training script for Hybrid SCA-NCA.
Runs SCA to generate seed positions, trains NCA model, and saves the model.
Use inference_hybrid.py to load the model and generate animations.
"""

import json
from pathlib import Path

from sca import SCAConfig
from nca import CAModel, Config as NCAConfig
from nca.training import Trainer
from nca.visualization import plot_training_loss

from hybrid import (
    grow_sca_with_frames,
    extract_seed_positions,
    save_frame_as_image,
    render_seeds_image
)


def main():
    image_path = 'images/Brini.png'
    output_dir = Path('outputs/hybrid')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    target_size = 128
    image_name = Path(image_path).stem
    
    print("1. Running SCA to generate skeleton...")
    sca_config = SCAConfig(
        mask_image_path=image_path,
        num_attractors=2000,
        max_iterations=800
    )
    tree, sca_frames = grow_sca_with_frames(sca_config, target_size, frame_skip=2)
    print(f"   Generated {len(tree.branches)} branches")
    
    save_frame_as_image(sca_frames[-1], str(output_dir / f'{image_name}_sca_final.png'))
    
    print(f"2. Extracting seed positions from {len(tree.branches)} branches...")
    nca_config = NCAConfig(
        image_path=image_path,
        target_size=target_size,
        n_epochs=2000,
        output_dir=str(output_dir)
    )
    seed_positions = extract_seed_positions(tree, target_size, mode='tips', max_seeds=100)
    print(f"   Found {len(seed_positions)} seed positions")
    
    seeds_img = render_seeds_image(seed_positions, target_size, sca_frames[-1])
    save_frame_as_image(seeds_img, str(output_dir / f'{image_name}_seed_positions.png'))
    
    print("3. Training NCA model with multi-seed...")
    model = CAModel(nca_config).to(nca_config.device)
    trainer = Trainer(model, nca_config, seed_positions=seed_positions)
    losses = trainer.train(image_path)
    
    model_path = str(output_dir / f'{image_name}_model.pt')
    trainer.save_model(model_path)
    
    metadata = {
        'image_path': image_path,
        'target_size': target_size,
        'seed_positions': seed_positions,
        'sca_config': {
            'num_attractors': sca_config.num_attractors,
            'max_iterations': sca_config.max_iterations,
            'influence_radius': sca_config.influence_radius,
            'kill_distance': sca_config.kill_distance,
            'growth_step': sca_config.growth_step
        },
        'num_branches': len(tree.branches),
        'final_loss': losses[-1] if losses else None
    }
    metadata_path = output_dir / f'{image_name}_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Saved metadata to {metadata_path}")
    
    loss_path = str(output_dir / f'{image_name}_loss.png')
    plot_training_loss(losses, save_path=loss_path)
    
    print(f"\nTraining complete!")
    print(f"   Model saved to: {model_path}")
    print(f"   Run inference_hybrid.py to generate animations")


if __name__ == '__main__':
    main()
