"""
Main entry point for Hybrid SCA-NCA.
Uses SCA tree branches as seed positions for NCA growth.
"""

from pathlib import Path

from sca import SCAConfig
from nca import CAModel, Config as NCAConfig
from nca.data import create_seed
from nca.training import Trainer
from nca.visualization import plot_training_loss

from hybrid import (
    grow_sca_with_frames,
    extract_seed_positions,
    save_combined_animation,
    save_frames_as_gif,
    save_frame_as_image,
    render_seeds_image
)


def main():
    image_path = 'images/dick.png'
    output_dir = Path('outputs/hybrid')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    target_size = 96
    
    print("1. Running SCA and collecting frames...")
    sca_config = SCAConfig(
        mask_image_path=image_path,
        num_attractors=2000,
        max_iterations=1000
    )
    tree, sca_frames = grow_sca_with_frames(sca_config, target_size, frame_skip=2)
    print(f"   Collected {len(sca_frames)} SCA frames")
    
    save_frame_as_image(sca_frames[-1], str(output_dir / 'sca_final.png'))
    save_frames_as_gif(sca_frames, str(output_dir / 'sca_growth.gif'), fps=20)
    
    print(f"2. Extracting seed positions from {len(tree.branches)} branches...")
    nca_config = NCAConfig(
        image_path=image_path,
        target_size=target_size,
        n_epochs=200,
        output_dir=str(output_dir)
    )
    seed_positions = extract_seed_positions(tree, target_size, mode='tips', max_seeds=100)
    print(f"   Found {len(seed_positions)} seed positions")
    
    seeds_img = render_seeds_image(seed_positions, target_size, sca_frames[-1])
    save_frame_as_image(seeds_img, str(output_dir / 'seed_positions.png'))
    
    print("3. Training NCA model with multi-seed...")
    model = CAModel(nca_config).to(nca_config.device)
    trainer = Trainer(model, nca_config, seed_positions=seed_positions)
    losses = trainer.train(image_path)
    
    model_path = str(output_dir / 'hybrid_model.pt')
    trainer.save_model(model_path)
    
    loss_path = str(output_dir / 'hybrid_loss.png')
    plot_training_loss(losses, save_path=loss_path)
    
    print("4. Generating NCA frames...")
    seed = create_seed(nca_config, positions=seed_positions)
    nca_frames = model.generate_frames(seed, steps=100)
    print(f"   Generated {len(nca_frames)} NCA frames")
    
    save_frames_as_gif(nca_frames, str(output_dir / 'nca_growth.gif'), fps=20)
    
    print("5. Creating combined animation...")
    gif_path = str(output_dir / 'sca_nca_combined.gif')
    save_combined_animation(sca_frames, nca_frames, gif_path, fps=20)


if __name__ == '__main__':
    main()
