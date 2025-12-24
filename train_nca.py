"""
NCA Training Script

Trains a Neural Cellular Automata model to grow a target image.
Supports single-resolution or progressive multi-resolution training.

Configuration is loaded from config/pipeline.json.
All output paths are derived from the target image name.

Outputs:
- Model weights (.pt)
- Training metadata (.json)
- Loss plot (.png)
- Preview animation (.gif)
"""

import json
from pathlib import Path

from config import load_config
from nca import CAModel, create_seed
from nca.training import Trainer
from nca.visualization import save_animation, plot_training_loss
from rendering import export_nca_frames
from config.nca_config import NCAConfig as Config


def train_progressive(nca_config: Config, output_dir: Path, model_path: Path):
    model = CAModel(nca_config).to(nca_config.device)
    trainer = Trainer(model, nca_config, output_dir=str(output_dir),
                                  seed_positions=nca_config.seed_positions)

    all_losses = trainer.train()
    trainer.save_model(str(model_path))

    for size, losses in all_losses.items():
        loss_path = output_dir / f"{model_path.stem}_loss_{size}x{size}.png"
        plot_training_loss(losses, save_path=str(loss_path))

    final_loss = list(all_losses.values())[-1][-1] if all_losses else None
    return model, final_loss


def train_single_resolution(nca_config: Config, model_path: Path, loss_path: Path):
    model = CAModel(nca_config).to(nca_config.device)
    output_dir = model_path.parent
    trainer = Trainer(model, nca_config, output_dir=str(output_dir), seed_positions=nca_config.seed_positions)

    all_losses = trainer.train()
    trainer.save_model(str(model_path))
    
    losses = list(all_losses.values())[0]
    plot_training_loss(losses, save_path=str(loss_path))

    final_loss = losses[-1] if losses else None
    return model, final_loss


def main():
    pipeline = load_config()
    pipeline.create_output_dirs()
    
    nca_config = pipeline.nca
    # Load seed positions if available
    seed_positions = pipeline.load_seed_positions()
    if seed_positions:
        nca_config.seed_positions = seed_positions

    use_progressive = len(nca_config.progressive_stages) > 1

    print(f"Training NCA on {pipeline.target_image}")
    print(f"Mode: {'Progressive' if use_progressive else 'Single resolution'}")
    print(f"Device: {nca_config.device}")
    if nca_config.seed_positions:
        print(f"Using {len(nca_config.seed_positions)} seed positions from SCA")
    print()

    if use_progressive:
        model, final_loss = train_progressive(
            nca_config, pipeline.nca_output_dir, pipeline.nca_model_path
        )
        final_size = nca_config.progressive_stages[-1].size
    else:
        model, final_loss = train_single_resolution(
            nca_config, pipeline.nca_model_path, pipeline.nca_loss_path
        )
        final_size = nca_config.target_size

    metadata = {
        'image_path': pipeline.target_image,
        'target_size': final_size,
        'channel_n': nca_config.channel_n,
        'hidden_size': nca_config.hidden_size,
        'progressive': use_progressive,
        'final_loss': final_loss,
        'model_path': str(pipeline.nca_model_path),
        'seed_positions': nca_config.seed_positions
    }
    with open(pipeline.nca_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {pipeline.nca_metadata_path}")

    print("\nGenerating preview animation...")
    seed = create_seed(nca_config, positions=nca_config.seed_positions)
    frames = model.generate_frames(seed, pipeline.animation_steps)
    save_animation(frames, str(pipeline.nca_animation_path))
    
    # Export frames for particle refinement
    npz_path = str(pipeline.render_output_dir / f'{pipeline.image_name}_nca_frames.npz')
    export_nca_frames(frames, npz_path)
    print(f"Saved NCA frames to {npz_path}")

    print("\nTraining complete!")
    print(f"  Model: {pipeline.nca_model_path}")
    print(f"  Metadata: {pipeline.nca_metadata_path}")
    print(f"  Animation: {pipeline.nca_animation_path}")


if __name__ == '__main__':
    main()
