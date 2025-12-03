"""
Main entry point for Neural Cellular Automata training.

Supports two training modes:
1. Single resolution: Traditional training at one resolution (config.target_size)
2. Progressive: Train at increasing resolutions (40 -> 128 -> 256)

Configure in nca/config.py:
- Set progressive_stages for multi-resolution training
- Set use_mixed_precision=True to halve memory usage
- Adjust batch_size and accumulation_steps per stage
"""

from pathlib import Path

from nca.config import Config
from nca.model import CAModel
from nca.data import create_seed
from nca.training import Trainer, ProgressiveTrainer
from nca.visualization import (
    display_animation, 
    save_animation, 
    plot_training_loss,
    show_image
)


def get_unique_name(output_dir: Path, base_name: str, extension: str) -> str:
    """Generate a unique filename by appending a number if the file already exists."""
    candidate = output_dir / f"{base_name}{extension}"
    if not candidate.exists():
        return base_name
    
    counter = 1
    while True:
        candidate = output_dir / f"{base_name}_{counter}{extension}"
        if not candidate.exists():
            return f"{base_name}_{counter}"
        counter += 1


def train_progressive(config: Config, output_dir: Path):
    """Train progressively through multiple resolutions."""
    model = CAModel(config).to(config.device)
    trainer = ProgressiveTrainer(model, config, output_dir=str(output_dir))
    
    all_losses = trainer.train()
    
    image_name = Path(config.image_path).stem
    unique_name = get_unique_name(output_dir, f"{image_name}_progressive", "_model.pt")
    
    model_path = output_dir / f"{unique_name}_model.pt"
    trainer.save_model(str(model_path))
    
    for size, losses in all_losses.items():
        loss_path = output_dir / f"{unique_name}_loss_{size}x{size}.png"
        plot_training_loss(losses, save_path=str(loss_path))
    
    return model, all_losses, unique_name


def train_single_resolution(config: Config, output_dir: Path):
    """Train at a single resolution (original behavior)."""
    model = CAModel(config).to(config.device)
    trainer = Trainer(model, config)
    
    losses = trainer.train(config.image_path)
    
    image_name = Path(config.image_path).stem
    unique_name = get_unique_name(output_dir, image_name, "_model.pt")
    
    model_path = output_dir / f"{unique_name}_model.pt"
    trainer.save_model(str(model_path))
    
    loss_path = output_dir / f"{unique_name}_loss.png"
    plot_training_loss(losses, save_path=str(loss_path))
    
    return model, losses, unique_name


def main():
    config = Config()
    # Experiment: single resolution training at 128x128 for 400 epochs
    # config = Config(
    #     target_size=128,
    #     n_epochs=1000,
    #     progressive_stages=[],  # No progressive training
    # )
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Use progressive training if stages are defined
    use_progressive = len(config.progressive_stages) > 1
    
    if use_progressive:
        model, _, unique_name = train_progressive(config, output_dir)
        final_size = config.progressive_stages[-1].size
    else:
        model, _, unique_name = train_single_resolution(config, output_dir)
        final_size = config.target_size
    
    # Generate animation at final resolution
    print("\nGenerating animation...")
    final_config = Config(
        target_size=final_size,
        device=config.device,
        channel_n=config.channel_n
    )
    seed = create_seed(final_config)
    frames = model.generate_frames(seed, config.animation_steps)
    
    if config.save_gif:
        gif_path = output_dir / f"{unique_name}_animation.gif"
        save_animation(frames, str(gif_path))
    else:
        display_animation(frames)
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
