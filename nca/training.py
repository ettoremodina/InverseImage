"""
Training module for Neural Cellular Automata.
Handles the training loop, optimization, and progressive resolution training.

Progressive Training Strategy:
- Train at low resolution first (fast, learns basic patterns)
- Fine-tune at higher resolutions (refines details)
- Uses gradient accumulation to maintain effective batch size at high resolutions
- Uses mixed precision (FP16) to reduce memory usage by ~50%
"""

from pathlib import Path
from typing import Optional, List, Dict, Union

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from config.nca_config import NCAConfig as Config
from config.common import ResolutionStage
from .model import CAModel
from .data import load_image, create_seed
from .visualization import save_animation


class SamplePool:
    """
    Maintains a pool of patterns for training stability (persistence).
    Prevents catastrophic forgetting by mixing seeds with evolved states.
    """
    def __init__(self, pool_size: int, seed: torch.Tensor, target: torch.Tensor):
        self.pool_size = pool_size
        self.seed = seed
        self.target = target
        self.device = seed.device
        
        # Initialize pool with seeds
        self.pool = seed.repeat(pool_size, 1, 1, 1)
        self.current_idxs = None

    def sample(self, batch_size: int):
        # Select random indices
        self.current_idxs = torch.randperm(self.pool_size)[:batch_size]
        batch = self.pool[self.current_idxs]
        
        # Replace worst sample with seed to prevent forgetting
        with torch.no_grad():
            # Calculate MSE loss for ranking
            target_batch = self.target.repeat(batch_size, 1, 1, 1)
            loss = F.mse_loss(batch[:, :4], target_batch, reduction='none')
            loss = loss.view(batch_size, -1).sum(dim=1)
            
            worst_idx = torch.argmax(loss)
            batch[worst_idx] = self.seed[0]
            
        return batch

    def update(self, new_states: torch.Tensor):
        # Update pool with new states
        with torch.no_grad():
            self.pool[self.current_idxs] = new_states.detach()


class Trainer:
    """
    Unified Trainer for CAModel.
    Handles both single-resolution and progressive training based on configuration.
    """
    
    def __init__(self, model: CAModel, config: Config, output_dir: str = None, seed_positions: list = None):
        self.model = model
        self.config = config
        self.seed_positions = seed_positions
        self.output_dir = Path(output_dir) if output_dir else Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.base_name = Path(config.image_path).stem
        
        self.scaler = GradScaler('cuda') if config.use_mixed_precision else None
        self.all_losses = {}
    
    def _create_optimizer(self, lr: float):
        """Create fresh optimizer."""
        return optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=self.config.betas
        )
    
    def _scale_positions(self, target_size: int):
        """Scale seed positions to target resolution."""
        if not self.seed_positions:
            return None
        
        # Determine base size for scaling
        if self.config.progressive_stages:
            base_size = self.config.progressive_stages[0].size
        else:
            base_size = self.config.target_size
            
        scale = target_size / base_size
        return [(int(x * scale), int(y * scale)) for x, y in self.seed_positions]
    
    def _load_target_and_seed(self, size: int):
        """Load target image and create seed at the given resolution."""
        temp_config = Config(
            target_size=size,
            device=self.config.device,
            channel_n=self.config.channel_n
        )
        target = load_image(self.config.image_path, temp_config)
        scaled_positions = self._scale_positions(size)
        seed = create_seed(temp_config, positions=scaled_positions)
        return target, seed
    
    def _get_steps(self):
        """Get number of steps for current iteration, possibly randomized."""
        if self.config.steps_variance > 0:
            steps = int(torch.normal(float(self.config.steps_per_epoch), self.config.steps_variance, (1,)).item())
            return max(1, steps)
        return self.config.steps_per_epoch
    
    def train_stage(self, stage: ResolutionStage, lr: float, verbose: bool = True):
        """
        Train at a single resolution stage.
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training at {stage.size}x{stage.size}")
            print(f"  Batch size: {stage.batch_size}")
            print(f"  Accumulation steps: {stage.accumulation_steps}")
            print(f"  Effective batch size: {stage.effective_batch_size}")
            print(f"  Epochs: {stage.epochs}")
            print(f"  Learning rate: {lr:.6f}")
            print(f"{'='*60}\n")
        
        target, seed = self._load_target_and_seed(stage.size)
        target_batch = target.repeat(stage.batch_size, 1, 1, 1)
        
        # Initialize Pool if enabled
        pool = None
        if self.config.use_pattern_pool:
            pool = SamplePool(self.config.pool_size, seed, target)
            if verbose:
                print(f"  Initialized pattern pool (size: {self.config.pool_size})")
        
        model_in_template = seed.repeat(stage.batch_size, 1, 1, 1)
        
        optimizer = self._create_optimizer(lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, self.config.lr_gamma)
        
        losses = []
        iterator = tqdm(range(stage.epochs)) if verbose else range(stage.epochs)
        current_epoch = 0
        
        try:
            for epoch in iterator:
                current_epoch = epoch
                optimizer.zero_grad()
                accumulated_loss = 0.0
                
                for _ in range(stage.accumulation_steps):
                    if pool:
                        model_in = pool.sample(stage.batch_size)
                    else:
                        model_in = model_in_template.clone()
                    
                    steps = self._get_steps()
                    
                    if self.config.use_mixed_precision and self.config.device == 'cuda':
                        with autocast('cuda'):
                            result = self.model(model_in, steps=steps)
                            loss = F.mse_loss(result[:, :4], target_batch)
                            loss = loss / stage.accumulation_steps
                        self.scaler.scale(loss).backward()
                    else:
                        result = self.model(model_in, steps=steps)
                        loss = F.mse_loss(result[:, :4], target_batch)
                        loss = loss / stage.accumulation_steps
                        loss.backward()
                    
                    if pool:
                        pool.update(result)
                    
                    accumulated_loss += loss.item()
                
                if self.config.use_mixed_precision and self.config.device == 'cuda':
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                
                scheduler.step()
                losses.append(accumulated_loss)
                
                if verbose and epoch % self.config.log_interval == 0:
                    tqdm.write(f"[{stage.size}x{stage.size}] Epoch {epoch}: Loss = {accumulated_loss:.6f}")
                
                if self.config.checkpoint_interval > 0 and epoch > 0 and epoch % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(stage.size, epoch, is_periodic=True)
        
        except KeyboardInterrupt:
            print(f"\nInterrupted at stage {stage.size}x{stage.size}, epoch {current_epoch}. Saving...")
            self._save_checkpoint(stage.size, current_epoch, interrupted=True)
            raise
        
        return losses

    def train(self, verbose: bool = True):
        """
        Run training pipeline.
        If progressive_stages are defined in config, runs progressive training.
        Otherwise, runs single-stage training based on config parameters.
        """
        print(f"Training Pipeline")
        print(f"Device: {self.config.device}")
        print(f"Mixed Precision: {self.config.use_mixed_precision}")
        
        stages = self.config.progressive_stages
        if not stages:
            # Create a single stage from base config
            stages = [ResolutionStage(
                size=self.config.target_size,
                epochs=self.config.n_epochs,
                batch_size=self.config.batch_size,
                accumulation_steps=1
            )]
            print("Mode: Single Resolution")
        else:
            print(f"Mode: Progressive {[s.size for s in stages]}")
        
        current_lr = self.config.lr
        
        for i, stage in enumerate(stages):
            # Decay LR for subsequent stages in progressive mode
            if i > 0:
                current_lr = current_lr * 0.5
            
            losses = self.train_stage(stage, lr=current_lr, verbose=verbose)
            self.all_losses[stage.size] = losses
            
            # Save checkpoint at end of stage
            self._save_checkpoint(stage.size, stage.epochs, is_periodic=False)
            
            if self.config.device == 'cuda':
                torch.cuda.empty_cache()
        
        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"{'='*60}")
        
        return self.all_losses
    
    def _save_checkpoint(self, size: int, epoch: int, interrupted: bool = False, is_periodic: bool = False):
        """Save model and animation."""
        if is_periodic:
            checkpoint_dir = self.output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = checkpoint_dir / f"{self.base_name}_{size}x{size}_epoch{epoch}_model.pt"
            self.save_model(str(path))
            return

        suffix = f"_interrupted_epoch{epoch}" if interrupted else ""
        model_path = self.output_dir / f"{self.base_name}_{size}x{size}{suffix}_model.pt"
        self.save_model(str(model_path))
        
        # Generate and save animation
        temp_config = Config(
            target_size=size,
            device=self.config.device,
            channel_n=self.config.channel_n
        )
        scaled_positions = self._scale_positions(size)
        seed = create_seed(temp_config, positions=scaled_positions)
        frames = self.model.generate_frames(seed, self.config.animation_steps)
        
        gif_path = self.output_dir / f"{self.base_name}_{size}x{size}{suffix}_animation.gif"
        save_animation(frames, str(gif_path))
        print(f"Checkpoint saved: {model_path.name}, {gif_path.name}")
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'all_losses': self.all_losses
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'all_losses' in checkpoint:
            self.all_losses = checkpoint['all_losses']
        print(f"Model loaded from {path}")
