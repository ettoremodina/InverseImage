"""
Training module for Neural Cellular Automata.
Handles the training loop, optimization, and progressive resolution training.

Progressive Training Strategy:
- Train at low resolution first (fast, learns basic patterns)
- Fine-tune at higher resolutions (refines details)
- Uses gradient accumulation to maintain effective batch size at high resolutions
- Uses mixed precision (FP16) to reduce memory usage by ~50%
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from .config import Config, ResolutionStage
from .model import CAModel
from .data import load_image, create_seed


class Trainer:
    """Handles training of the CAModel at a single resolution."""
    
    def __init__(self, model: CAModel, config: Config = None):
        if config is None:
            config = Config()
        
        self.model = model
        self.config = config
        self.losses = []
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=config.betas
        )
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            config.lr_gamma
        )
        
        self.scaler = GradScaler() if config.use_mixed_precision else None
    
    def train(self, target_path, seed=None, verbose=True):
        target = load_image(target_path, self.config)
        target_batch = target.repeat(self.config.batch_size, 1, 1, 1)
        
        if seed is None:
            seed = create_seed(self.config)
        model_in = seed.repeat(self.config.batch_size, 1, 1, 1)
        
        self.losses = []
        iterator = tqdm(range(self.config.n_epochs)) if verbose else range(self.config.n_epochs)
        
        for epoch in iterator:
            self.optimizer.zero_grad()
            
            if self.config.use_mixed_precision and self.config.device == 'cuda':
                with autocast():
                    result = self.model(model_in, steps=self.config.steps_per_epoch)
                    loss = F.mse_loss(result[:, :4], target_batch)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                result = self.model(model_in, steps=self.config.steps_per_epoch)
                loss = F.mse_loss(result[:, :4], target_batch)
                loss.backward()
                self.optimizer.step()
            
            self.scheduler.step()
            self.losses.append(loss.item())
            
            if verbose and epoch % self.config.log_interval == 0:
                tqdm.write(f"Epoch {epoch}: Loss = {loss.item():.6f}")
        
        return self.losses
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")


class ProgressiveTrainer:
    """
    Trains the model progressively at increasing resolutions.
    
    How it works:
    1. Start training at low resolution (e.g., 40x40)
       - Fast iteration, learns the basic growth patterns
       - Full batch size fits in memory
    
    2. Fine-tune at medium resolution (e.g., 128x128)
       - Smaller batch size to fit in memory
       - Gradient accumulation maintains effective batch size
       - Model already knows patterns, just refines them
    
    3. Final fine-tuning at target resolution (e.g., 256x256)
       - Batch size of 1 with more accumulation steps
       - Mixed precision halves memory usage
       - Learns fine details
    
    Why this works:
    - The NCA "genome" (weights) is resolution-independent
    - Patterns learned at 40x40 transfer to 256x256
    - High-res training only needs to refine, not learn from scratch
    """
    
    def __init__(self, model: CAModel, config: Config):
        self.model = model
        self.config = config
        self.all_losses = {}
        
        self.scaler = GradScaler() if config.use_mixed_precision else None
    
    def _create_optimizer(self, lr: float):
        """Create fresh optimizer (needed when changing resolution)."""
        return optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=self.config.betas
        )
    
    def _load_target_and_seed(self, stage: ResolutionStage):
        """Load target image and create seed at the given resolution."""
        temp_config = Config(
            target_size=stage.size,
            device=self.config.device,
            channel_n=self.config.channel_n
        )
        target = load_image(self.config.image_path, temp_config)
        seed = create_seed(temp_config)
        return target, seed
    
    def train_stage(self, stage: ResolutionStage, lr: float, verbose: bool = True):
        """
        Train at a single resolution stage.
        
        Uses gradient accumulation: instead of batch_size=8 at once,
        we do batch_size=1 eight times and accumulate gradients.
        Same mathematical result, 8x less memory.
        """
        print(f"\n{'='*60}")
        print(f"Training at {stage.size}x{stage.size}")
        print(f"  Batch size: {stage.batch_size}")
        print(f"  Accumulation steps: {stage.accumulation_steps}")
        print(f"  Effective batch size: {stage.effective_batch_size}")
        print(f"  Epochs: {stage.epochs}")
        print(f"  Learning rate: {lr:.6f}")
        print(f"{'='*60}\n")
        
        target, seed = self._load_target_and_seed(stage)
        target_batch = target.repeat(stage.batch_size, 1, 1, 1)
        model_in_template = seed.repeat(stage.batch_size, 1, 1, 1)
        
        optimizer = self._create_optimizer(lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, self.config.lr_gamma)
        
        losses = []
        iterator = tqdm(range(stage.epochs)) if verbose else range(stage.epochs)
        
        for epoch in iterator:
            optimizer.zero_grad()
            accumulated_loss = 0.0
            
            # Gradient accumulation loop
            for acc_step in range(stage.accumulation_steps):
                model_in = model_in_template.clone()
                
                if self.config.use_mixed_precision and self.config.device == 'cuda':
                    with autocast():
                        result = self.model(model_in, steps=self.config.steps_per_epoch)
                        loss = F.mse_loss(result[:, :4], target_batch)
                        loss = loss / stage.accumulation_steps  # Scale for accumulation
                    self.scaler.scale(loss).backward()
                else:
                    result = self.model(model_in, steps=self.config.steps_per_epoch)
                    loss = F.mse_loss(result[:, :4], target_batch)
                    loss = loss / stage.accumulation_steps
                    loss.backward()
                
                accumulated_loss += loss.item()
            
            # Update weights after accumulating all gradients
            if self.config.use_mixed_precision and self.config.device == 'cuda':
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            
            scheduler.step()
            losses.append(accumulated_loss)
            
            if verbose and epoch % self.config.log_interval == 0:
                tqdm.write(f"[{stage.size}x{stage.size}] Epoch {epoch}: Loss = {accumulated_loss:.6f}")
        
        return losses
    
    def train(self, verbose: bool = True):
        """
        Run full progressive training through all resolution stages.
        
        Returns dict mapping resolution -> losses
        """
        print(f"Progressive Training Pipeline")
        print(f"Device: {self.config.device}")
        print(f"Mixed Precision: {self.config.use_mixed_precision}")
        print(f"Stages: {[s.size for s in self.config.progressive_stages]}")
        
        current_lr = self.config.lr
        
        for i, stage in enumerate(self.config.progressive_stages):
            # Reduce learning rate for fine-tuning stages
            if i > 0:
                current_lr = current_lr * 0.5
            
            losses = self.train_stage(stage, lr=current_lr, verbose=verbose)
            self.all_losses[stage.size] = losses
            
            # Clear GPU cache between stages
            if self.config.device == 'cuda':
                torch.cuda.empty_cache()
        
        print(f"\n{'='*60}")
        print("Progressive training complete!")
        print(f"{'='*60}")
        
        return self.all_losses
    
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
