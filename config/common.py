"""
Common configuration utilities and shared data structures.
"""

from dataclasses import dataclass
import torch

def get_device():
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

@dataclass
class ResolutionStage:
    """Configuration for one resolution stage in progressive training."""
    size: int
    epochs: int
    batch_size: int
    accumulation_steps: int = 1

    @property
    def effective_batch_size(self):
        return self.batch_size * self.accumulation_steps
