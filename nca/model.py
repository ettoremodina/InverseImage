"""
Neural Cellular Automata Model.
Defines the CAModel architecture for learning cellular automata rules.
"""

import torch
import torch.nn as nn

from .utils import create_filters, perchannel_conv, alive_mask
from config.nca_config import NCAConfig as Config


class CAModel(nn.Module):
    """
    Cellular Automata Model that learns update rules.
    
    Each cell perceives its neighborhood through filters, processes the
    perception through an MLP, and stochastically updates its state.
    """
    
    def __init__(self, config: Config = None):
        super().__init__()
        
        if config is None:
            config = Config()
        
        self.config = config
        self.channel_n = config.channel_n
        self.update_rate = config.update_rate
        self.device = config.device
        
        self.filters = create_filters(self.device)
        
        self.brain = nn.Sequential(
            nn.Conv2d(config.channel_n * 3, config.hidden_size, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(config.hidden_size, config.channel_n, kernel_size=1, bias=False)
        )
        
        with torch.no_grad():
            self.brain[-1].weight.zero_()

    def step(self, x, update_rate=None):
        y = perchannel_conv(x, self.filters)
        y = self.brain(y)

        B, C, H, W = y.shape
        update_rate = update_rate or self.update_rate
        update_mask = (torch.rand(B, 1, H, W).to(self.device) + update_rate).floor()
        x = x + y * update_mask

        mask = alive_mask(x[:, 3:4, :, :], threshold=0.1)
        x = x * mask

        return x

    def forward(self, x, steps=1, update_rate=None):
        for _ in range(steps):
            x = self.step(x, update_rate=update_rate)
        return x
    
    def generate_frames(self, seed, steps, update_rate=None):
        """Generate animation frames from seed."""
        x = seed.clone()
        frames = [torch.clamp(x[0, :4].detach().cpu().permute(1, 2, 0), 0, 1)]
        
        for _ in range(steps):
            x = self.step(x, update_rate=update_rate)
            frames.append(torch.clamp(x[0, :4].detach().cpu().permute(1, 2, 0), 0, 1))
        
        return frames
