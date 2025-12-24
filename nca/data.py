"""
Data loading and preprocessing for Neural Cellular Automata.
Handles image loading and seed creation.
"""

import numpy as np
from PIL import Image
import torch

from config.nca_config import NCAConfig as Config


def prepare_image(path, size):
    """Load image from path, resize and ensure RGBA format."""
    img = Image.open(path).resize((size, size))
    
    if img.mode == 'RGBA':
        return img
    
    if img.mode == 'RGB':
        img_array = np.array(img)
        alpha = np.ones((size, size), dtype=np.uint8) * 255
        img_rgba = np.dstack((img_array, alpha))
        return Image.fromarray(img_rgba, mode='RGBA')
    
    if img.mode == 'L':
        img_array = np.array(img)
        rgb = np.stack([img_array] * 3, axis=-1)
        alpha = np.ones((size, size), dtype=np.uint8) * 255
        img_rgba = np.dstack((rgb, alpha))
        return Image.fromarray(img_rgba, mode='RGBA')
    
    return img.convert('RGBA')


def load_image(path, config: Config = None):
    """Load image and convert to tensor [B, C, H, W] on device."""
    if config is None:
        config = Config()
    
    img = prepare_image(path, size=config.target_size)
    img = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img).permute(2, 0, 1)[None].to(config.device)
    
    return img_tensor


def create_seed(config: Config = None, positions: list = None):
    """
    Create initial seed tensor.
    
    Args:
        config: NCA configuration
        positions: Optional list of (x, y) tuples for seed positions.
                   If None, uses single center seed.
    """
    if config is None:
        config = Config()
    
    size = config.target_size
    seed = torch.zeros(1, config.channel_n, size, size).to(config.device)
    
    if positions is None:
        seed[:, 3:, size // 2, size // 2] = 1.0
    else:
        for x, y in positions:
            if 0 <= x < size and 0 <= y < size:
                seed[:, 3:, int(y), int(x)] = 1.0
    
    return seed
