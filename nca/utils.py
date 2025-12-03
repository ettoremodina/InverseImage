"""
Utility functions for Neural Cellular Automata.
Contains filters and helper operations.
"""

import torch
import torch.nn.functional as F


def create_filters(device):
    """Create perception filters: identity, vertical sobel, horizontal sobel."""
    return torch.stack([
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]),
        torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).T
    ]).to(device)


def perchannel_conv(x, filters):
    """Apply filters per-channel with circular padding."""
    b, c, h, w = x.shape
    y = x.reshape(b * c, 1, h, w)
    y = F.pad(y, (1, 1, 1, 1), mode='circular')
    y = F.conv2d(y, filters[:, None])
    return y.reshape(b, -1, h, w)


def alive_mask(x, threshold=0.1):
    """Compute alive mask based on alpha channel neighbors."""
    x = F.pad(x, (1, 1, 1, 1), mode='circular')
    return F.max_pool2d(x, 3, stride=1, padding=0) > threshold
