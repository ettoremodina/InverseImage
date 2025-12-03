"""
Visualization utilities for Neural Cellular Automata.
Provides functions for displaying and saving animations.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import numpy as np


def display_animation(frames, figsize=(6, 6), interval=50):
    """
    Display animation in matplotlib.
    
    Args:
        frames: List of frames, each of shape [H, W, 4]
        figsize: Figure size
        interval: Delay between frames in ms
    """
    fig = plt.figure(figsize=figsize)
    plt.axis('off')
    
    ims = [[plt.imshow(frame, animated=True)] for frame in frames]
    ani = animation.ArtistAnimation(fig, ims, interval=interval, repeat_delay=1000)
    
    plt.show()
    return ani


def save_animation(frames, path, interval=50, fps=20):
    """
    Save animation as GIF.
    
    Args:
        frames: List of frames, each of shape [H, W, 4]
        path: Output path for GIF
        interval: Delay between frames in ms
        fps: Frames per second for the output
    """
    fig = plt.figure(figsize=(6, 6))
    plt.axis('off')
    
    ims = [[plt.imshow(frame, animated=True)] for frame in frames]
    ani = animation.ArtistAnimation(fig, ims, interval=interval, repeat_delay=1000)
    
    ani.save(path, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"Animation saved to {path}")


def save_training_gif(frames, path, duration=50):
    """
    Save frames as GIF using PIL directly.
    
    Args:
        frames: List of numpy arrays or tensors [H, W, 4]
        path: Output path for GIF
        duration: Duration per frame in ms
    """
    pil_frames = []
    
    for frame in frames:
        if hasattr(frame, 'numpy'):
            frame = frame.numpy()
        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        pil_frames.append(Image.fromarray(frame))
    
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0
    )
    print(f"GIF saved to {path}")


def plot_training_loss(losses, save_path=None):
    """Plot training loss curve."""
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Loss plot saved to {save_path}")
    
    plt.show()


def show_image(img_tensor, title=None):
    """Display a single image tensor."""
    if len(img_tensor.shape) == 4:
        img_tensor = img_tensor[0]
    
    img = img_tensor[:4].detach().cpu().permute(1, 2, 0)
    img = img.clamp(0, 1)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()
