"""
Visualization utilities for hybrid SCA-NCA.
"""

import numpy as np
import torch
from PIL import Image


def _frame_to_rgb(frame, bg_color=(255, 255, 255)):
    """Convert a frame (numpy or tensor) to RGB PIL Image with background."""
    if isinstance(frame, torch.Tensor):
        frame = frame.numpy()
    
    img = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img)
    
    rgb_img = Image.new('RGB', pil_img.size, bg_color)
    if pil_img.mode == 'RGBA':
        rgb_img.paste(pil_img, mask=pil_img.split()[3])
    else:
        rgb_img.paste(pil_img)
    
    return rgb_img


def save_combined_animation(sca_frames: list, nca_frames: list, save_path: str, fps: int = 20):
    """
    Save combined animation: SCA frames first, then NCA frames.
    """
    print(f"   Processing {len(sca_frames)} SCA frames...")
    all_frames = [_frame_to_rgb(f) for f in sca_frames]
    
    print(f"   Processing {len(nca_frames)} NCA frames...")
    all_frames.extend([_frame_to_rgb(f) for f in nca_frames])
    
    print(f"   Saving {len(all_frames)} total frames...")
    all_frames[0].save(
        save_path,
        save_all=True,
        append_images=all_frames[1:],
        duration=1000 // fps,
        loop=0
    )
    print(f"Saved combined animation ({len(sca_frames)} SCA + {len(nca_frames)} NCA frames) to {save_path}")


def save_frames_as_gif(frames: list, save_path: str, fps: int = 20):
    """Save a list of frames as GIF."""
    pil_frames = [_frame_to_rgb(f) for f in frames]
    
    pil_frames[0].save(
        save_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=1000 // fps,
        loop=0
    )
    print(f"Saved animation ({len(pil_frames)} frames) to {save_path}")


def save_frame_as_image(frame, save_path: str):
    """Save a single frame as PNG."""
    if isinstance(frame, torch.Tensor):
        frame = frame.numpy()
    img = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(save_path)
    print(f"Saved image to {save_path}")


def render_seeds_image(seed_positions: list, target_size: int, sca_frame: np.ndarray = None):
    """Render seed positions as an image, optionally overlaid on SCA skeleton."""
    if sca_frame is not None:
        img = sca_frame.copy()
    else:
        img = np.zeros((target_size, target_size, 4), dtype=np.float32)
    
    for x, y in seed_positions:
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                px, py = x + dx, y + dy
                if 0 <= px < target_size and 0 <= py < target_size:
                    img[py, px] = [0.0, 1.0, 0.0, 1.0]
    
    return img
