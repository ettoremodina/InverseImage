"""
Data exporters to convert simulation data into renderer-friendly format.
Keeps rendering module decoupled from simulation code.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any


def export_sca_data(tree, output_path: str):
    """
    Export SCA tree to JSON format for rendering.
    
    Format:
    {
        "source_width": int,
        "source_height": int,
        "branches": [
            {
                "start": [x, y],
                "end": [x, y],
                "depth": int,  # distance from root
                "is_tip": bool
            }
        ]
    }
    """
    def get_depth(branch) -> int:
        depth = 0
        current = branch
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth
    
    branches_data = []
    for branch in tree.branches:
        branches_data.append({
            "start": [branch.start_pos.x, branch.start_pos.y],
            "end": [branch.end_pos.x, branch.end_pos.y],
            "depth": get_depth(branch),
            "is_tip": branch.is_tip
        })
    
    data = {
        "source_width": tree.mask_width,
        "source_height": tree.mask_height,
        "branches": branches_data
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return data


def export_nca_frames(frames: List, output_path: str):
    """
    Export NCA animation frames to NPZ format for rendering.
    
    Args:
        frames: List of RGBA frames (torch tensors or numpy arrays)
                Each frame shape: [H, W, 4] with values in [0, 1]
        output_path: Path to save .npz file
    
    Format:
        - frames: np.ndarray of shape [N, H, W, 4], float32
        - source_height: int
        - source_width: int
        - num_frames: int
    """
    processed = []
    for frame in frames:
        if hasattr(frame, 'numpy'):
            frame = frame.numpy()
        frame = np.clip(frame, 0, 1).astype(np.float32)
        processed.append(frame)
    
    frames_array = np.stack(processed, axis=0)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        frames=frames_array,
        source_height=frames_array.shape[1],
        source_width=frames_array.shape[2],
        num_frames=frames_array.shape[0]
    )
    
    print(f"Exported {len(frames)} NCA frames to {output_path}")
    print(f"  Shape: {frames_array.shape}")
    
    return {
        "source_height": frames_array.shape[1],
        "source_width": frames_array.shape[2],
        "num_frames": len(frames)
    }


def load_sca_data(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def load_nca_frames(path: str) -> Dict[str, Any]:
    """
    Load NCA frames from NPZ file.
    
    Returns:
        Dict with keys: frames, source_height, source_width, num_frames
    """
    data = np.load(path)
    return {
        "frames": data["frames"],
        "source_height": int(data["source_height"]),
        "source_width": int(data["source_width"]),
        "num_frames": int(data["num_frames"])
    }
