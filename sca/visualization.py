"""
Visualization utilities for Space Colonization Algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from typing import List, Optional, Tuple
from pathlib import Path

from .tree import Tree
from .config import SCAConfig


def visualize_tree(
    tree: Tree,
    show_attractors: bool = False,
    show_mask: bool = True,
    branch_color: str = 'saddlebrown',
    branch_width: float = 1.0,
    attractor_color: str = 'green',
    attractor_size: float = 1.0,
    figsize: Tuple[int, int] = (16, 16),
    save_path: Optional[str] = None
):
    """Visualize the current state of the tree."""
    fig, ax = plt.subplots(figsize=figsize)
    
    if show_mask:
        ax.imshow(tree.mask, cmap='gray', alpha=0.2, origin='upper')
    
    segments = [
        [(b.start_pos.x, b.start_pos.y), (b.end_pos.x, b.end_pos.y)]
        for b in tree.branches
    ]
    
    if segments:
        lc = LineCollection(segments, colors=branch_color, linewidths=branch_width)
        ax.add_collection(lc)
    
    if show_attractors and tree.attractors:
        attractor_positions = np.array([
            [a.position.x, a.position.y] 
            for a in tree.attractors if a.alive
        ])
        if len(attractor_positions) > 0:
            ax.scatter(
                attractor_positions[:, 0], 
                attractor_positions[:, 1],
                c=attractor_color, 
                s=attractor_size, 
                alpha=0.5
            )
    
    ax.set_xlim(0, tree.mask_width)
    ax.set_ylim(tree.mask_height, 0)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"Saved visualization to {save_path}")
    
    plt.show()
    return fig, ax


def animate_growth(
    config: SCAConfig,
    interval: int = 50,
    show_attractors: bool = True,
    branch_color: str = 'saddlebrown',
    branch_width: float = 1.0,
    figsize: Tuple[int, int] = (16, 16),
    save_path: Optional[str] = None,
    frame_skip: int = 1
) -> FuncAnimation:
    """
    Create an animation of the tree growth process.
    
    frame_skip: Only record every Nth iteration (default 10). Higher = faster, fewer frames.
    """
    tree = Tree(config)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.imshow(tree.mask, cmap='gray', alpha=0.2, origin='upper')
    ax.set_xlim(0, tree.mask_width)
    ax.set_ylim(tree.mask_height, 0)
    ax.set_aspect('equal')
    ax.axis('off')
    
    branch_collection = LineCollection([], colors=branch_color, linewidths=branch_width)
    ax.add_collection(branch_collection)
    
    if show_attractors:
        attractor_scatter = ax.scatter([], [], c='green', s=1, alpha=0.3)
    
    title = ax.set_title('Iteration: 0')
    
    frames_data = []
    
    def init():
        branch_collection.set_segments([])
        if show_attractors:
            attractor_scatter.set_offsets(np.empty((0, 2)))
        return [branch_collection]
    
    def collect_frame():
        frames_data.append({
            'segments': [
                [(b.start_pos.x, b.start_pos.y), (b.end_pos.x, b.end_pos.y)]
                for b in tree.branches
            ],
            'attractors': np.array([
                [a.position.x, a.position.y]
                for a in tree.attractors if a.alive
            ]) if tree.attractors else np.empty((0, 2)),
            'iteration': tree.iteration
        })
    
    collect_frame()
    
    while tree.grow_step():
        if tree.iteration % frame_skip == 0:
            collect_frame()
        if tree.iteration >= config.max_iterations:
            break
    
    collect_frame()
    
    print(f"Collected {len(frames_data)} frames for animation")
    
    def update(frame_idx):
        data = frames_data[frame_idx]
        branch_collection.set_segments(data['segments'])
        
        if show_attractors:
            if len(data['attractors']) > 0:
                attractor_scatter.set_offsets(data['attractors'])
            else:
                attractor_scatter.set_offsets(np.empty((0, 2)))
        
        title.set_text(f"Iteration: {data['iteration']}")
        return [branch_collection]
    
    anim = FuncAnimation(
        fig, update, 
        frames=len(frames_data),
        init_func=init,
        interval=interval,
        blit=False,
        repeat=True
    )
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving animation ({len(frames_data)} frames)...")
        anim.save(save_path, writer='pillow', fps=20)
        print(f"Saved animation to {save_path}")
    
    plt.show()
    return anim


def plot_growth_statistics(tree: Tree, save_path: Optional[str] = None):
    """Plot statistics about the grown tree."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    branch_lengths = [b.length for b in tree.branches]
    axes[0].hist(branch_lengths, bins=30, color='saddlebrown', edgecolor='black')
    axes[0].set_xlabel('Branch Length')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Branch Length Distribution')
    
    depths = []
    for branch in tree.branches:
        depth = 0
        current = branch
        while current.parent is not None:
            depth += 1
            current = current.parent
        depths.append(depth)
    
    max_depth = max(depths) if depths else 0
    depth_counts = [depths.count(d) for d in range(max_depth + 1)]
    axes[1].bar(range(max_depth + 1), depth_counts, color='forestgreen', edgecolor='black')
    axes[1].set_xlabel('Tree Depth')
    axes[1].set_ylabel('Branch Count')
    axes[1].set_title('Branches per Depth Level')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved statistics to {save_path}")
    
    plt.show()
    return fig, axes
