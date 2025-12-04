"""
Main entry point for Space Colonization Algorithm (SCA).

This implements the algorithm by Runions et al. (2007) for creating
open, tree-like branching structures within a mask boundary.
"""

from pathlib import Path

from sca import Tree, SCAConfig, visualize_tree, animate_growth
from sca.visualization import plot_growth_statistics


def main():
    config = SCAConfig()
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_name = Path(config.mask_image_path).stem
    
    if config.animate:
        save_path = str(output_dir / f"{image_name}_growth.gif")
        animate_growth(
            config,
            interval=50,
            show_attractors=config.show_attractors,
            save_path=save_path,
            frame_skip=20  # Only save every 20th frame for speed
        )
    else:
        tree = Tree(config)
        tree.grow()
        
        save_path = str(output_dir / f"{image_name}_tree.png")
        visualize_tree(
            tree,
            show_attractors=config.show_attractors,
            save_path=save_path
        )
        
        stats_path = str(output_dir / f"{image_name}_stats.png")
        plot_growth_statistics(tree, save_path=stats_path)


if __name__ == '__main__':
    main()
