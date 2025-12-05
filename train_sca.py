"""
SCA Growth Script

Runs the Space Colonization Algorithm to generate branching structures.
Extracts seed positions for NCA and saves all metadata.

Configuration is loaded from config/pipeline.json.
All output paths are derived from the target image name.

Outputs:
- Render data (.json) for high-resolution rendering
- Seed positions (.json) for NCA training
- Final tree visualization (.png)
- Growth statistics (.png)
"""

import json
from pathlib import Path

from config import load_config
from sca import Tree, SCAConfig, visualize_tree, extract_seed_positions
from sca.visualization import plot_growth_statistics
from rendering.exporters import export_sca_data


def main():
    pipeline = load_config()
    pipeline.create_output_dirs()
    
    sca_config = SCAConfig.from_pipeline(pipeline)

    print(f"Running SCA on {pipeline.target_image}")
    print(f"  Attractors: {sca_config.num_attractors}")
    print(f"  Max iterations: {sca_config.max_iterations}")
    print()

    tree = Tree(sca_config)
    tree.grow()

    print(f"Generated {len(tree.branches)} branches in {tree.iteration} iterations")

    visualize_tree(
        tree,
        show_attractors=sca_config.show_attractors,
        save_path=str(pipeline.sca_tree_path)
    )

    # plot_growth_statistics(tree, save_path=str(pipeline.sca_output_dir / f'{pipeline.image_name}_stats.png'))

    export_sca_data(tree, str(pipeline.sca_render_data_path))
    print(f"Exported render data to: {pipeline.sca_render_data_path}")

    seed_positions = extract_seed_positions(
        tree, pipeline.target_size, 
        mode=pipeline.seed_mode, 
        max_seeds=pipeline.max_seeds
    )
    pipeline.save_seed_positions(seed_positions)

    metadata = {
        'image_path': pipeline.target_image,
        'num_attractors': sca_config.num_attractors,
        'max_iterations': sca_config.max_iterations,
        'influence_radius': sca_config.influence_radius,
        'kill_distance': sca_config.kill_distance,
        'growth_step': sca_config.growth_step,
        'num_branches': len(tree.branches),
        'iterations': tree.iteration,
        'num_seeds': len(seed_positions),
        'render_data_path': str(pipeline.sca_render_data_path),
        'seeds_path': str(pipeline.sca_seeds_path)
    }
    with open(pipeline.sca_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {pipeline.sca_metadata_path}")

    print("\nSCA complete!")
    print(f"  Tree: {pipeline.sca_tree_path}")
    print(f"  Seeds: {pipeline.sca_seeds_path}")
    print(f"  Metadata: {pipeline.sca_metadata_path}")
    print(f"\nTo use seeds in NCA training, set in config/pipeline.json:")
    print(f'  "seed_positions_path": "{pipeline.sca_seeds_path}"')


if __name__ == '__main__':
    main()
