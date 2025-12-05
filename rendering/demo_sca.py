"""
Demo script for rendering pre-exported SCA data.

This script only loads and renders - no simulation code.
First run main_sca.py to generate the render data.

Run from project root:
    python -m rendering.demo_sca
"""

from pathlib import Path

from rendering.exporters import load_sca_data
from rendering.sca_renderer import SCARenderer
from rendering.config import RenderConfig


def main():
    print("=== SCA Rendering Demo ===\n")
    
    data_path = "outputs/sca/ragnopiccolo_render_data.json"
    
    if not Path(data_path).exists():
        print(f"Error: Data file not found: {data_path}")
        print("Run main_sca.py first to generate render data.")
        return
    
    print(f"Loading data from: {data_path}")
    data = load_sca_data(data_path)
    print(f"  Source resolution: {data['source_width']}x{data['source_height']}")
    print(f"  Branches: {len(data['branches'])}")
    
    max_depth = max(b.get('depth', 0) for b in data['branches'])
    print(f"  Max depth: {max_depth}")
    
    output_dir = Path("outputs/rendering")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    res = 512
    config = RenderConfig(
        output_width=res,
        output_height=res,
        branch_base_width=res / 150,
        branch_tip_width=res / 600,
    )
    renderer = SCARenderer(config)
    
    print("\nRendering growth animation...")
    animation_path = output_dir / "sca_growth.mp4"
    renderer.render_animation(data, str(animation_path), fps=30, frame_skip=1)
    
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
