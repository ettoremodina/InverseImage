"""
Demo script for rendering pre-exported NCA data at high resolution.

This script only loads and renders - no simulation code.
First run inference_hybrid.py to generate the render data.

Run from project root:
    python -m rendering.demo_nca
"""

from pathlib import Path

from rendering.exporters import load_nca_frames
from rendering.nca_renderer import NCARenderer, NCARenderConfig


def main():
    print("=== NCA Rendering Demo ===\n")
    
    data_path = "outputs/hybrid/ragnopiccolo_nca_render.npz"
    
    if not Path(data_path).exists():
        print(f"Error: Data file not found: {data_path}")
        print("Run inference_hybrid.py first to generate render data.")
        return
    
    print(f"Loading data from: {data_path}")
    data = load_nca_frames(data_path)
    print(f"  Source resolution: {data['source_width']}x{data['source_height']}")
    print(f"  Frames: {data['num_frames']}")
    
    output_dir = Path("outputs/rendering")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = NCARenderConfig(
        output_width=512,
        output_height=512,
        cell_shape="circle",
        cell_scale=1.0,
        alpha_threshold=0.1,
    )
    renderer = NCARenderer(config)
    
    print("\nRendering growth animation...")
    animation_path = output_dir / "nca_growth.mp4"
    renderer.render_animation(data, str(animation_path), fps=30)
    
    print("\nSaving final frame as PNG...")
    final_frame = data["frames"][-1]
    renderer.save_frame(
        final_frame, 
        data["source_width"], 
        data["source_height"],
        str(output_dir / "nca_final.png")
    )
    
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
