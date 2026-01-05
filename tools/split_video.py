"""
Script to split the full pipeline video into 3 separate GIFs for the README.
"""

import imageio
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import load_config

def split_video():
    pipeline = load_config()
    
    # 1. Define Paths
    video_path = pipeline.render_output_dir / f'{pipeline.image_name}_full_pipeline.mp4'
    output_dir = Path('docs/assets')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not video_path.exists():
        print(f"Error: Video not found at {video_path}")
        print("Please run 'python render.py --mode combined' first.")
        return

    print(f"Processing video: {video_path}")

    # 2. Calculate Durations (in seconds)
    fps = pipeline.render_fps
    sca_duration = pipeline.total_video_duration_seconds * pipeline.sca_percentage
    nca_duration = pipeline.total_video_duration_seconds * pipeline.nca_percentage
    particle_duration = pipeline.particles.particle_duration_seconds
    
    print(f"Durations calculated from config:")
    print(f"  FPS: {fps}")
    print(f"  SCA: {sca_duration:.2f}s")
    print(f"  NCA: {nca_duration:.2f}s")
    print(f"  Particles: {particle_duration:.2f}s")

    # 3. Read Video
    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()
    video_fps = meta['fps']
    total_frames = reader.count_frames()
    duration = meta['duration']
    
    print(f"Video Metadata:")
    print(f"  FPS: {video_fps}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Duration: {duration:.2f}s")

    # 4. Define Frame Ranges
    # We use the config FPS to calculate frame counts, but we must align with the actual video FPS
    # Ideally they are the same.
    
    sca_end_time = sca_duration
    nca_end_time = sca_duration + nca_duration
    
    # Convert times to frame indices
    sca_end_frame = int(sca_end_time * video_fps)
    nca_end_frame = int(nca_end_time * video_fps)
    
    ranges = [
        ('sca', 0, sca_end_frame),
        ('nca', sca_end_frame, nca_end_frame),
        ('particles', nca_end_frame, total_frames) # Rest of the video
    ]

    # 5. Extract and Save GIFs
    summary_frames = []
    
    for name, start, end in ranges:
        output_path = output_dir / f'{name}.gif'
        print(f"Creating {output_path} (Frames {start} to {end})...")
        
        frames = []
        for i in range(start, end):
            try:
                frame = reader.get_data(i)
                # Optional: Resize or downsample if GIF is too huge
                # frame = frame[::2, ::2] 
                frames.append(frame)
            except IndexError:
                break
        
        if frames:
            # Save as GIF
            # loop=0 means infinite loop
            imageio.mimsave(output_path, frames, fps=video_fps, loop=0)
            print(f"  Saved {len(frames)} frames.")
            summary_frames.append(frames[-1])
        else:
            print("  Warning: No frames extracted for this section.")

    # 6. Save Final Frames (Optional, for static previews)
    if len(summary_frames) == 3:
        print("Saving final frames...")
        imageio.imwrite(output_dir / 'sca_final.png', summary_frames[0])
        imageio.imwrite(output_dir / 'nca_final.png', summary_frames[1])
        imageio.imwrite(output_dir / 'particles_final.png', summary_frames[2])
        
        print("Creating summary triptych...")
        # Concatenate horizontally side-by-side
        triptych = np.concatenate(summary_frames, axis=1)
        triptych_path = output_dir / 'pipeline_triptych.png'
        imageio.imwrite(triptych_path, triptych)
        print(f"Saved triptych to {triptych_path}")

    # 7. Save Target Image
    import shutil
    target_src = Path(pipeline.target_image)
    target_dst = output_dir / 'target.png'
    if target_src.exists():
        shutil.copy(target_src, target_dst)
        print(f"Saved target image to {target_dst}")
    else:
        print(f"Warning: Target image not found at {target_src}")

    reader.close()
    print("Done!")

if __name__ == "__main__":
    split_video()
