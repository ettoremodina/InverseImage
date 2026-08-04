"""
Rendering Script

Generates high-quality animations using Cairo-based renderers.
Supports NCA, SCA, Particles, and combined SCA->NCA->Particles animations.

Configuration is loaded from config/pipeline.json.
All paths are derived from the target image name.

Modes:
    timeline  - Single overlapped timeline: SCA + NCA (+ swarm slot). The one
                that implements PLAN 3.1-3.6. Default.
    still     - One graded frame at a given time, for calibration
    nca       - Render NCA growth animation from trained model (single seed)
    sca       - Render SCA growth animation from metadata
    particles - Render Particle refinement animation (requires NCA output)
    combined  - Legacy sequential pipeline: SCA -> NCA -> Particles, concatenated
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Patch for backward compatibility with pickled models
import config.nca_config
sys.modules['nca.config'] = config.nca_config

import imageio
import imageio_ffmpeg
import numpy as np
import torch

from color.grading import Grader
from config import load_config
from nca import CAModel, Config as NCAConfig
from nca.data import create_seed
from nca.seeding import SeedSchedule
from particles import generate_particle_animation
from rendering import (
    export_nca_frames,
    load_sca_data,
    load_nca_frames,
    SCARenderer,
    NCARenderer,
    CombinedRenderer,
    TimelineRenderer,
    Camera,
    resolve,
    load_rgb_image
)
from utils.log import get_logger

logger = get_logger(__name__)


def make_resolver(pipeline, total_frames: int = 1, animate_camera: bool = True):
    """
    The tail of every frame: camera crop, downscale to video size, grading.

    One function for every stage, which is the point of PLAN g -- the three
    stages stop looking like three different exports because they now leave the
    pipeline through the same door.
    """
    grader = Grader(pipeline.grading)
    camera = Camera(pipeline.camera, pipeline.canvas_size)

    def resolve_frame(frame, index=0):
        progress = index / max(1, total_frames - 1) if animate_camera else 0.0
        cropped = resolve(frame, pipeline.render_size, camera.crop(progress))
        return grader.apply(cropped)[..., :3]

    return resolve_frame


def remove_if_exists(path: str):
    """Remove file if it exists to ensure fresh write."""
    p = Path(path)
    if p.exists():
        try:
            os.remove(p)
            print(f"Removed existing file: {path}")
        except OSError as e:
            print(f"Error removing {path}: {e}")

def load_nca_model(model_path: str, device: str = 'cuda'):
    """Load the trained NCA model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = CAModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config


def require_sca_artifacts(pipeline):
    """Fail early and clearly when train_sca.py has not been run."""
    if not pipeline.sca_render_data_path.exists():
        raise FileNotFoundError(
            f"SCA render data not found at {pipeline.sca_render_data_path}. "
            f"Please run train_sca.py first."
        )
    if not pipeline.sca_seeds_path.exists():
        raise FileNotFoundError(
            f"SCA seed positions not found at {pipeline.sca_seeds_path}. "
            f"Please run train_sca.py first."
        )


def generate_nca_frames(pipeline, sca_data, seed_positions, steps: int):
    """
    Run the NCA, optionally seeding it progressively (PLAN 3.2).

    With `seeding.enabled` the seeds are not all written at t=0: each one is
    injected when the tree has grown down to it, through the hook added to
    `CAModel.generate_frames`. The model is untouched -- this is step 1 of the
    plan, the experiment at fixed weights.
    """
    model, nca_config = load_nca_model(str(pipeline.nca_model_path), pipeline.device)

    if pipeline.seeding.enabled and seed_positions:
        schedule = SeedSchedule.from_sca(
            seed_positions, sca_data, nca_config.target_size, steps, pipeline.seeding
        )
        state = schedule.initial_state(
            nca_config.channel_n, nca_config.target_size, pipeline.device)
        hook = schedule.inject
    else:
        state = create_seed(nca_config, positions=seed_positions)
        hook = None

    logger.info('Generating %d NCA frames...', steps)
    with torch.no_grad():
        frames = model.generate_frames(state, steps=steps, hook=hook)

    npz_path = str(pipeline.render_output_dir / f'{pipeline.image_name}_nca_frames.npz')
    export_nca_frames(frames, npz_path)
    return load_nca_frames(npz_path)


def build_timeline(pipeline):
    return TimelineRenderer(
        sca_config=pipeline.sca_render,
        nca_config=pipeline.nca_render,
        timing=pipeline.timing,
        camera_config=pipeline.camera,
        grading_config=pipeline.grading,
        scaffold_config=pipeline.scaffold,
        output_size=pipeline.render_size,
        fps=pipeline.render_fps,
    )


def render_timeline(pipeline, reuse_frames: bool = False):
    """
    The single overlapped timeline (PLAN 3.1-3.6).

    Replaces the sequential 'combined' mode: one loop of frames, every stage
    evaluated against its own window, no cut and no frozen background.

    Stage 3 is not wired yet -- its window renders as a hold. It plugs in
    through the `stages` argument of TimelineRenderer.render.
    """
    require_sca_artifacts(pipeline)

    logger.info('Loading SCA data from %s...', pipeline.sca_render_data_path)
    sca_data = load_sca_data(str(pipeline.sca_render_data_path))

    with open(pipeline.sca_seeds_path, 'r') as f:
        seed_positions = json.load(f)['positions']

    npz_path = pipeline.render_output_dir / f'{pipeline.image_name}_nca_frames.npz'
    if reuse_frames and npz_path.exists():
        logger.info('Reusing NCA frames from %s', npz_path)
        nca_data = load_nca_frames(str(npz_path))
    else:
        steps = max(pipeline.animation_steps,
                    int(pipeline.timing.nca.duration * pipeline.render_fps))
        nca_data = generate_nca_frames(pipeline, sca_data, seed_positions, steps)

    output_path = str(pipeline.render_output_dir / f'{pipeline.image_name}_timeline.mp4')
    remove_if_exists(output_path)

    build_timeline(pipeline).render(sca_data, nca_data, output_path)
    logger.info('Saved timeline to %s', output_path)
    return output_path


def render_still(pipeline, time: float, reuse_frames: bool = True):
    """
    One graded frame at `time`, for calibration.

    Tuning cellularity, lighting and grading on a video is a waste of minutes;
    this renders the single frame the parameters are being judged on.
    """
    require_sca_artifacts(pipeline)

    sca_data = load_sca_data(str(pipeline.sca_render_data_path))
    with open(pipeline.sca_seeds_path, 'r') as f:
        seed_positions = json.load(f)['positions']

    npz_path = pipeline.render_output_dir / f'{pipeline.image_name}_nca_frames.npz'
    if reuse_frames and npz_path.exists():
        nca_data = load_nca_frames(str(npz_path))
    else:
        steps = max(pipeline.animation_steps,
                    int(pipeline.timing.nca.duration * pipeline.render_fps))
        nca_data = generate_nca_frames(pipeline, sca_data, seed_positions, steps)

    frame = build_timeline(pipeline).render_still(sca_data, nca_data, time)

    output_path = (pipeline.render_output_dir /
                   f'{pipeline.image_name}_still_{time:.1f}s.png')
    imageio.imwrite(str(output_path), frame)
    logger.info('Saved still frame at t=%.1fs to %s', time, output_path)
    return output_path


def render_nca(pipeline):
    """Render high-quality NCA growth animation using Cairo (Single Seed)."""
    print(f"Loading NCA model from {pipeline.nca_model_path}...")
    model, config = load_nca_model(str(pipeline.nca_model_path), pipeline.device)

    print(f"Generating {pipeline.animation_steps} frames...")
    seed = create_seed(config)
    with torch.no_grad():
        frames = model.generate_frames(seed, steps=pipeline.animation_steps)

    npz_path = str(pipeline.render_output_dir / f'{pipeline.image_name}_nca_frames.npz')
    export_nca_frames(frames, npz_path)

    print(f"Rendering at {pipeline.render_size}x{pipeline.render_size}...")
    renderer = NCARenderer(pipeline.nca_render)

    data = load_nca_frames(npz_path)
    output_path = str(pipeline.render_nca_gif_path.with_suffix('.mp4'))
    remove_if_exists(output_path)

    total_frames = int(round(pipeline.total_video_duration_seconds * pipeline.render_fps))
    renderer.render_animation(
        data, output_path,
        fps=pipeline.render_fps,
        duration_seconds=pipeline.total_video_duration_seconds,
        resolve=make_resolver(pipeline, total_frames)
    )

    print(f"Saved animation to {output_path}")
    return frames


def render_sca(pipeline):
    """Render high-quality SCA growth animation using Cairo."""
    if not pipeline.sca_render_data_path.exists():
        raise FileNotFoundError(
            f"SCA render data not found at {pipeline.sca_render_data_path}. "
            f"Please run train_sca.py first to generate the SCA data."
        )
    
    print(f"Loading SCA data from {pipeline.sca_render_data_path}...")
    sca_data = load_sca_data(str(pipeline.sca_render_data_path))

    print(f"Rendering at {pipeline.render_size}x{pipeline.render_size} with Cairo...")
    renderer = SCARenderer(pipeline.sca_render)

    output_path = str(pipeline.render_sca_gif_path.with_suffix('.mp4'))
    remove_if_exists(output_path)

    total_frames = int(round(pipeline.total_video_duration_seconds * pipeline.render_fps))
    resolver = make_resolver(pipeline, total_frames)
    renderer.render_animation(
        sca_data, output_path,
        fps=pipeline.render_fps,
        duration_seconds=pipeline.total_video_duration_seconds,
        resolve=resolver
    )

    final_frame_path = str(pipeline.render_output_dir / f'{pipeline.image_name}_sca_final.png')
    renderer.save_frame(sca_data, final_frame_path, resolve=resolver)

    print(f"Saved animation to {output_path}")
    return sca_data


def render_particles(pipeline, background_image=None):
    """Render particle refinement animation based on the last NCA frame."""
    nca_npz_path = pipeline.render_output_dir / f'{pipeline.image_name}_nca_frames.npz'
    
    if not nca_npz_path.exists():
        raise FileNotFoundError(
            f"NCA frames not found at {nca_npz_path}. "
            f"Please run NCA generation (nca or combined mode) first."
        )

    print(f"Loading NCA frames from {nca_npz_path}...")
    nca_data = load_nca_frames(str(nca_npz_path))
    
    # nca_data is expected to be the frames array
    final_nca_frame = nca_data['frames'][-1]

    # Load Target Image for color sampling, flattening transparency onto the
    # same background the tree is rendered on.
    print(f"Loading target image for coloring: {pipeline.target_image}")
    target_img = load_rgb_image(
        pipeline.target_image,
        background=pipeline.sca_render.background_color[:3]
    )

    particle_output_path = str(pipeline.render_output_dir / f'{pipeline.image_name}_particles.mp4')
    remove_if_exists(particle_output_path)
    
    particle_steps = int(pipeline.particles.particle_duration_seconds * pipeline.render_fps)
    print(f"Generating {particle_steps} particle frames ({pipeline.particles.particle_duration_seconds}s at {pipeline.render_fps} fps)")
    
    generate_particle_animation(
        nca_final_frame=final_nca_frame,
        target_image=target_img,
        steps=particle_steps,
        width=pipeline.render_size,
        height=pipeline.render_size,
        output_path=particle_output_path,
        fps=pipeline.render_fps,
        num_particles=pipeline.particles.particle_count,
        speed=pipeline.particles.particle_speed,
        trail_fade=pipeline.particles.particle_trail_fade,
        stretch_factor=pipeline.particles.particle_stretch_factor,
        radius=pipeline.particles.particle_radius,
        device=pipeline.device,
        background_image=background_image,
        outline_width=pipeline.particles.particle_outline_width,
        outline_color=pipeline.particles.particle_outline_color
    )
    
    return particle_output_path


def merge_videos(video_paths: list, output_path: str):
    """
    Concatenate videos without re-encoding.

    Every clip in the pipeline is written by the same libx264 settings
    (rendering/utils.py), so ffmpeg's concat demuxer can copy the streams
    straight through: no generation loss, no bitrate blow-up, and it takes a
    fraction of a second instead of decoding and re-encoding every frame.
    """
    print(f"Merging {len(video_paths)} videos into {output_path}...")

    remove_if_exists(output_path)

    listing = Path(output_path).with_suffix('.concat.txt')
    listing.write_text(
        ''.join(f"file '{Path(p).resolve().as_posix()}'\n" for p in video_paths),
        encoding='utf-8'
    )

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'concat', '-safe', '0', '-i', str(listing),
        '-c', 'copy', str(output_path),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Full video saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to merge videos: {e.stderr.strip()}")
    finally:
        listing.unlink(missing_ok=True)


def render_combined(pipeline):
    """Render combined SCA->NCA->Particles animation."""
    # 1. Check prerequisites
    if not pipeline.sca_render_data_path.exists():
        raise FileNotFoundError(
            f"SCA render data not found at {pipeline.sca_render_data_path}. "
            f"Please run train_sca.py first."
        )
    
    if not pipeline.sca_seeds_path.exists():
        raise FileNotFoundError(
            f"SCA seed positions not found at {pipeline.sca_seeds_path}. "
            f"Please run train_sca.py first."
        )

    # 2. Load Data
    print("1. Loading SCA data...")
    sca_data = load_sca_data(str(pipeline.sca_render_data_path))
    
    print("2. Loading seed positions...")
    with open(pipeline.sca_seeds_path, 'r') as f:
        seed_data = json.load(f)
        seed_positions = seed_data['positions']

    print(f"3. Loading NCA model from {pipeline.nca_model_path}...")
    model, nca_config = load_nca_model(str(pipeline.nca_model_path), pipeline.device)

    # 3. Calculate Timing
    total_frames = int(pipeline.total_video_duration_seconds * pipeline.render_fps)
    sca_frames = int(total_frames * pipeline.sca_percentage)
    nca_frames = int(total_frames * pipeline.nca_percentage)
    nca_steps_needed = max(nca_frames, pipeline.animation_steps)
    
    print(f"4. Video configuration:")
    print(f"   Total duration: {pipeline.total_video_duration_seconds}s")
    print(f"   SCA phase: {sca_frames} frames")
    print(f"   NCA phase: {nca_frames} frames")
    
    # 4. Generate NCA Frames (Seeded by SCA)
    print(f"\n5. Generating {nca_steps_needed} NCA frames...")
    seed = create_seed(nca_config, positions=seed_positions)
    with torch.no_grad():
        nca_frames_data = model.generate_frames(seed, steps=nca_steps_needed)

    nca_npz_path = str(pipeline.render_output_dir / f'{pipeline.image_name}_nca_frames.npz')
    export_nca_frames(nca_frames_data, nca_npz_path)

    # 5. Render Combined SCA -> NCA Video
    print("\n6. Rendering combined SCA->NCA animation...")
    combined_renderer = CombinedRenderer(pipeline.sca_render, pipeline.nca_render)
    nca_data = load_nca_frames(nca_npz_path)
    
    combined_output = str(pipeline.render_combined_gif_path.with_suffix('.mp4'))
    remove_if_exists(combined_output)

    resolver = make_resolver(pipeline, sca_frames + nca_frames)
    combined_renderer.render_animation(
        sca_data,
        nca_data,
        combined_output,
        fps=pipeline.render_fps,
        sca_frames=sca_frames,
        nca_frames=nca_frames,
        resolve=resolver
    )

    # 6. Get Final Frame for Particles Background
    print("\n7. Generating background for particles...")
    # Render the very last frame of the combined animation to use as background
    # This ensures the SCA tree and NCA growth persist. It has to go through the
    # same resolver as the video, or the particle clip would be at canvas
    # resolution and the concat would fail.
    final_nca_frame = nca_data['frames'][-1]
    final_combined_frame = resolver(
        combined_renderer.render_frame(
            sca_data,
            nca_frame=final_nca_frame,
            max_depth_limit=None,
            time=(sca_frames + nca_frames) / pipeline.render_fps
        ),
        sca_frames + nca_frames - 1
    )
    
    # 7. Render Particles
    print("\n8. Running Particle Refinement...")
    particle_output = render_particles(pipeline, background_image=final_combined_frame)

    # 8. Merge Videos
    print("\n9. Merging videos...")
    final_video_path = str(pipeline.render_output_dir / f'{pipeline.image_name}_full_pipeline.mp4')
    merge_videos([combined_output, particle_output], final_video_path)


def main():
    parser = argparse.ArgumentParser(description="Render animations for the NCA-SCA pipeline.")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['timeline', 'still', 'nca', 'sca', 'particles', 'combined'],
        default='timeline',
        help='Rendering mode (default: timeline). "combined" is the legacy '
             'sequential pipeline.'
    )
    parser.add_argument(
        '--time', type=float, default=None,
        help='For --mode still: the instant to render, in seconds. '
             'Defaults to the end of the NCA window.'
    )
    parser.add_argument(
        '--reuse-frames', action='store_true',
        help='Reuse the NCA frames already on disk instead of re-running the model.'
    )
    args = parser.parse_args()

    pipeline = load_config()
    pipeline.create_output_dirs()

    logger.info('Rendering for: %s', pipeline.target_image)
    logger.info('Output: %s', pipeline.render_output_dir)
    logger.info('Mode: %s | canvas %dpx -> video %dpx (supersample %dx)',
                args.mode, pipeline.canvas_size, pipeline.render_size,
                pipeline.render_supersample)

    if args.mode == 'timeline':
        render_timeline(pipeline, reuse_frames=args.reuse_frames)
    elif args.mode == 'still':
        time = args.time if args.time is not None else pipeline.timing.nca.end
        render_still(pipeline, time)
    elif args.mode == 'nca':
        render_nca(pipeline)
    elif args.mode == 'sca':
        render_sca(pipeline)
    elif args.mode == 'particles':
        render_particles(pipeline)
    elif args.mode == 'combined':
        render_combined(pipeline)


if __name__ == '__main__':
    main()
