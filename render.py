"""
Rendering Script

Generates high-quality animations using Cairo-based renderers.

Configuration is loaded from config/pipeline.json.
All paths are derived from the target image name.

Modes:
    timeline  - The pipeline: SCA + NCA + evolutionary swarm on one overlapped
                timeline (PLAN 3.1-3.6). Default, and the only full render.
    still     - One graded frame at a given time, for calibration
    nca       - Render NCA growth animation from trained model (single seed)
    sca       - Render SCA growth animation from metadata

The old sequential path -- SCA -> NCA -> particle advection, rendered as
separate clips and concatenated -- is gone. Stage 3 is the evolutionary swarm
(docs/Evolutionary_Swarm.md) and it runs inside the timeline, so there is no
longer a second video to merge or a frozen background to hide the cut.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

# Patch for backward compatibility with pickled models
import config.nca_config
sys.modules['nca.config'] = config.nca_config

import imageio
import numpy as np
import torch

from color.grading import Grader
from config import load_config
from nca import CAModel, Config as NCAConfig
from nca.data import create_seed
from nca.seeding import SeedSchedule
from rendering import (
    export_nca_frames,
    load_sca_data,
    load_nca_frames,
    SCARenderer,
    NCARenderer,
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


def nca_frames_path(pipeline) -> Path:
    return pipeline.render_output_dir / f'{pipeline.image_name}_nca_frames.npz'


def _seeding_signature(pipeline, steps: int) -> dict:
    """What the cached frames depend on, so a stale cache can be recognised."""
    return {'steps': steps, 'seeding': asdict(pipeline.seeding)}


def load_reusable_nca_frames(pipeline, steps: int):
    """
    The frames on disk, but only if they were made the way we would make them now.

    `--reuse-frames` used to hand back whatever npz happened to be there. That
    is how a render silently came out with all 1000 seeds alive at frame 0: the
    cache predated progressive seeding, and nothing compared the two. The
    signature is written next to the npz and checked here, so a changed seeding
    config invalidates the cache instead of quietly overriding it.
    """
    npz_path = nca_frames_path(pipeline)
    if not npz_path.exists():
        return None

    sidecar = npz_path.with_suffix('.meta.json')
    if not sidecar.exists():
        logger.warning('NCA frames at %s have no seeding signature (written by an older '
                       'version); regenerating rather than trusting them', npz_path)
        return None

    with open(sidecar) as handle:
        stored = json.load(handle)

    wanted = _seeding_signature(pipeline, steps)
    if stored != wanted:
        logger.info('NCA frames on disk were made with a different seeding config or '
                    'step count; regenerating')
        return None

    logger.info('Reusing NCA frames from %s', npz_path)
    return load_nca_frames(str(npz_path))


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

    npz_path = nca_frames_path(pipeline)
    export_nca_frames(frames, str(npz_path))

    with open(npz_path.with_suffix('.meta.json'), 'w') as handle:
        json.dump(_seeding_signature(pipeline, steps), handle, indent=2)

    return load_nca_frames(str(npz_path))


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


def render_timeline(pipeline, reuse_frames: bool = False, swarm: bool = True):
    """
    The single overlapped timeline (PLAN 3.1-3.6).

    Replaces the sequential 'combined' mode: one loop of frames, every stage
    evaluated against its own window, no cut and no frozen background.

    Stage 3 plugs in through the `stages` argument: the swarm repaints the NCA
    tissue inside its own window, using the parameters the lab calibrated
    (`SwarmStageConfig.preset`). Pass `swarm=False` to render the first two
    stages alone, which is also the fallback if stage 3 is disabled in config.
    """
    require_sca_artifacts(pipeline)

    logger.info('Loading SCA data from %s...', pipeline.sca_render_data_path)
    sca_data = load_sca_data(str(pipeline.sca_render_data_path))

    with open(pipeline.sca_seeds_path, 'r') as f:
        seed_positions = json.load(f)['positions']

    steps = max(pipeline.animation_steps,
                int(pipeline.timing.nca.duration * pipeline.render_fps))
    nca_data = load_reusable_nca_frames(pipeline, steps) if reuse_frames else None
    if nca_data is None:
        nca_data = generate_nca_frames(pipeline, sca_data, seed_positions, steps)

    output_path = str(pipeline.render_output_dir / f'{pipeline.image_name}_timeline.mp4')
    remove_if_exists(output_path)

    stages = {}
    swarm_stage = None
    if swarm:
        from swarm.stage import build_swarm_stage
        swarm_stage = build_swarm_stage(pipeline)
        if swarm_stage is not None:
            stages['swarm'] = swarm_stage

    build_timeline(pipeline).render(sca_data, nca_data, output_path, stages=stages)

    if swarm_stage is not None:
        swarm_stage.report()

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

    steps = max(pipeline.animation_steps,
                int(pipeline.timing.nca.duration * pipeline.render_fps))
    nca_data = load_reusable_nca_frames(pipeline, steps) if reuse_frames else None
    if nca_data is None:
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


def main():
    parser = argparse.ArgumentParser(description="Render animations for the NCA-SCA pipeline.")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['timeline', 'still', 'nca', 'sca'],
        default='timeline',
        help='Rendering mode (default: timeline). "nca" and "sca" render one '
             'stage on its own, for inspection.'
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
    parser.add_argument(
        '--no-swarm', action='store_true',
        help='Render the timeline without stage 3, for comparing against it.'
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
        render_timeline(pipeline, reuse_frames=args.reuse_frames, swarm=not args.no_swarm)
    elif args.mode == 'still':
        time = args.time if args.time is not None else pipeline.timing.nca.end
        render_still(pipeline, time)
    elif args.mode == 'nca':
        render_nca(pipeline)
    elif args.mode == 'sca':
        render_sca(pipeline)


if __name__ == '__main__':
    main()
