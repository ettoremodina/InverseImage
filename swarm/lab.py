"""
The swarm testing suite (Evolutionary_Swarm.md §11).

The swarm is not integrated into the pipeline yet -- there are too many
coupled parameters to tune against a full render. Instead this module runs
the same `Simulation` engine against a single test image, with a *surrogate*
for the NCA output (§11.1: downsample the target to `nca_size` and upsample
it back -- exactly the missing-high-frequency defect a real NCA leaves), so
tuning can start before stage 2 exists.

Three modes, one engine:

    python -m swarm.lab --mode live      # interactive, trackbars, no restart on change
    python -m swarm.lab --mode sweep --param deposit_alpha --values 0.05,0.1,0.15,0.2
    python -m swarm.lab --mode compare --config-a a.json --config-b b.json
"""

import argparse
import copy
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, List

import cv2
import numpy as np
import torch
from tqdm import tqdm

from config.swarm_config import SwarmConfig
from swarm.agents import population_density
from swarm.colorspace import oklab_to_srgb
from swarm.simulation import Simulation
from utils.log import get_logger

logger = get_logger(__name__)

OUT_DIR = 'outputs/swarm/lab'
VIEW_NAMES = ['canvas', 'error', 'pheromone', 'population', 'genes']


# ==================== §11.1 the surrogate ====================

def load_target(image_path: str, work_size: int):
    """RGB float32 [0,1] at `work_size`, plus its alpha (nutrient source)."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(image_path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (work_size, work_size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    if img.shape[2] == 4:
        return img[..., :3], img[..., 3]
    return img, np.ones(img.shape[:2], dtype=np.float32)


def build_surrogate(target_rgb: np.ndarray, work_size: int, nca_size: int = 128,
                     color_noise: float = 0.0, posterize_levels: int = 0, seed: int = 0) -> np.ndarray:
    """
    The surrogate for the NCA output (§11.1). Downsampling to `nca_size` and
    upsampling back reproduces the actual defect -- missing high frequency --
    without needing a trained NCA. `color_noise` and `posterize_levels`
    optionally corrupt colour and gradation too, so a calibration can be
    checked against a starting point that is *wrong*, not just *soft*.
    """
    small = cv2.resize(target_rgb, (nca_size, nca_size), interpolation=cv2.INTER_AREA)
    base = cv2.resize(small, (work_size, work_size), interpolation=cv2.INTER_LINEAR)

    if posterize_levels > 0:
        levels = max(2, posterize_levels)
        base = np.round(base * (levels - 1)) / (levels - 1)

    if color_noise > 0:
        rng = np.random.default_rng(seed)
        base = base + rng.normal(0.0, color_noise, base.shape).astype(np.float32)

    return np.clip(base, 0.0, 1.0)


def make_simulation(config: SwarmConfig, image_path: str, nca_size: int = 128,
                     color_noise: float = 0.0, posterize_levels: int = 0) -> Simulation:
    """Build a `Simulation` from one test image, using the §11.1 surrogate."""
    target_rgb, alpha = load_target(image_path, config.work_size)
    base_rgb = build_surrogate(target_rgb, config.work_size, nca_size, color_noise, posterize_levels, config.seed)
    nutrient = cv2.GaussianBlur(alpha, (5, 5), 0)

    target_t = torch.from_numpy(target_rgb.copy())
    base_t = torch.from_numpy(base_rgb.copy())
    nutrient_t = torch.from_numpy(nutrient.copy())
    return Simulation.from_srgb(config, target_t, base_t, nutrient_t)


# ==================== views ====================

def _colorize(field: np.ndarray, colormap=cv2.COLORMAP_INFERNO) -> np.ndarray:
    lo, hi = float(field.min()), float(field.max())
    norm = (field - lo) / max(hi - lo, 1e-6)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), colormap)


def render_view(sim: Simulation, view: int) -> np.ndarray:
    """BGR uint8 frame for one of the five views in §11.2."""
    name = VIEW_NAMES[view]

    if name == 'canvas':
        return cv2.cvtColor(sim.render_canvas_srgb().cpu().numpy(), cv2.COLOR_RGB2BGR)

    if name == 'error':
        return _colorize(sim.fields.compute_error(raw=True).cpu().numpy())

    if name == 'pheromone':
        return _colorize(sim.fields.pheromone.cpu().numpy(), cv2.COLORMAP_VIRIDIS)

    if name == 'population':
        density = population_density(sim.agents, sim.fields.height, sim.fields.width)
        return _colorize(density.cpu().numpy(), cv2.COLORMAP_VIRIDIS)

    # 'genes' -- each alive agent's own colour, dropped at its own position.
    h, w = sim.fields.height, sim.fields.width
    img = np.zeros((h, w, 3), dtype=np.float32)
    alive = sim.agents.alive.cpu().numpy()
    if alive.any():
        pos = sim.agents.pos[sim.agents.alive].cpu()
        srgb = oklab_to_srgb(sim.agents.gene[sim.agents.alive]).cpu().numpy()
        xi = np.clip(pos[:, 0].numpy().astype(int), 0, w - 1)
        yi = np.clip(pos[:, 1].numpy().astype(int), 0, h - 1)
        img[yi, xi] = srgb
    return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


# ==================== §11.2 live ====================

_HOT_PARAMS = [
    # (label, attr, scale, max_scaled_value)
    ('deposit_alpha x1000', 'deposit_alpha', 1000, 1000),
    ('mutation_sigma x1000', 'mutation_sigma', 1000, 300),
    ('cost_life x10000', 'cost_life', 10000, 500),
    ('cost_deposit x1000', 'cost_deposit', 1000, 500),
    ('evaporation x100', 'evaporation', 100, 100),
    ('tolerance x1000', 'tolerance', 1000, 300),
    ('brush_radius x10', 'brush_radius', 10, 50),
    ('reproduction_thr x100', 'reproduction_threshold', 100, 500),
]


def _save_snapshot(sim: Simulation, config: SwarmConfig, out_dir: str = OUT_DIR):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')

    frame = sim.render_canvas_srgb().cpu().numpy()
    cv2.imwrite(f'{out_dir}/{stamp}_canvas.png', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    with open(f'{out_dir}/{stamp}_config.json', 'w') as f:
        json.dump(asdict(config), f, indent=2)
    logger.info('Snapshot saved: %s/%s_*', out_dir, stamp)


def run_live(sim_factory: Callable[[], Simulation], config: SwarmConfig, display_scale: int = 3):
    """
    Interactive tuning window. Trackbars edit the hot parameters of `config`
    in place -- the simulation is never rebuilt on a slider move, so you
    watch the swarm react instead of guessing from a still frame.

    Keys: space=pause, .=single step (also pauses), r=reset (same config,
    fresh seed state), 1-5=switch view (canvas/error/pheromone/population/
    genes), s=save frame+config, q/Esc=quit.
    """
    win = 'swarm lab'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    for label, attr, scale, max_scaled in _HOT_PARAMS:
        start = int(round(getattr(config, attr) * scale))
        cv2.createTrackbar(label, win, min(start, max_scaled), max_scaled, lambda v: None)

    sim = sim_factory()
    view = 0
    paused = False
    single_step = False

    logger.info('Live lab: space=pause . =step r=reset 1-5=view s=save q=quit')

    while True:
        for label, attr, scale, _ in _HOT_PARAMS:
            setattr(config, attr, cv2.getTrackbarPos(label, win) / scale)

        if not paused or single_step:
            sim.step()
            single_step = False

        frame = render_view(sim, view)
        frame = cv2.resize(frame, (frame.shape[1] * display_scale, frame.shape[0] * display_scale),
                            interpolation=cv2.INTER_NEAREST)

        if sim.history:
            m = sim.history[-1]
            text = (f'step {m.step}  err {m.mean_error:.4f}  pop {m.population}  '
                    f'energy {m.total_energy:.1f}  div {m.gene_diversity:.4f}  '
                    f'view:{VIEW_NAMES[view]}{" [paused]" if paused else ""}')
            cv2.putText(frame, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q')):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('.'):
            paused = True
            single_step = True
        elif key == ord('r'):
            sim = sim_factory()
        elif ord('1') <= key <= ord('5'):
            view = key - ord('1')
        elif key == ord('s'):
            _save_snapshot(sim, config)

    cv2.destroyAllWindows()


# ==================== §11.2 sweep ====================

def _write_contact_sheet(thumbs: List[tuple], out_path: str, param: str, label_height: int = 24):
    h, w = thumbs[0][1].shape[:2]
    sheet = np.zeros((h + label_height, w * len(thumbs), 3), dtype=np.uint8)
    for i, (value, frame) in enumerate(thumbs):
        sheet[label_height:, i * w:(i + 1) * w] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(sheet, f'{param}={value:g}', (i * w + 4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, sheet)


def run_sweep(config: SwarmConfig, image_path: str, param: str, values: List[float],
              steps: int = None, out_dir: str = OUT_DIR):
    """
    Headless grid over one parameter (§11.2). N runs at the same seed, a
    contact sheet of final canvases plus a CSV of the §12 metrics -- so
    alternatives get compared against saved output, not memory of a previous
    run.
    """
    steps = steps or config.sim_steps
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')

    thumbs, rows = [], []
    for value in values:
        cfg = copy.deepcopy(config)
        setattr(cfg, param, value)
        sim = make_simulation(cfg, image_path)

        for _ in tqdm(range(steps), desc=f'{param}={value:g}'):
            sim.step()

        thumbs.append((value, sim.render_canvas_srgb().cpu().numpy()))
        m = sim.history[-1]
        rows.append({'param': param, 'value': value, 'step': m.step, 'mean_error': m.mean_error,
                     'population': m.population, 'gene_diversity': m.gene_diversity,
                     'total_energy': m.total_energy, 'mean_gain': m.mean_gain})

    sheet_path = f'{out_dir}/{stamp}_{param}_sweep.png'
    _write_contact_sheet(thumbs, sheet_path, param)

    csv_path = f'{out_dir}/{stamp}_{param}_sweep.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    logger.info('Sweep saved: %s, %s', sheet_path, csv_path)


# ==================== §11.2 compare ====================

def run_compare(config_a: SwarmConfig, config_b: SwarmConfig, image_path: str,
                 label_a: str = 'A', label_b: str = 'B', display_scale: int = 3):
    """Two configs, same seed and target, advanced in lockstep (§11.2)."""
    sim_a = make_simulation(config_a, image_path)
    sim_b = make_simulation(config_b, image_path)

    win = 'swarm lab -- compare'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    paused = False

    logger.info('Compare lab: space=pause q=quit')

    while True:
        if not paused:
            sim_a.step()
            sim_b.step()

        frame_a = cv2.cvtColor(sim_a.render_canvas_srgb().cpu().numpy(), cv2.COLOR_RGB2BGR)
        frame_b = cv2.cvtColor(sim_b.render_canvas_srgb().cpu().numpy(), cv2.COLOR_RGB2BGR)
        gap = np.full((frame_a.shape[0], 4, 3), 128, dtype=np.uint8)
        pair = np.concatenate([frame_a, gap, frame_b], axis=1)
        pair = cv2.resize(pair, (pair.shape[1] * display_scale, pair.shape[0] * display_scale),
                           interpolation=cv2.INTER_NEAREST)

        if sim_a.history and sim_b.history:
            m_a, m_b = sim_a.history[-1], sim_b.history[-1]
            cv2.putText(pair, f'{label_a}: err {m_a.mean_error:.4f} pop {m_a.population}',
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            offset = frame_a.shape[1] * display_scale + 4 * display_scale
            cv2.putText(pair, f'{label_b}: err {m_b.mean_error:.4f} pop {m_b.population}',
                        (offset + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(win, pair)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord(' '):
            paused = not paused

    cv2.destroyAllWindows()


# ==================== CLI ====================

def _load_config_overrides(base: SwarmConfig, path: str) -> SwarmConfig:
    """Deep-copy `base` and apply flat scalar overrides from a JSON file."""
    cfg = copy.deepcopy(base)
    if path:
        with open(path) as f:
            overrides = json.load(f)
        for key, value in overrides.items():
            setattr(cfg, key, value)
    return cfg


def main():
    parser = argparse.ArgumentParser(description='Evolutionary swarm testing lab (Evolutionary_Swarm.md §11)')
    parser.add_argument('--mode', choices=['live', 'sweep', 'compare'], default='live')
    parser.add_argument('--image', default='images/jellyfish.png')
    parser.add_argument('--work-size', type=int, default=256)
    parser.add_argument('--population', type=int, default=4096)
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nca-size', type=int, default=128)
    parser.add_argument('--color-noise', type=float, default=0.0)
    parser.add_argument('--posterize', type=int, default=0)
    parser.add_argument('--display-scale', type=int, default=3)
    parser.add_argument('--param', help='sweep: which SwarmConfig field to vary')
    parser.add_argument('--values', help='sweep: comma-separated values, e.g. 0.05,0.1,0.15')
    parser.add_argument('--config-a', help='compare: JSON file of SwarmConfig overrides')
    parser.add_argument('--config-b', help='compare: JSON file of SwarmConfig overrides')
    args = parser.parse_args()

    config = SwarmConfig(work_size=args.work_size, population_cap=args.population,
                          sim_steps=args.steps, device=args.device, seed=args.seed)

    if args.mode == 'live':
        def factory():
            return make_simulation(config, args.image, args.nca_size, args.color_noise, args.posterize)
        run_live(factory, config, display_scale=args.display_scale)

    elif args.mode == 'sweep':
        if not args.param or not args.values:
            parser.error('--param and --values are required for sweep mode')
        values = [float(v) for v in args.values.split(',')]
        run_sweep(config, args.image, args.param, values)

    elif args.mode == 'compare':
        cfg_a = _load_config_overrides(config, args.config_a)
        cfg_b = _load_config_overrides(config, args.config_b)
        run_compare(cfg_a, cfg_b, args.image)


if __name__ == '__main__':
    main()
