"""
The swarm testing suite (Evolutionary_Swarm.md §11).

The swarm is not integrated into the pipeline yet -- there are too many coupled
parameters to tune against a full render. This module runs the same
`Simulation` engine against a single test image, with a surrogate for the NCA
output (§11.1), so tuning can start before stage 2 exists.

Six modes, one engine:

    python -m swarm.lab --mode live                     # interactive, trackbars
    python -m swarm.lab --mode preset --preset fed      # one named config, artefacts on disk
    python -m swarm.lab --mode compare --preset defaults --preset-b fed
    python -m swarm.lab --mode sweep --param tolerance --values 0,0.01,0.02
    python -m swarm.lab --mode study --study survival   # a named grid -> showcase folder
    python -m swarm.lab --mode showcase                 # every curated preset, side by side

`live` and `compare` need a display; everything else is headless and writes a
self-contained folder under `outputs/swarm/studies/` with a contact sheet, a
ranked table, per-run diagnostics, and an `index.html` tying them together.
The presets and studies themselves live in `config/lab_config.py` -- a run is
named and reproducible, not a command line somebody remembers typing.
"""

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from config.lab_config import LabConfig, PRESETS, SHOWCASE, STUDIES, SurrogateConfig
from config.swarm_config import SwarmConfig, config_to_dict, known_params, set_param
from swarm.agents import population_density
from swarm.colorspace import oklab_to_srgb
from swarm.experiment import make_simulation, run_experiment, slugify, write_run
from swarm.report import (save_contact_sheet, save_grid_sheet, save_rgb,
                          write_gallery_html, write_showcase_html, write_summary_csv)
from swarm.simulation import Simulation
from utils.log import get_logger

logger = get_logger(__name__)

VIEW_NAMES = ['canvas', 'error', 'pheromone', 'population', 'genes']


# ==================== views (§11.2) ====================

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
    if sim.agents.alive.any():
        pos = sim.agents.pos[sim.agents.alive].cpu()
        srgb = oklab_to_srgb(sim.agents.gene[sim.agents.alive]).cpu().numpy()
        xi = np.clip(pos[:, 0].numpy().astype(int), 0, w - 1)
        yi = np.clip(pos[:, 1].numpy().astype(int), 0, h - 1)
        img[yi, xi] = srgb
    return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


# ==================== §11.2 live ====================

# (label, dotted param, slider scale, max slider value). Every entry here now
# actually reaches the simulation: before the climate curves became multipliers,
# `cost_life`, `gain_scale` and `mutation_sigma` were read from the curve and
# these sliders moved nothing.
_HOT_PARAMS = [
    ('gain_scale x10', 'gain_scale', 10, 400),
    ('cost_life x10000', 'cost_life', 10000, 300),
    ('cost_deposit x10000', 'cost_deposit', 10000, 500),
    ('deposit_alpha x1000', 'deposit_alpha', 1000, 600),
    ('brush_radius x10', 'brush_radius', 10, 50),
    ('tolerance x1000', 'tolerance', 1000, 200),
    ('mutation_sigma x1000', 'mutation_sigma', 1000, 200),
    ('evaporation x100', 'evaporation', 100, 100),
    ('reproduction_thr x100', 'reproduction_threshold', 100, 500),
]


def _save_snapshot(sim: Simulation, config: SwarmConfig, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')

    save_rgb(out_dir / f'{stamp}_canvas.png', sim.render_canvas_srgb().cpu().numpy())
    with open(out_dir / f'{stamp}_config.json', 'w') as handle:
        json.dump(config_to_dict(config), handle, indent=2)
    logger.info('Snapshot saved: %s/%s_*', out_dir, stamp)


def run_live(lab: LabConfig, overrides: Dict[str, Any] = None):
    """
    Interactive tuning window. Trackbars edit the live config in place -- the
    simulation is never rebuilt on a slider move, so you watch the swarm react
    instead of guessing from a still frame.

    Keys: space=pause, .=single step, r=reset (same config, fresh state),
    1-5=switch view, s=save frame+config, q/Esc=quit.
    """
    config = lab.to_swarm_config(overrides)
    config.metrics_stride = 1        # the HUD reads history[-1] every frame

    win = 'swarm lab'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    for label, param, scale, max_scaled in _HOT_PARAMS:
        start = int(round(getattr(config, param) * scale))
        cv2.createTrackbar(label, win, min(start, max_scaled), max_scaled, lambda v: None)

    def build():
        return make_simulation(config, lab.image, lab.surrogate)

    sim = build()
    view, paused, single_step = 0, False, False
    logger.info('Live lab: space=pause . =step r=reset 1-5=view s=save q=quit')

    while True:
        for label, param, scale, _ in _HOT_PARAMS:
            set_param(config, param, cv2.getTrackbarPos(label, win) / scale)

        if not paused or single_step:
            sim.step()
            single_step = False

        frame = render_view(sim, view)
        frame = cv2.resize(frame, (frame.shape[1] * lab.display_scale, frame.shape[0] * lab.display_scale),
                           interpolation=cv2.INTER_NEAREST)

        if sim.history:
            m = sim.history[-1]
            delta = (1.0 - m.mean_error / max(sim.baseline_error, 1e-9)) * 100.0
            cv2.putText(frame, f'step {m.step}  err {m.mean_error:.4f} ({delta:+.1f}% vs base)  '
                               f'pop {m.population}  eating {m.positive_gain_fraction * 100:.0f}%  '
                               f'div {m.gene_diversity:.4f}  view:{VIEW_NAMES[view]}'
                               f'{" [paused]" if paused else ""}',
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q')):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('.'):
            paused, single_step = True, True
        elif key == ord('r'):
            sim = build()
        elif ord('1') <= key <= ord('5'):
            view = key - ord('1')
        elif key == ord('s'):
            _save_snapshot(sim, config, Path(lab.output_root) / 'snapshots')

    cv2.destroyAllWindows()


# ==================== §11.2 compare ====================

def run_compare(lab: LabConfig, overrides_a: Dict[str, Any], overrides_b: Dict[str, Any],
                label_a: str = 'A', label_b: str = 'B'):
    """Two configs, same seed and target, advanced in lockstep (§11.2)."""
    sim_a = make_simulation(lab.to_swarm_config(overrides_a), lab.image, lab.surrogate)
    sim_b = make_simulation(lab.to_swarm_config(overrides_b), lab.image, lab.surrogate)

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
        pair = cv2.resize(pair, (pair.shape[1] * lab.display_scale, pair.shape[0] * lab.display_scale),
                          interpolation=cv2.INTER_NEAREST)

        for sim, label, x in ((sim_a, label_a, 8),
                              (sim_b, label_b, (frame_a.shape[1] + 4) * lab.display_scale + 8)):
            if not sim.history:
                continue
            m = sim.history[-1]
            delta = (1.0 - m.mean_error / max(sim.baseline_error, 1e-9)) * 100.0
            cv2.putText(pair, f'{label}: {delta:+.1f}% vs base  pop {m.population}',
                        (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(win, pair)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord(' '):
            paused = not paused

    cv2.destroyAllWindows()


# ==================== headless batches ====================

def _batch_dir(lab: LabConfig, name: str) -> Path:
    out = Path(lab.output_root) / 'studies' / f'{time.strftime("%Y%m%d_%H%M%S")}_{slugify(name)}'
    out.mkdir(parents=True, exist_ok=True)
    return out


def _finish_batch(lab: LabConfig, out_dir: Path, results: Sequence, title: str, doc: str,
                  sheets: Dict[str, str]) -> Path:
    """Write the cross-run artefacts and refresh the top-level gallery."""
    write_summary_csv(out_dir / 'summary.csv', results)
    write_showcase_html(out_dir / 'index.html', title, doc, results, sheets)
    refresh_gallery(lab)

    best = max(results, key=lambda r: r.summary.improvement)
    logger.info('--- %s: %d runs ---', title, len(results))
    logger.info('Best: %s at %+.2f%% vs the stage-2 baseline (pop %d, births %d)',
                best.label, best.summary.improvement * 100,
                best.summary.final_population, best.summary.total_births)
    logger.info('Showcase: %s', out_dir / 'index.html')
    return out_dir


def run_batch(lab: LabConfig, name: str, doc: str,
              jobs: List[Tuple[str, str, str, Dict[str, Any]]]) -> Path:
    """
    Run a list of `(run_name, label, doc, overrides)` jobs into one showcase folder.

    The shared path behind `preset`, `showcase` and `sweep`; `study` adds a grid
    sheet on top because it knows its two axes.
    """
    out_dir = _batch_dir(lab, name)
    results = []

    for run_name, label, run_doc, overrides in jobs:
        logger.info('[%d/%d] %s', len(results) + 1, len(jobs), label)
        result = run_experiment(lab, run_name, overrides, doc=run_doc, label=label)
        write_run(result, out_dir / run_name)
        results.append(result)
        s = result.summary
        logger.info('    %+.2f%% vs baseline | pop %d (floor %.0f%% of run) | births %d | %.0f steps/s',
                    s.improvement * 100, s.final_population, s.floor_fraction * 100,
                    s.total_births, s.steps_per_sec)

    save_contact_sheet(out_dir / 'contact_sheet.png', results, cell=lab.thumb_size, title=name)
    return _finish_batch(lab, out_dir, results, name, doc,
                         {'Final canvases': 'contact_sheet.png'})


def run_study(lab: LabConfig, study_name: str) -> Path:
    """
    A named grid from `config/lab_config.py`, written as a showcase folder.

    Two-axis studies also get a matrix sheet: a 4x4 grid flattened into a
    16-wide strip hides exactly the interaction the grid was run to find.
    """
    if study_name not in STUDIES:
        raise KeyError(f'unknown study {study_name!r}; known: {", ".join(sorted(STUDIES))}')
    study = STUDIES[study_name]

    lab = _with_steps(lab, study.steps)
    out_dir = _batch_dir(lab, study_name)
    combinations = list(itertools.product(*[[(a.param, v) for v in a.values] for a in study.axes]))
    results = []

    logger.info('Study %r: %d runs x %d steps -- %s',
                study_name, len(combinations), lab.steps, study.doc)

    for combo in combinations:
        overrides = dict(study.base)
        overrides.update(dict(combo))
        label = ' '.join(f'{p}={v:g}' if isinstance(v, (int, float)) else f'{p}={v}'
                         for p, v in combo)
        run_name = slugify(label.replace(' ', '__'))

        logger.info('[%d/%d] %s', len(results) + 1, len(combinations), label)
        result = run_experiment(lab, run_name, overrides, doc=study.doc, label=label)
        write_run(result, out_dir / run_name)
        results.append(result)
        s = result.summary
        logger.info('    %+.2f%% vs baseline | pop %d (floor %.0f%% of run) | births %d',
                    s.improvement * 100, s.final_population, s.floor_fraction * 100, s.total_births)

    sheets = {}
    save_contact_sheet(out_dir / 'contact_sheet.png', results, cell=lab.thumb_size, title=study_name)
    if len(study.axes) == 2:
        save_grid_sheet(out_dir / 'grid.png', results, study.axes[0].param, study.axes[1].param,
                        title=f'{study_name}: {study.axes[0].param} (x) vs {study.axes[1].param} (y)')
        sheets['Parameter matrix'] = 'grid.png'
    sheets['Final canvases'] = 'contact_sheet.png'

    return _finish_batch(lab, out_dir, results, f'study: {study_name}', study.doc, sheets)


def run_showcase(lab: LabConfig, names: List[str] = None) -> Path:
    """Every curated preset, rendered side by side (`SHOWCASE` in lab_config)."""
    names = names or SHOWCASE
    jobs = [(slugify(n), n, PRESETS[n].doc, PRESETS[n].overrides) for n in names]
    return run_batch(lab, 'showcase', 'Curated presets on one target, same seed, same steps.', jobs)


def run_sweep(lab: LabConfig, params: List[str], values: List[List[Any]],
              base: Dict[str, Any] = None) -> Path:
    """Ad-hoc grid over 1-2 parameters given on the command line."""
    base = base or {}
    combinations = list(itertools.product(*[[(p, v) for v in vals]
                                            for p, vals in zip(params, values)]))
    jobs = []
    for combo in combinations:
        overrides = dict(base)
        overrides.update(dict(combo))
        label = ' '.join(f'{p}={v}' for p, v in combo)
        jobs.append((slugify(label.replace(' ', '__')), label, 'ad-hoc sweep', overrides))

    name = '_'.join(params) + '_sweep'
    out_dir = run_batch(lab, name, f'Ad-hoc sweep over {", ".join(params)}.', jobs)

    return out_dir


def refresh_gallery(lab: LabConfig) -> Path:
    """Rebuild `outputs/swarm/index.html` from whatever study folders exist."""
    root = Path(lab.output_root)
    studies_dir = root / 'studies'
    entries = []

    for folder in sorted(studies_dir.glob('*'), reverse=True):
        index = folder / 'index.html'
        summary = folder / 'summary.csv'
        if not index.exists():
            continue

        best, runs = '', ''
        if summary.exists():
            import csv
            with open(summary, newline='') as handle:
                rows = list(csv.DictReader(handle))
            runs = str(len(rows))
            if rows:
                top = max(rows, key=lambda r: float(r['improvement']))
                best = f'best {top["label"]} at {float(top["improvement"]) * 100:+.1f}%'

        entries.append({'href': f'studies/{folder.name}/index.html',
                        'title': folder.name, 'doc': '', 'runs': runs, 'best': best})

    root.mkdir(parents=True, exist_ok=True)
    write_gallery_html(root / 'index.html', entries)
    return root / 'index.html'


# ==================== CLI ====================

def _with_steps(lab: LabConfig, steps: int = None) -> LabConfig:
    if steps is None:
        return lab
    import copy
    out = copy.deepcopy(lab)
    out.steps = steps
    return out


def _parse_value(text: str) -> Any:
    """CLI values stay strings only when they have to; `set_param` coerces after."""
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def _overrides_from_args(args) -> Dict[str, Any]:
    """Preset + JSON file + inline `--set k=v`, applied in that order."""
    overrides: Dict[str, Any] = {}

    if args.preset:
        if args.preset not in PRESETS:
            raise SystemExit(f'unknown preset {args.preset!r}; known: {", ".join(sorted(PRESETS))}')
        overrides.update(PRESETS[args.preset].overrides)

    if args.config:
        with open(args.config) as handle:
            overrides.update(json.load(handle))

    for item in args.set or []:
        key, _, value = item.partition('=')
        if not value:
            raise SystemExit(f'--set expects key=value, got {item!r}')
        overrides[key.strip()] = _parse_value(value.strip())

    unknown = set(overrides) - set(known_params())
    if unknown:
        raise SystemExit(f'unknown swarm parameter(s): {", ".join(sorted(unknown))}')

    return overrides


def main():
    parser = argparse.ArgumentParser(
        description='Evolutionary swarm laboratory (docs/Evolutionary_Swarm.md §11)')
    parser.add_argument('--mode', default='live',
                        choices=['live', 'preset', 'compare', 'sweep', 'study', 'showcase', 'gallery'])

    parser.add_argument('--image', default=None)
    parser.add_argument('--work-size', type=int, default=None)
    parser.add_argument('--population', type=int, default=None)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=None)

    parser.add_argument('--nca-size', type=int, default=None, help='§11.1 surrogate resolution')
    parser.add_argument('--color-noise', type=float, default=None, help='§11.1 surrogate colour corruption')
    parser.add_argument('--posterize', type=int, default=None, help='§11.1 surrogate gradation loss')

    parser.add_argument('--preset', default=None, help=f'one of: {", ".join(sorted(PRESETS))}')
    parser.add_argument('--preset-b', default=None, help='compare: the right-hand preset')
    parser.add_argument('--config', default=None, help='JSON file of SwarmConfig overrides')
    parser.add_argument('--config-b', default=None, help='compare: right-hand JSON overrides')
    parser.add_argument('--set', action='append', metavar='PARAM=VALUE',
                        help='inline override, repeatable; dotted paths allowed '
                             '(e.g. climate_cost_life.end=4.0)')

    parser.add_argument('--study', default=None, help=f'one of: {", ".join(sorted(STUDIES))}')
    parser.add_argument('--param', action='append', help='sweep: parameter to vary, repeatable (max 2)')
    parser.add_argument('--values', action='append',
                        help='sweep: comma-separated values, one --values per --param')
    parser.add_argument('--only', action='append', help='showcase: restrict to these presets')

    args = parser.parse_args()

    lab = LabConfig()
    for attr, value in (('image', args.image), ('work_size', args.work_size),
                        ('population_cap', args.population), ('steps', args.steps),
                        ('device', args.device), ('seed', args.seed)):
        if value is not None:
            setattr(lab, attr, value)

    surrogate = SurrogateConfig()
    for attr, value in (('nca_size', args.nca_size), ('color_noise', args.color_noise),
                        ('posterize_levels', args.posterize)):
        if value is not None:
            setattr(surrogate, attr, value)
    lab.surrogate = surrogate

    overrides = _overrides_from_args(args)
    logger.info('Lab: %s | %dpx | %d agents | %d steps | device %s | seed %d',
                lab.image, lab.work_size, lab.population_cap, lab.steps, lab.device, lab.seed)

    if args.mode == 'live':
        run_live(lab, overrides)

    elif args.mode == 'preset':
        name = args.preset or 'custom'
        doc = PRESETS[name].doc if name in PRESETS else 'ad-hoc configuration'
        run_batch(lab, name, doc, [(slugify(name), name, doc, overrides)])

    elif args.mode == 'compare':
        right = dict(overrides)
        if args.preset_b:
            if args.preset_b not in PRESETS:
                raise SystemExit(f'unknown preset {args.preset_b!r}')
            right = dict(PRESETS[args.preset_b].overrides)
        if args.config_b:
            with open(args.config_b) as handle:
                right.update(json.load(handle))
        run_compare(lab, overrides, right, args.preset or 'A', args.preset_b or 'B')

    elif args.mode == 'sweep':
        if not args.param or not args.values or len(args.param) != len(args.values):
            parser.error('sweep needs matching --param/--values pairs (at most two)')
        if len(args.param) > 2:
            parser.error('sweep takes at most two axes')
        values = [[_parse_value(v) for v in group.split(',')] for group in args.values]
        run_sweep(lab, args.param, values, base=overrides)

    elif args.mode == 'study':
        if not args.study:
            parser.error('--study is required; known: ' + ', '.join(sorted(STUDIES)))
        run_study(lab, args.study)

    elif args.mode == 'showcase':
        run_showcase(lab, args.only)

    elif args.mode == 'gallery':
        refresh_gallery(lab)


if __name__ == '__main__':
    main()
