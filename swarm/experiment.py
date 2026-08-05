"""
One run of the swarm, from config to artefacts on disk (Evolutionary_Swarm.md §11).

This is the layer the old lab was missing. `sweep` used to keep a single
thumbnail and the last row of the metrics, which is exactly enough information
to *not* notice that the population had been dead since step 50. A run now
leaves behind everything needed to diagnose it after the fact:

    <run>/
      config.json        full SwarmConfig dump -- §11.3's reproducibility rule
      summary.json       the RunSummary: one row, sortable
      metrics.csv        the whole per-step history
      canvas.png         final canvas
      filmstrip.png      base -> intermediate steps -> final -> target
      diagnostics.png    the four §12 curves

`run_experiment` is deliberately pure with respect to the filesystem -- it
returns a `RunResult` and `write_run` puts it on disk -- so the live and
compare modes can reuse the same construction without writing anything.
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from config.lab_config import LabConfig, SurrogateConfig
from config.swarm_config import SwarmConfig, config_to_dict
from swarm.metrics import RunSummary, StepMetrics, summarize, write_history_csv
from swarm.simulation import Simulation
from utils.log import get_logger

logger = get_logger(__name__)


# ==================== §11.1 the surrogate ====================

def load_target(image_path: str, work_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """RGB float32 [0,1] at `work_size`, plus its alpha (the nutrient source)."""
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


def build_surrogate(target_rgb: np.ndarray, work_size: int, surrogate: SurrogateConfig,
                    seed: int = 0) -> np.ndarray:
    """
    The stand-in for the NCA output (§11.1). Downsampling to `nca_size` and
    upsampling back reproduces the actual defect -- missing high frequency --
    without needing a trained NCA. `color_noise` and `posterize_levels`
    optionally corrupt colour and gradation too, so a calibration can be
    checked against a starting point that is *wrong*, not just *soft*.
    """
    small = cv2.resize(target_rgb, (surrogate.nca_size, surrogate.nca_size),
                       interpolation=cv2.INTER_AREA)
    base = cv2.resize(small, (work_size, work_size), interpolation=cv2.INTER_LINEAR)

    if surrogate.posterize_levels > 0:
        levels = max(2, surrogate.posterize_levels)
        base = np.round(base * (levels - 1)) / (levels - 1)

    if surrogate.color_noise > 0:
        rng = np.random.default_rng(seed)
        base = base + rng.normal(0.0, surrogate.color_noise, base.shape).astype(np.float32)

    return np.clip(base, 0.0, 1.0)


def make_simulation(config: SwarmConfig, image_path: str,
                    surrogate: SurrogateConfig = None) -> Simulation:
    """Build a `Simulation` from one test image, using the §11.1 surrogate."""
    surrogate = surrogate or SurrogateConfig()
    target_rgb, alpha = load_target(image_path, config.work_size)
    base_rgb = build_surrogate(target_rgb, config.work_size, surrogate, config.seed)
    nutrient = cv2.GaussianBlur(alpha, (5, 5), 0)

    return Simulation.from_srgb(
        config,
        torch.from_numpy(target_rgb.copy()),
        torch.from_numpy(base_rgb.copy()),
        torch.from_numpy(nutrient.copy()),
    )


# ==================== a run ====================

@dataclass
class RunResult:
    """Everything one simulation produced, in memory. `write_run` persists it."""

    name: str                       # filesystem-safe identifier
    label: str                      # short human label for sheets and legends
    doc: str                        # what this run was testing
    overrides: Dict[str, Any]
    config: SwarmConfig
    summary: RunSummary
    history: List[StepMetrics]

    canvas: np.ndarray              # (H, W, 3) uint8 RGB, final
    base: np.ndarray                # the surrogate the run started from
    target: np.ndarray              # ground truth
    frames: List[Tuple[int, np.ndarray]]   # (step, canvas) captures through the run


def _capture_steps(total: int, count: int) -> List[int]:
    """Evenly spaced interior steps, excluding 0 and the final step."""
    if count <= 0 or total <= 1:
        return []
    return sorted({max(1, round(total * (i + 1) / (count + 1))) for i in range(count)})


def run_experiment(lab: LabConfig, name: str, overrides: Dict[str, Any] = None,
                   doc: str = '', label: str = None, progress: bool = True) -> RunResult:
    """
    Run one configuration to completion and collect its artefacts.

    Reproducibility is by seed and device only (§11.3): the same `overrides` on
    the same machine produce the same canvas bit for bit, which is what makes a
    saved contact sheet worth more than a memory of the previous run.
    """
    config = lab.to_swarm_config(overrides)
    sim = make_simulation(config, lab.image, lab.surrogate)

    target = sim.render_target_srgb().cpu().numpy()
    base = sim.render_base_srgb().cpu().numpy()

    captures = _capture_steps(config.sim_steps, lab.filmstrip_frames)
    frames: List[Tuple[int, np.ndarray]] = [(0, base)]

    iterator = range(config.sim_steps)
    if progress:
        iterator = tqdm(iterator, desc=label or name, leave=False)

    start = time.perf_counter()
    for _ in iterator:
        sim.step()
        if sim.step_count in captures:
            frames.append((sim.step_count, sim.render_canvas_srgb().cpu().numpy()))
    wall_time = time.perf_counter() - start

    canvas = sim.render_canvas_srgb().cpu().numpy()
    frames.append((sim.step_count, canvas))

    summary = summarize(sim.history, sim.baseline_error, sim.population_floor, wall_time)

    return RunResult(
        name=name, label=label or name, doc=doc, overrides=dict(overrides or {}),
        config=config, summary=summary, history=sim.history,
        canvas=canvas, base=base, target=target, frames=frames,
    )


def write_run(result: RunResult, out_dir: Path) -> Path:
    """Persist a `RunResult`. Returns the directory it wrote into."""
    from swarm.report import save_diagnostics, save_filmstrip, save_rgb

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_rgb(out_dir / 'canvas.png', result.canvas)
    save_filmstrip(out_dir / 'filmstrip.png', result)
    save_diagnostics(out_dir / 'diagnostics.png', result)
    write_history_csv(result.history, out_dir / 'metrics.csv')

    with open(out_dir / 'config.json', 'w') as handle:
        json.dump({'name': result.name, 'doc': result.doc,
                   'overrides': result.overrides,
                   'config': config_to_dict(result.config)}, handle, indent=2)

    with open(out_dir / 'summary.json', 'w') as handle:
        json.dump(asdict(result.summary), handle, indent=2)

    return out_dir


def slugify(text: str) -> str:
    """Filesystem-safe run name. Keeps '=' readable as '_' and drops the rest."""
    safe = []
    for ch in str(text):
        safe.append(ch if ch.isalnum() or ch in '-.' else '_')
    return ''.join(safe).strip('_') or 'run'
