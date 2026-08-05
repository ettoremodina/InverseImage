"""
The tuning suite: a search that calibrates stage 3 without a human in the loop.

    python -m swarm.tuning                            # the default search
    python -m swarm.tuning --space full --generations 60
    python -m swarm.tuning --space economy --seed-from rooted --workers 8
    python -m swarm.tuning --adopt                    # install the winner as `tuned`

The lab (`swarm/lab.py`) can only compare configurations a person thought of.
That is fine for a two-parameter question and useless for a coupled dozen --
§13 lists "tuning permaloso" as a named risk, and the studies run so far bear
it out: the calibrated `rooted` preset still leaves the picture 0.5% *worse*
than stage 2 handed it over. This module closes that loop mechanically:
propose a configuration, run it, score it against the success criteria in
`config/tuning_config.py`, keep what scores well, repeat.

**How the search works.** A (μ+λ) evolution strategy over the normalised unit
cube, with one adaptive step size shared by all dimensions:

1. `initial_random` probes, one of which is the `seed_from` preset, so the
   search never begins less informed than the last hand calibration.
2. Each generation, λ children are sampled around the μ elites -- a parent
   chosen by rank, a gaussian step, reflected at the walls of the cube so
   nothing piles up on a boundary.
3. Parents and children compete together (that is the `+` in (μ+λ): an elite
   survives until something actually beats it, which matters when the
   objective is noisy).
4. The step size grows after a generation that improved the best and shrinks
   after one that did not -- coarse exploration first, refinement once the
   region is found.

It is worth saying plainly that this is the same algorithm as the thing being
tuned, one level up: a population, a mutation, and a selection that can only
say no. The swarm evolves colour against the picture; this evolves the swarm's
constants against the success criteria.

**What it leaves behind.** Everything lands under
`outputs/swarm/tuning/<timestamp>_<space>/`:

    index.html          the page to actually open: best so far, curves, films
    history.csv         every evaluation, every metric, every parameter
    progress.png        score curves, the current best's scorecard, drift
    best.json           the winning overrides -- feeds `lab.py --config`
    generations/        per-generation best canvas + mp4 (every `animate_every`)
    progress.mp4        one held shot per generation: the search, as a film
    final/              the winner re-run long, on several seeds, with video

The animation cadence is the point of `animate_every`: metrics say a
configuration improved, film says whether it looks like painting. Nothing in
the loop asks a language model to judge anything -- the criteria are arithmetic
and the pictures are for the person who reads the report afterwards.
"""

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config.lab_config import LabConfig, PRESETS, SurrogateConfig
from config.swarm_config import SwarmConfig, apply_overrides, config_to_dict
from config.tuning_config import ObjectiveConfig, Param, SPACES, TuningConfig
from swarm.metrics import summarize
from swarm.objective import evaluate as score_run, describe
from swarm.quality import measure_simulation
from utils.log import get_logger

logger = get_logger(__name__)


# ==================== one evaluation ====================

@dataclass
class Evaluation:
    """One configuration, run and scored."""

    index: int
    generation: int
    unit: List[float]                       # coordinates in the normalised cube
    overrides: Dict[str, Any]               # the same point, in SwarmConfig units
    score: float
    parts: Dict[str, float] = field(default_factory=dict)     # per-criterion sub-scores
    metrics: Dict[str, float] = field(default_factory=dict)   # the flattened readings
    seed_scores: List[float] = field(default_factory=list)
    wall_time: float = 0.0
    thumb: Optional[np.ndarray] = None      # small RGB of the final canvas

    @property
    def label(self) -> str:
        return f'gen{self.generation:02d}#{self.index:04d}'


def _lab_from(tuning: TuningConfig, steps: int = None, seed: int = 0) -> LabConfig:
    """A `LabConfig` matching the tuning settings -- one place builds these."""
    lab = LabConfig(image=tuning.image)
    lab.work_size = tuning.work_size
    lab.population_cap = tuning.population_cap
    lab.steps = steps or tuning.steps
    lab.device = tuning.device
    lab.seed = seed
    lab.metrics_stride = tuning.metrics_stride
    lab.surrogate = SurrogateConfig()
    return lab


def run_and_score(tuning: TuningConfig, objective: ObjectiveConfig,
                  overrides: Dict[str, Any], seed: int, steps: int = None,
                  want_thumb: bool = False):
    """
    Run one configuration on one seed and score it. The unit of work of the
    whole suite -- called in-process here and inside the worker pool below.

    Returns `(score, parts, metrics, thumb)`.
    """
    from swarm.experiment import make_simulation      # local: keeps worker import light

    lab = _lab_from(tuning, steps=steps, seed=seed)
    config = lab.to_swarm_config(overrides)

    start = time.perf_counter()
    sim = make_simulation(config, lab.image, lab.surrogate)
    for _ in range(config.sim_steps):
        sim.step()
    wall = time.perf_counter() - start

    summary = summarize(sim.history, sim.baseline_error, sim.population_floor, wall)
    quality = measure_simulation(sim)
    result = score_run(summary, quality, config, objective)

    thumb = None
    if want_thumb:
        import cv2
        canvas = sim.render_canvas_srgb().cpu().numpy()
        thumb = cv2.resize(canvas, (tuning.thumb_size, tuning.thumb_size),
                           interpolation=cv2.INTER_AREA)

    from swarm.objective import flatten
    metrics = flatten(summary, quality, config)
    parts = {p.key: p.score for p in result.parts}
    return result.total, parts, metrics, thumb


def _worker(payload: Tuple) -> Dict[str, Any]:
    """
    Pool entry point. Must stay module level and picklable -- Windows spawns
    workers rather than forking, so a closure or a lambda would not survive.
    """
    tuning, objective, overrides, seeds, want_thumb, threads = payload

    import torch
    torch.set_num_threads(max(1, int(threads)))

    scores, metric_rows, part_rows, thumb = [], [], [], None
    start = time.perf_counter()
    for i, seed in enumerate(seeds):
        total, parts, metrics, image = run_and_score(
            tuning, objective, overrides, seed, want_thumb=want_thumb and i == 0)
        scores.append(total)
        part_rows.append(parts)
        metric_rows.append(metrics)
        if image is not None:
            thumb = image

    return {
        'score': float(np.mean(scores)),
        'seed_scores': [float(s) for s in scores],
        'parts': _mean_dicts(part_rows),
        'metrics': _mean_dicts(metric_rows),
        'wall_time': time.perf_counter() - start,
        'thumb': thumb,
    }


def _mean_dicts(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Average a list of same-keyed numeric dicts, skipping non-numeric values."""
    if not rows:
        return {}
    out = {}
    for key in rows[0]:
        values = [r[key] for r in rows if isinstance(r.get(key), (int, float, bool))]
        if values:
            out[key] = float(np.mean([float(v) for v in values]))
    return out


# ==================== the search space, in coordinates ====================

class Space:
    """A named list of `Param`s, plus the mapping to and from the unit cube."""

    def __init__(self, name: str):
        if name not in SPACES:
            raise KeyError(f'unknown search space {name!r}; known: {", ".join(sorted(SPACES))}')
        self.name = name
        self.params: List[Param] = list(SPACES[name])

    def __len__(self) -> int:
        return len(self.params)

    @property
    def paths(self) -> List[str]:
        return [p.path for p in self.params]

    def to_overrides(self, unit: Sequence[float]) -> Dict[str, Any]:
        return {p.path: p.to_value(u) for p, u in zip(self.params, unit)}

    def from_overrides(self, overrides: Dict[str, Any], reference: SwarmConfig = None) -> List[float]:
        """
        Coordinates for an existing configuration.

        Parameters the preset does not mention fall back to the `SwarmConfig`
        default, so seeding from a partial preset lands on the point that
        preset actually describes rather than in a corner of the cube.
        """
        reference = reference or SwarmConfig()
        unit = []
        for param in self.params:
            value = overrides.get(param.path, getattr(reference, param.path, None))
            unit.append(param.to_unit(value) if value is not None else 0.5)
        return unit


def _reflect(value: float) -> float:
    """
    Fold a coordinate back into [0, 1] instead of clamping it.

    Clamping makes the walls of the cube attractors: half of a gaussian step
    near an edge lands on the edge exactly, so the search accumulates there and
    reports "the optimum is at the boundary" when it is nothing of the kind.
    """
    for _ in range(8):
        if value < 0.0:
            value = -value
        elif value > 1.0:
            value = 2.0 - value
        else:
            return value
    return min(1.0, max(0.0, value))


# ==================== the tuner ====================

class Tuner:
    """
    The search loop and its bookkeeping.

    Kept as a class only because a search has state worth inspecting after it
    stops: `history` is every evaluation ever run, `elites` the current front,
    `best` the answer.
    """

    def __init__(self, tuning: TuningConfig, objective: ObjectiveConfig = None,
                 out_dir: Path = None):
        self.tuning = tuning
        self.objective = objective or ObjectiveConfig()
        self.space = Space(tuning.space)
        self.rng = random.Random(12345)

        stamp = time.strftime('%Y%m%d_%H%M%S')
        self.out_dir = Path(out_dir or Path(tuning.output_root) / f'{stamp}_{tuning.space}')
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / 'generations').mkdir(exist_ok=True)

        self.history: List[Evaluation] = []
        self.elites: List[Evaluation] = []
        self.generation = 0
        self.sigma = tuning.sigma_init
        self.reel: List[Tuple[str, np.ndarray]] = []
        self._counter = 0
        self._pool: Optional[ProcessPoolExecutor] = None

    # ------------------------------------------------------------ evaluation

    def _submit(self, units: List[List[float]], want_thumb: bool = True) -> List[Evaluation]:
        """Evaluate a batch of coordinates, in the pool when there is one."""
        payloads = []
        for unit in units:
            overrides = self.space.to_overrides(unit)
            payloads.append((self.tuning, self.objective, overrides, list(self.tuning.seeds),
                             want_thumb, self.tuning.threads_per_worker))

        if self._pool is not None:
            raw = list(self._pool.map(_worker, payloads))
        else:
            raw = [_worker(p) for p in payloads]

        out = []
        for unit, payload, result in zip(units, payloads, raw):
            self._counter += 1
            out.append(Evaluation(
                index=self._counter, generation=self.generation, unit=list(unit),
                overrides=payload[2], score=result['score'], parts=result['parts'],
                metrics=result['metrics'], seed_scores=result['seed_scores'],
                wall_time=result['wall_time'], thumb=result['thumb'],
            ))
        self.history.extend(out)
        return out

    # ------------------------------------------------------------ sampling

    def _initial_units(self) -> List[List[float]]:
        """
        The opening probes: the seed preset, then a stratified random fill.

        Stratified rather than uniform because with 30-odd probes in a dozen
        dimensions, plain uniform sampling leaves whole decades of the important
        log-scaled axes untouched by luck alone.
        """
        units: List[List[float]] = []

        preset = PRESETS.get(self.tuning.seed_from)
        if preset is not None:
            units.append(self.space.from_overrides(preset.overrides))
            logger.info("Seeding the search from preset '%s'", self.tuning.seed_from)

        count = max(0, self.tuning.initial_random - len(units))
        if count:
            strata = []
            for _ in range(len(self.space)):
                column = [(i + self.rng.random()) / count for i in range(count)]
                self.rng.shuffle(column)
                strata.append(column)
            units += [[strata[d][i] for d in range(len(self.space))] for i in range(count)]

        return units

    def _children(self) -> List[List[float]]:
        """λ children around the elites, by rank-weighted parent choice."""
        weights = [1.0 / (i + 1) for i in range(len(self.elites))]   # 1, 1/2, 1/3, ...
        children = []
        for _ in range(self.tuning.children):
            parent = self.rng.choices(self.elites, weights=weights, k=1)[0]
            children.append([_reflect(u + self.rng.gauss(0.0, self.sigma)) for u in parent.unit])
        return children

    def _adapt_sigma(self, improved: bool):
        factor = self.tuning.sigma_grow if improved else self.tuning.sigma_shrink
        self.sigma = min(self.tuning.sigma_max, max(self.tuning.sigma_min, self.sigma * factor))

    # ------------------------------------------------------------ the loop

    @property
    def best(self) -> Optional[Evaluation]:
        return self.elites[0] if self.elites else None

    def run(self) -> Optional[Evaluation]:
        """Search until the target score, the generation budget or patience runs out."""
        logger.info('Tuning %r: %d parameters, budget %d evaluations, %d worker(s)',
                    self.space.name, len(self.space), self.tuning.evaluations_budget(),
                    max(1, self.tuning.workers))
        logger.info('Output: %s', self.out_dir)

        self._open_pool()
        try:
            batch = self._submit(self._initial_units())
            self.elites = sorted(batch, key=lambda e: -e.score)[:self.tuning.elites]
            self._after_generation(batch, improved=True)

            stale = 0
            while self.generation < self.tuning.generations:
                if self.best.score >= self.objective.target_score:
                    logger.info('Target score %.2f reached -- stopping early',
                                self.objective.target_score)
                    break
                if stale >= self.tuning.patience:
                    logger.info('No improvement for %d generations -- stopping', stale)
                    break

                self.generation += 1
                previous = self.best.score

                batch = self._submit(self._children())
                pool = self.elites + batch
                self.elites = sorted(pool, key=lambda e: -e.score)[:self.tuning.elites]

                improved = self.best.score > previous + 1e-6
                stale = 0 if improved else stale + 1
                self._adapt_sigma(improved)
                self._after_generation(batch, improved)
        finally:
            self._close_pool()

        self._finalise()
        return self.best

    def _open_pool(self):
        if self.tuning.workers and self.tuning.workers > 1:
            self._pool = ProcessPoolExecutor(max_workers=self.tuning.workers)

    def _close_pool(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    # ------------------------------------------------------------ artefacts

    def _after_generation(self, batch: List[Evaluation], improved: bool):
        """Persist everything a person might want to look at mid-search."""
        from swarm.tuning_report import write_history_row, write_progress, write_tuning_html

        for evaluation in batch:
            write_history_row(self.out_dir / 'history.csv', evaluation, self.space.paths)

        best = self.best
        logger.info('gen %02d | best %.4f (%s) | batch best %.4f | sigma %.3f | %s',
                    self.generation, best.score,
                    'improved' if improved else 'held',
                    max(e.score for e in batch), self.sigma,
                    ' '.join(f'{k}={v:.2f}' for k, v in sorted(best.parts.items())[:4]))

        with open(self.out_dir / 'best.json', 'w') as handle:
            json.dump(best.overrides, handle, indent=2)

        if best.thumb is not None:
            self.reel.append((f'gen {self.generation:02d}  score {best.score:.3f}', best.thumb))

        # Thumbnails are only needed for the elites (which may still be filmed)
        # and for the reel (which already has its copy). Dropping the rest keeps
        # a 500-evaluation search from carrying ~50 MB of dead images.
        keep = {id(e) for e in self.elites}
        for evaluation in self.history:
            if id(evaluation) not in keep:
                evaluation.thumb = None

        write_progress(self.out_dir / 'progress.png', self)

        animate = (self.tuning.animate_every > 0
                   and self.generation % self.tuning.animate_every == 0
                   and improved)
        video = self._animate_best() if animate else None

        write_tuning_html(self.out_dir / 'index.html', self, latest_video=video)

    def _animate_best(self) -> Optional[Path]:
        """
        Film the current best.

        Deliberately a separate, longer run than the search evaluations: the
        search is scored at `steps` for throughput, but a film at that length
        would stop before the regime the arc (§7) is supposed to reach.
        """
        from swarm.experiment import run_experiment, write_run

        tuning = self.tuning
        folder = self.out_dir / 'generations' / f'gen{self.generation:02d}'
        folder.mkdir(parents=True, exist_ok=True)

        lab = _lab_from(tuning, steps=tuning.animation_steps, seed=tuning.seeds[0])
        label = f'gen {self.generation:02d} best (score {self.best.score:.3f})'

        logger.info('Filming generation %02d best over %d steps...',
                    self.generation, tuning.animation_steps)
        weakest = min(self.best.parts.items(), key=lambda kv: kv[1])[0] if self.best.parts else '-'
        result = run_experiment(
            lab, name=f'gen{self.generation:02d}', overrides=self.best.overrides,
            doc=f'Best of generation {self.generation} (score {self.best.score:.3f}, '
                f'weakest criterion: {weakest})',
            label=label, progress=False,
            video_path=folder / 'evolution.mp4', video_stride=tuning.animation_capture_stride,
            video_fps=tuning.animation_fps,
        )
        write_run(result, folder)
        return folder / 'evolution.mp4'

    def _best_score_object(self):
        """Rebuild an `ObjectiveScore`-shaped summary line for the best, for captions."""
        from swarm.objective import CriterionScore, ObjectiveScore
        parts = [CriterionScore(key=c.key, value=self.best.metrics.get(c.key, float('nan')),
                                score=self.best.parts.get(c.key, 0.0), weight=c.weight,
                                contribution=self.best.parts.get(c.key, 0.0) * c.weight,
                                doc=c.doc)
                 for c in self.objective.criteria]
        return ObjectiveScore(total=self.best.score, raw=self.best.score, gate=1.0, parts=parts)

    def _finalise(self):
        """The winner, run long and on several seeds, plus the progress reel."""
        from swarm.animate import save_generation_reel
        from swarm.experiment import run_experiment, write_run
        from swarm.tuning_report import write_final_report

        best = self.best
        if best is None:
            logger.warning('Tuning produced no evaluations')
            return

        logger.info('--- best configuration (score %.4f) ---', best.score)
        logger.info('\n%s', describe(self._best_score_object()))

        if self.reel:
            save_generation_reel(self.out_dir / 'progress.mp4', self.reel)

        final_dir = self.out_dir / 'final'
        lab = _lab_from(self.tuning, steps=self.tuning.final_steps, seed=self.tuning.final_seeds[0])
        result = run_experiment(
            lab, name='final', overrides=best.overrides,
            doc=f'Winner of the {self.space.name} search, re-run over {self.tuning.final_steps} steps.',
            label=f'best (score {best.score:.3f})', progress=False,
            video_path=final_dir / 'evolution.mp4',
            video_stride=self.tuning.animation_capture_stride, video_fps=self.tuning.animation_fps,
        )
        write_run(result, final_dir)

        # Re-scored on every final seed: a score that survives a change of seed
        # is a property of the configuration, one that does not is luck.
        confirmations = []
        for seed in self.tuning.final_seeds:
            total, parts, metrics, _ = run_and_score(
                self.tuning, self.objective, best.overrides, seed, steps=self.tuning.final_steps)
            confirmations.append({'seed': seed, 'score': total,
                                  'improvement': metrics.get('improvement', 0.0)})
            logger.info('  seed %d -> score %.4f (improvement %+.2f%%)',
                        seed, total, metrics.get('improvement', 0.0) * 100)

        with open(self.out_dir / 'best.json', 'w') as handle:
            json.dump(best.overrides, handle, indent=2)
        with open(self.out_dir / 'final_report.json', 'w') as handle:
            json.dump({'space': self.space.name, 'score': best.score,
                       'overrides': best.overrides, 'metrics': best.metrics,
                       'parts': best.parts, 'confirmations': confirmations,
                       'evaluations': len(self.history),
                       'config': config_to_dict(apply_overrides(SwarmConfig(), best.overrides))},
                      handle, indent=2)

        write_final_report(self.out_dir / 'index.html', self, result, confirmations)
        logger.info('Report: %s', self.out_dir / 'index.html')


# ==================== adoption ====================

TUNED_CONFIG_PATH = Path('config/tuned_swarm.json')


def adopt(overrides: Dict[str, Any], path: Path = TUNED_CONFIG_PATH) -> Path:
    """
    Install a tuned configuration as the `tuned` preset.

    Writing JSON that `lab_config` reads at import, rather than editing the
    Python, keeps the calibration a *result* -- something a run produced and
    can reproduce -- instead of a number that got typed into a source file and
    then drifted away from the study that justified it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as handle:
        json.dump(overrides, handle, indent=2)
    logger.info('Adopted as the `tuned` preset: %s', path)
    return path


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(
        description='Automatic calibration of the evolutionary swarm '
                    '(docs/Swarm_Tuning.md)')

    parser.add_argument('--space', default='core', help=f'one of: {", ".join(sorted(SPACES))}')
    parser.add_argument('--image', default=None)
    parser.add_argument('--out', default=None, help='output directory (default: timestamped)')

    parser.add_argument('--steps', type=int, default=None, help='simulation steps per evaluation')
    parser.add_argument('--final-steps', type=int, default=None)
    parser.add_argument('--work-size', type=int, default=None)
    parser.add_argument('--population', type=int, default=None)
    parser.add_argument('--seeds', default=None, help='comma-separated seeds per evaluation')

    parser.add_argument('--generations', type=int, default=None)
    parser.add_argument('--initial-random', type=int, default=None)
    parser.add_argument('--elites', type=int, default=None)
    parser.add_argument('--children', type=int, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--seed-from', default=None, help=f'preset to seed from, or "none"')
    parser.add_argument('--target-score', type=float, default=None)

    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--animate-every', type=int, default=None,
                        help='generations between films of the current best; 0 disables')
    parser.add_argument('--adopt', action='store_true',
                        help='write the winner to config/tuned_swarm.json as the `tuned` preset')
    parser.add_argument('--adopt-from', default=None, metavar='BEST_JSON',
                        help='adopt an earlier search\'s best.json without searching again')

    args = parser.parse_args()

    if args.adopt_from:
        with open(args.adopt_from) as handle:
            adopt(json.load(handle))
        return

    tuning = TuningConfig()
    for attr, value in (('space', args.space), ('image', args.image), ('steps', args.steps),
                        ('final_steps', args.final_steps), ('work_size', args.work_size),
                        ('population_cap', args.population), ('generations', args.generations),
                        ('initial_random', args.initial_random), ('elites', args.elites),
                        ('children', args.children), ('patience', args.patience),
                        ('workers', args.workers), ('device', args.device),
                        ('animate_every', args.animate_every)):
        if value is not None:
            setattr(tuning, attr, value)

    if args.seeds:
        tuning.seeds = [int(s) for s in args.seeds.split(',')]
    if args.seed_from is not None:
        tuning.seed_from = '' if args.seed_from.lower() == 'none' else args.seed_from

    objective = ObjectiveConfig()
    if args.target_score is not None:
        objective.target_score = args.target_score

    tuner = Tuner(tuning, objective, out_dir=Path(args.out) if args.out else None)
    best = tuner.run()

    if best is not None and args.adopt:
        adopt(best.overrides)


if __name__ == '__main__':
    main()
