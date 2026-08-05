"""
The face of a tuning run: one page, refreshed after every generation.

A search that runs for an hour is only useful if it can be interrupted and read
at any moment, so every artefact here is rewritten from scratch each generation
rather than appended to at the end. Open `index.html` while the tuner is still
going and it shows the state as of the last completed generation -- score
curves, the current best's scorecard, and whatever films have been shot so far.

Three things get plotted, and the choice of the third is the one worth
explaining. Score-per-evaluation and the best's scorecard are obvious. The
third panel is a parallel-coordinates plot of the elite configurations in
normalised space: it answers "has the search decided?" per parameter. An axis
where all elites sit on top of each other has converged and can be frozen; one
where they are still spread either does not matter or has not been resolved
yet -- and telling those two apart by staring at a CSV is hopeless.
"""

import csv
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402  (must follow the Agg switch)

from swarm.report import _HTML_HEAD
from utils.log import get_logger

logger = get_logger(__name__)

_FG = '#e8e8ea'
_MUTED = '#8a8a92'
_PANEL = '#1a1a1e'
_FIG = '#121214'
_ACCENT = '#7fd1ff'
_GOOD = '#8fe39a'
_WARN = '#f0c674'


# ==================== history ====================

def write_history_row(path, evaluation, param_paths: Sequence[str]) -> None:
    """
    Append one evaluation to the run-long CSV, writing the header on first use.

    The column set is fixed from the first row: parameters, then sub-scores,
    then every metric. A stable header is what lets the file be opened in a
    spreadsheet mid-run instead of only after the search has finished.
    """
    path = Path(path)
    metric_keys = sorted(evaluation.metrics)
    part_keys = sorted(evaluation.parts)
    fieldnames = (['index', 'generation', 'score', 'wall_time']
                  + list(param_paths)
                  + [f'score_{k}' for k in part_keys]
                  + list(metric_keys))

    row = {'index': evaluation.index, 'generation': evaluation.generation,
           'score': f'{evaluation.score:.6f}', 'wall_time': f'{evaluation.wall_time:.2f}'}
    row.update({p: evaluation.overrides.get(p, '') for p in param_paths})
    row.update({f'score_{k}': f'{evaluation.parts[k]:.4f}' for k in part_keys})
    row.update({k: f'{evaluation.metrics[k]:.6g}' for k in metric_keys})

    exists = path.exists()
    with open(path, 'a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore', restval='')
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# ==================== plots ====================

def _style(ax):
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.grid(alpha=0.12, color='#ffffff')
    for spine in ax.spines.values():
        spine.set_color('#33333a')


def write_progress(path, tuner) -> Path:
    """The three-panel state of the search."""
    history = tuner.history
    if not history:
        return Path(path)

    fig = plt.figure(figsize=(12.5, 7.4), facecolor=_FIG)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.32, wspace=0.22)

    best = tuner.best
    fig.suptitle(f'swarm tuning  |  space "{tuner.space.name}"  |  '
                 f'{len(history)} evaluations  |  best {best.score:.4f} '
                 f'(target {tuner.objective.target_score:.2f})',
                 color=_FG, fontsize=11)

    # --- 1. the search, evaluation by evaluation ---
    ax = fig.add_subplot(grid[0, :])
    _style(ax)
    xs = [e.index for e in history]
    ys = [e.score for e in history]
    generations = [e.generation for e in history]

    scatter = ax.scatter(xs, ys, c=generations, cmap='viridis', s=16, alpha=0.85, linewidths=0)
    running = np.maximum.accumulate(ys)
    ax.plot(xs, running, color=_GOOD, lw=1.6, label='best so far')
    ax.axhline(tuner.objective.target_score, color=_WARN, ls='--', lw=1.0,
               label=f'target {tuner.objective.target_score:.2f}')
    ax.set_title('score per evaluation (colour = generation)', color=_FG, fontsize=9)
    ax.set_xlabel('evaluation', color=_MUTED, fontsize=8)
    ax.set_ylabel('objective score', color=_MUTED, fontsize=8)
    ax.legend(fontsize=7, facecolor=_PANEL, edgecolor='#33333a', labelcolor='#c8c8cc', loc='lower right')
    bar = fig.colorbar(scatter, ax=ax, pad=0.01)
    bar.ax.tick_params(colors=_MUTED, labelsize=7)

    # --- 2. the current best, criterion by criterion ---
    ax = fig.add_subplot(grid[1, 0])
    _style(ax)
    criteria = tuner.objective.criteria
    names = [c.key for c in criteria]
    scores = [best.parts.get(c.key, 0.0) for c in criteria]
    weights = [c.weight for c in criteria]
    y = np.arange(len(names))

    ax.barh(y, scores, color=[_GOOD if s > 0.66 else (_WARN if s > 0.33 else '#f08a7a')
                              for s in scores], height=0.62)
    ax.barh(y, [1.0] * len(names), color='#26262c', height=0.62, zorder=0)
    for i, (name, score) in enumerate(zip(names, scores)):
        value = best.metrics.get(name, float('nan'))
        ax.text(1.02, i, f'{value:.4g}', color=_MUTED, fontsize=7, va='center')
    ax.set_yticks(y)
    ax.set_yticklabels([f'{n}  x{w:.2f}' for n, w in zip(names, weights)],
                       color='#c8c8cc', fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.25)
    ax.set_title('best run: score per criterion (right column = measured value)',
                 color=_FG, fontsize=9)

    # --- 3. have the parameters decided? ---
    ax = fig.add_subplot(grid[1, 1])
    _style(ax)
    paths = tuner.space.paths
    x = np.arange(len(paths))
    for rank, elite in enumerate(tuner.elites):
        alpha = 0.95 if rank == 0 else 0.35
        ax.plot(x, elite.unit, color=_ACCENT if rank == 0 else '#6b6b78',
                lw=1.8 if rank == 0 else 1.0, alpha=alpha, marker='o', ms=3)
    ax.set_xticks(x)
    ax.set_xticklabels(paths, rotation=55, ha='right', color='#c8c8cc', fontsize=7)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title('elites in normalised parameter space (blue = best)', color=_FG, fontsize=9)
    ax.set_ylabel('position in range', color=_MUTED, fontsize=8)

    fig.savefig(path, dpi=105, facecolor=_FIG, bbox_inches='tight')
    plt.close(fig)
    return Path(path)


# ==================== the page ====================

def _scorecard_table(tuner) -> str:
    best = tuner.best
    rows = ['<table><thead><tr>'
            '<th>criterion</th><th>measured</th><th>score</th><th>weight</th>'
            '<th>contribution</th><th style="text-align:left">what it asks</th>'
            '</tr></thead><tbody>']

    for criterion in sorted(tuner.objective.criteria, key=lambda c: -c.weight):
        score = best.parts.get(criterion.key, 0.0)
        value = best.metrics.get(criterion.key, float('nan'))
        cls = 'good' if score > 0.66 else ('warn' if score > 0.33 else 'bad')
        band = (f'{criterion.mode} {criterion.lo:g}..{criterion.hi:g}'
                + (f' ±{criterion.soft:g}' if criterion.mode == 'band' else ''))
        rows.append(
            f'<tr><td>{html.escape(criterion.key)}<br>'
            f'<span class="tag">{html.escape(band)}</span></td>'
            f'<td>{value:.4g}</td>'
            f'<td class="{cls}">{score:.2f}</td>'
            f'<td>{criterion.weight:.2f}</td>'
            f'<td>{score * criterion.weight:.3f}</td>'
            f'<td style="text-align:left;white-space:normal">{html.escape(criterion.doc)}</td></tr>')

    rows.append('</tbody></table>')
    return '\n'.join(rows)


def _overrides_block(overrides: Dict[str, Any]) -> str:
    items = ' '.join(f'<code>{html.escape(k)}={v:.5g}</code>' if isinstance(v, (int, float))
                     else f'<code>{html.escape(k)}={html.escape(str(v))}</code>'
                     for k, v in sorted(overrides.items()))
    return items or '<span class="tag">defaults</span>'


def _leaderboard(tuner, limit: int = 20) -> str:
    criteria = [c.key for c in tuner.objective.criteria]
    header = ['rank', 'run', 'score'] + criteria
    rows = ['<table><thead><tr>' + ''.join(f'<th>{html.escape(c)}</th>' for c in header)
            + '</tr></thead><tbody>']

    ranked = sorted(tuner.history, key=lambda e: -e.score)[:limit]
    for i, evaluation in enumerate(ranked, start=1):
        cells = [f'<td>{i}</td>', f'<td>{html.escape(evaluation.label)}</td>',
                 f'<td class="good">{evaluation.score:.4f}</td>']
        for key in criteria:
            score = evaluation.parts.get(key, 0.0)
            cls = 'good' if score > 0.66 else ('warn' if score > 0.33 else 'bad')
            cells.append(f'<td class="{cls}">{score:.2f}</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')

    rows.append('</tbody></table>')
    return '\n'.join(rows)


def _generation_cards(out_dir: Path) -> str:
    """One card per filmed generation, newest first."""
    folder = out_dir / 'generations'
    if not folder.exists():
        return ''

    cards = ['<div class="cards">']
    for run in sorted(folder.glob('gen*'), reverse=True):
        video = run / 'evolution.mp4'
        canvas = run / 'canvas.png'
        doc = ''
        config_file = run / 'config.json'
        if config_file.exists():
            try:
                doc = json.loads(config_file.read_text()).get('doc', '')
            except (ValueError, OSError):
                doc = ''

        media = (f'<video src="generations/{run.name}/evolution.mp4" controls loop muted '
                 f'style="width:100%;border-radius:6px"></video>' if video.exists()
                 else (f'<img src="generations/{run.name}/canvas.png">' if canvas.exists() else ''))

        cards.append(
            f'<div class="card"><h3>{html.escape(run.name)}</h3>'
            f'<p>{html.escape(doc)}</p>{media}'
            f'<p style="margin-top:10px">'
            f'<a href="generations/{run.name}/filmstrip.png">filmstrip</a> · '
            f'<a href="generations/{run.name}/diagnostics.png">diagnostics</a> · '
            f'<a href="generations/{run.name}/config.json">config</a></p></div>')

    cards.append('</div>')
    return '\n'.join(cards)


def write_tuning_html(path, tuner, latest_video: Optional[Path] = None,
                      final_section: str = '') -> Path:
    """The live report. Rewritten in full after every generation."""
    best = tuner.best
    if best is None:
        return Path(path)

    subtitle = (f'Space <code>{html.escape(tuner.space.name)}</code> · '
                f'{len(tuner.history)} evaluations · generation {tuner.generation} · '
                f'step size {tuner.sigma:.3f} · '
                f'best score <strong>{best.score:.4f}</strong> of a target '
                f'{tuner.objective.target_score:.2f}')

    parts = [_HTML_HEAD.format(title=html.escape(f'swarm tuning — {tuner.space.name}'),
                               subtitle=subtitle)]

    parts.append('<h2>Search progress</h2>'
                 '<div class="sheet"><img src="progress.png" alt="progress"></div>')

    if (Path(tuner.out_dir) / 'progress.mp4').exists():
        parts.append('<h2>The search as a film</h2>'
                     '<p class="sub">One held shot per generation: what the optimiser '
                     'traded away as the number went up.</p>'
                     '<video src="progress.mp4" controls loop muted '
                     'style="max-width:520px;border-radius:6px"></video>')

    parts.append('<h2>Best configuration</h2>')
    parts.append(f'<p class="sub">{_overrides_block(best.overrides)}</p>')
    parts.append('<p class="sub">Run it with '
                 '<code>python -m swarm.lab --mode preset --config best.json</code>, '
                 'or install it as the <code>tuned</code> preset with '
                 '<code>python -m swarm.tuning --adopt</code>.</p>')

    parts.append('<h2>Scorecard</h2>')
    parts.append('<p class="sub">Every criterion, what it measured, and how much of the '
                 'composite it carries. Definitions and bands come from '
                 '<code>config/tuning_config.py</code>.</p>')
    parts.append(_scorecard_table(tuner))

    if final_section:
        parts.append(final_section)

    generations = _generation_cards(Path(tuner.out_dir))
    if generations:
        parts.append('<h2>Films of the best, by generation</h2>')
        parts.append('<p class="sub">Shot every '
                     f'{tuner.tuning.animate_every} generations, over '
                     f'{tuner.tuning.animation_steps} simulation steps: '
                     'stage-2 input, the swarm, the target.</p>')
        parts.append(generations)

    parts.append('<h2>Leaderboard</h2>')
    parts.append(_leaderboard(tuner))

    Path(path).write_text('\n'.join(parts), encoding='utf-8')
    return Path(path)


def write_final_report(path, tuner, result, confirmations: List[Dict[str, float]]) -> Path:
    """The page once the search has stopped, with the long confirmation run on it."""
    rows = ['<table><thead><tr><th>seed</th><th>score</th><th>improvement</th>'
            '</tr></thead><tbody>']
    for row in confirmations:
        rows.append(f'<tr><td>{row["seed"]}</td><td>{row["score"]:.4f}</td>'
                    f'<td>{row["improvement"] * 100:+.2f}%</td></tr>')
    rows.append('</tbody></table>')

    section = [
        '<h2>The winner, confirmed</h2>',
        f'<p class="sub">Re-run over {tuner.tuning.final_steps} steps on '
        f'{len(confirmations)} seeds. A score that survives a change of seed is a '
        'property of the configuration; one that does not was luck.</p>',
        '\n'.join(rows),
        '<video src="final/evolution.mp4" controls loop muted '
        'style="max-width:100%;border-radius:6px;margin-top:16px"></video>',
        '<div class="sheet" style="margin-top:16px">'
        '<img src="final/filmstrip.png" alt="filmstrip"></div>',
        '<div class="sheet"><img src="final/diagnostics.png" alt="diagnostics"></div>',
    ]

    out = write_tuning_html(path, tuner, final_section='\n'.join(section))
    logger.info('Final report: %s', out)
    return out
