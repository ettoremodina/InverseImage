"""
Artefacts that make swarm runs comparable (Evolutionary_Swarm.md §11.2, §12).

The old contact sheet packed labels into a 24px strip and let them overrun each
other, and carried no reference for what the swarm started from -- so three
runs that all did nothing looked like three successful runs. Everything here is
built around the opposite principle: a sheet is only useful if it shows the
**baseline next to the result** and prints the number that says which is better.

Matplotlib is used head-less (Agg) for the curve plots; the sheets are plain
OpenCV compositing, because they need pixel-exact thumbnails and no resampling
surprises.
"""

import csv
import html
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402  (must follow the Agg switch)

from swarm.metrics import SUMMARY_FIELDS
from utils.log import get_logger

logger = get_logger(__name__)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_BG = (18, 18, 20)          # BGR, matches the dark plots
_FG = (235, 235, 235)
_MUTED = (150, 150, 155)
_GOOD = (120, 220, 140)
_BAD = (120, 120, 245)


# ==================== primitives ====================

def save_rgb(path, rgb: np.ndarray) -> None:
    """Write an (H, W, 3) uint8 RGB array as a PNG."""
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _thumb(rgb: np.ndarray, size: int) -> np.ndarray:
    """BGR thumbnail. Nearest-neighbour upscaling keeps single-pixel strokes visible."""
    interp = cv2.INTER_AREA if rgb.shape[0] > size else cv2.INTER_NEAREST
    return cv2.cvtColor(cv2.resize(rgb, (size, size), interpolation=interp), cv2.COLOR_RGB2BGR)


def _text(canvas: np.ndarray, text: str, org, scale: float = 0.42,
          color=_FG, thickness: int = 1) -> None:
    cv2.putText(canvas, text, org, _FONT, scale, color, thickness, cv2.LINE_AA)


def _fit_text(text: str, width_px: int, scale: float = 0.42) -> str:
    """Truncate with an ellipsis so labels can never overrun their cell."""
    max_chars = max(4, int(width_px / (scale * 19.0)))
    return text if len(text) <= max_chars else text[:max_chars - 1] + '…'


def _verdict_color(improvement: float):
    return _GOOD if improvement > 0 else _BAD


# ==================== per-run artefacts ====================

def save_filmstrip(path, result, cell: int = 192) -> None:
    """
    `base -> captured steps -> final -> target` in one row.

    The base and target panels are the whole point: they turn "here is a
    picture" into "here is what changed and how far it had to go".
    """
    panels = [('base (stage 2)', result.base)]
    panels += [(f'step {step}', frame) for step, frame in result.frames[1:]]
    panels.append(('target', result.target))

    label_h = 22
    sheet = np.full((cell + label_h, cell * len(panels), 3), _BG, dtype=np.uint8)

    for i, (name, frame) in enumerate(panels):
        x = i * cell
        sheet[label_h:, x:x + cell] = _thumb(frame, cell)
        _text(sheet, _fit_text(name, cell), (x + 5, 15), color=_MUTED)
        cv2.line(sheet, (x, label_h), (x, sheet.shape[0]), _BG, 1)

    cv2.imwrite(str(path), sheet)


def save_diagnostics(path, result) -> None:
    """
    The four §12 curves: error, population, diversity, and the energy economy.

    The baseline error is drawn as a dashed line on the error panel. A run whose
    curve sits above that line made the picture worse than the NCA output it
    was handed, and no other reading matters until that is fixed.
    """
    history = result.history
    steps = [m.step for m in history]

    doc = result.doc or 'swarm run'
    if len(doc) > 96:
        doc = doc[:95] + '…'

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), facecolor='#121214')
    fig.suptitle(f'{result.label}   |   {doc}', color='#e8e8e8', fontsize=10)

    for ax in axes.flat:
        ax.set_facecolor('#1a1a1e')
        ax.tick_params(colors='#9a9aa0', labelsize=8)
        ax.grid(alpha=0.12, color='#ffffff')
        for spine in ax.spines.values():
            spine.set_color('#33333a')

    err = axes[0, 0]
    err.plot(steps, [m.mean_error for m in history], color='#7fd1ff', lw=1.4, label='canvas')
    err.axhline(result.summary.baseline_error, color='#ff8a6b', ls='--', lw=1.1,
                label=f'stage-2 baseline {result.summary.baseline_error:.4f}')
    err.set_title('mean error vs target', color='#e8e8e8', fontsize=9)
    err.legend(fontsize=7, facecolor='#1a1a1e', edgecolor='#33333a', labelcolor='#c8c8cc')

    pop = axes[0, 1]
    pop.plot(steps, [m.population for m in history], color='#9ee493', lw=1.4, label='alive')
    pop.axhline(result.config.min_population_fraction * result.config.population_cap,
                color='#ff8a6b', ls='--', lw=1.1, label='respawn floor')
    pop.set_title('population', color='#e8e8e8', fontsize=9)
    pop.legend(fontsize=7, facecolor='#1a1a1e', edgecolor='#33333a', labelcolor='#c8c8cc')

    div = axes[1, 0]
    div.plot(steps, [m.gene_diversity for m in history], color='#d7a1ff', lw=1.4)
    div.set_title('gene diversity (summed variance)', color='#e8e8e8', fontsize=9)
    div.set_xlabel('step', color='#9a9aa0', fontsize=8)

    eco = axes[1, 1]
    eco.plot(steps, [m.mean_energy for m in history], color='#ffd479', lw=1.4, label='mean energy')
    eco.plot(steps, [m.positive_gain_fraction for m in history], color='#7fd1ff', lw=1.1,
             label='share eating')
    eco.axhline(0.0, color='#66666e', lw=0.8)
    eco.set_title('the economy', color='#e8e8e8', fontsize=9)
    eco.set_xlabel('step', color='#9a9aa0', fontsize=8)
    eco.legend(fontsize=7, facecolor='#1a1a1e', edgecolor='#33333a', labelcolor='#c8c8cc')

    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


# ==================== cross-run artefacts ====================

def save_contact_sheet(path, results: Sequence, cell: int = 192, columns: int = None,
                       title: str = '') -> None:
    """
    A grid of final canvases, each captioned with the number that ranks it.

    The first cell is always the stage-2 baseline, so every comparison on the
    sheet is against the thing the swarm has to beat rather than against the
    neighbouring cell.
    """
    if not results:
        raise ValueError('nothing to put on a contact sheet')

    panels = [('stage-2 baseline', results[0].base, None),
              ('target', results[0].target, None)]
    panels += [(r.label, r.canvas, r.summary) for r in results]

    columns = columns or min(6, max(3, int(np.ceil(np.sqrt(len(panels))))))
    rows = int(np.ceil(len(panels) / columns))

    caption_h = 40
    title_h = 30 if title else 0
    cell_w, cell_h = cell, cell + caption_h

    sheet = np.full((title_h + rows * cell_h, columns * cell_w, 3), _BG, dtype=np.uint8)
    if title:
        _text(sheet, _fit_text(title, columns * cell_w, 0.5), (8, 20), scale=0.5)

    for i, (label, image, summary) in enumerate(panels):
        r, c = divmod(i, columns)
        x, y = c * cell_w, title_h + r * cell_h
        sheet[y:y + cell, x:x + cell] = _thumb(image, cell)

        _text(sheet, _fit_text(label, cell_w), (x + 5, y + cell + 14))
        if summary is not None:
            verdict = f'{summary.improvement * 100:+.1f}% vs baseline'
            _text(sheet, _fit_text(verdict, cell_w), (x + 5, y + cell + 27),
                  scale=0.38, color=_verdict_color(summary.improvement))
            state = 'FLOOR-PINNED' if summary.extinct else f'pop {summary.final_population}'
            _text(sheet, _fit_text(f'{state}  births {summary.total_births}', cell_w),
                  (x + 5, y + cell + 37), scale=0.34, color=_MUTED)
        else:
            _text(sheet, 'reference', (x + 5, y + cell + 27), scale=0.38, color=_MUTED)

    cv2.imwrite(str(path), sheet)


def save_grid_sheet(path, results: Sequence, x_param: str, y_param: str,
                    cell: int = 160, title: str = '') -> None:
    """
    A two-axis study laid out as an actual matrix, not a strip.

    A 4x4 grid read as a 16-wide row tells you nothing about interaction between
    the two parameters; read as a matrix, the live region is visible at a glance.
    """
    def order(value):
        """Numeric axes sort numerically; categorical ones fall back to text."""
        try:
            return (0, float(value), '')
        except (TypeError, ValueError):
            return (1, 0.0, str(value))

    xs = sorted({r.overrides[x_param] for r in results}, key=order)
    ys = sorted({r.overrides[y_param] for r in results}, key=order)
    by_cell = {(r.overrides[x_param], r.overrides[y_param]): r for r in results}

    margin_l, margin_t = 116, 52 if title else 30
    caption_h = 26
    sheet = np.full((margin_t + len(ys) * (cell + caption_h), margin_l + len(xs) * cell, 3),
                    _BG, dtype=np.uint8)

    if title:
        _text(sheet, title, (8, 20), scale=0.5)
    _text(sheet, f'x: {x_param}', (8, margin_t - 16), scale=0.4, color=_MUTED)

    for cx, xv in enumerate(xs):
        _text(sheet, _fit_text(f'{xv:g}' if isinstance(xv, (int, float)) else str(xv), cell),
              (margin_l + cx * cell + 4, margin_t - 6), scale=0.42)

    for cy, yv in enumerate(ys):
        y = margin_t + cy * (cell + caption_h)
        _text(sheet, f'{y_param}', (6, y + 16), scale=0.34, color=_MUTED)
        _text(sheet, f'{yv:g}' if isinstance(yv, (int, float)) else str(yv),
              (6, y + 32), scale=0.44)

        for cx, xv in enumerate(xs):
            result = by_cell.get((xv, yv))
            if result is None:
                continue
            x = margin_l + cx * cell
            sheet[y:y + cell, x:x + cell] = _thumb(result.canvas, cell)

            s = result.summary
            _text(sheet, f'{s.improvement * 100:+.1f}%', (x + 4, y + cell + 13),
                  scale=0.42, color=_verdict_color(s.improvement))
            state = 'floor' if s.extinct else f'pop {s.final_population}'
            _text(sheet, _fit_text(f'{state}  b{s.total_births}', cell), (x + 4, y + cell + 23),
                  scale=0.34, color=_MUTED)

    cv2.imwrite(str(path), sheet)


def write_summary_csv(path, results: Sequence) -> None:
    """One row per run: the overrides that defined it, then every summary field."""
    override_keys = sorted({k for r in results for k in r.overrides})
    fieldnames = ['name', 'label'] + override_keys + list(SUMMARY_FIELDS)

    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {'name': r.name, 'label': r.label}
            row.update({k: r.overrides.get(k, '') for k in override_keys})
            row.update(asdict(r.summary))
            writer.writerow(row)


# ==================== the showcase ====================

_HTML_HEAD = """<!doctype html>
<meta charset="utf-8">
<title>{title}</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ background:#121214; color:#e8e8ea; font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;
         margin:0; padding:32px 40px 64px; }}
  h1 {{ font-size:21px; margin:0 0 4px; }}
  h2 {{ font-size:15px; margin:36px 0 10px; color:#c8c8cc; font-weight:600; }}
  .sub {{ color:#8a8a92; margin:0 0 24px; max-width:74ch; }}
  img {{ max-width:100%; display:block; border-radius:6px; background:#1a1a1e; }}
  .sheet {{ overflow-x:auto; margin-bottom:28px; }}
  table {{ border-collapse:collapse; font-size:12.5px; width:100%; }}
  th, td {{ text-align:right; padding:5px 9px; border-bottom:1px solid #26262c; white-space:nowrap; }}
  th {{ color:#8a8a92; font-weight:600; text-align:right; position:sticky; top:0; background:#121214; }}
  td:first-child, th:first-child {{ text-align:left; }}
  tr:hover td {{ background:#18181d; }}
  .good {{ color:#8fe39a; }} .bad {{ color:#f08a7a; }} .warn {{ color:#f0c674; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:20px; }}
  .card {{ background:#17171b; border:1px solid #26262c; border-radius:8px; padding:14px; }}
  .card h3 {{ margin:0 0 2px; font-size:14px; }}
  .card p {{ margin:0 0 10px; color:#8a8a92; font-size:12.5px; }}
  code {{ background:#1e1e24; padding:1px 5px; border-radius:4px; font-size:12px; color:#b9c8e0; }}
  .tag {{ font-size:11px; padding:2px 7px; border-radius:10px; background:#26262c; color:#a8a8b0; }}
</style>
<h1>{title}</h1>
<p class="sub">{subtitle}</p>
"""


def _fmt(value, digits: int = 4) -> str:
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return html.escape(str(value))


def _summary_table(results: Sequence, override_keys: List[str]) -> str:
    columns = ['run'] + override_keys + ['improvement', 'final error', 'baseline',
                                         'pop', 'floor %', 'births', 'diversity', 'steps/s']
    out = ['<table><thead><tr>' + ''.join(f'<th>{html.escape(c)}</th>' for c in columns)
           + '</tr></thead><tbody>']

    for r in sorted(results, key=lambda x: -x.summary.improvement):
        s = r.summary
        cls = 'good' if s.improvement > 0 else 'bad'
        floor_cls = 'bad' if s.extinct else ('warn' if s.floor_fraction > 0.25 else '')
        cells = [f'<td><a href="{html.escape(r.name)}/filmstrip.png">{html.escape(r.label)}</a></td>']
        cells += [f'<td>{_fmt(r.overrides.get(k, ""))}</td>' for k in override_keys]
        cells += [
            f'<td class="{cls}">{s.improvement * 100:+.2f}%</td>',
            f'<td>{s.final_error:.5f}</td>',
            f'<td>{s.baseline_error:.5f}</td>',
            f'<td>{s.final_population}</td>',
            f'<td class="{floor_cls}">{s.floor_fraction * 100:.0f}%</td>',
            f'<td>{s.total_births}</td>',
            f'<td>{s.gene_diversity:.4f}</td>',
            f'<td>{s.steps_per_sec:.0f}</td>',
        ]
        out.append('<tr>' + ''.join(cells) + '</tr>')

    out.append('</tbody></table>')
    return '\n'.join(out)


def write_showcase_html(path, title: str, subtitle: str, results: Sequence,
                        sheets: Dict[str, str]) -> None:
    """
    The comparison page: sheets first, then the ranked table, then one card per
    run linking to its own artefacts.

    Self-contained apart from the PNGs it sits next to, so the whole folder can
    be zipped or committed and still read correctly.
    """
    override_keys = sorted({k for r in results for k in r.overrides})
    parts = [_HTML_HEAD.format(title=html.escape(title), subtitle=html.escape(subtitle))]

    for caption, filename in sheets.items():
        parts.append(f'<h2>{html.escape(caption)}</h2>'
                     f'<div class="sheet"><img src="{html.escape(filename)}" alt="{html.escape(caption)}"></div>')

    parts.append('<h2>Ranked results</h2>')
    parts.append('<p class="sub">Sorted by improvement over the stage-2 baseline. '
                 '<code>floor %</code> is the share of the run spent pinned at the respawn '
                 'floor &mdash; a high value means the population is being propped up rather '
                 'than feeding itself.</p>')
    parts.append(_summary_table(results, override_keys))

    parts.append('<h2>Runs</h2><div class="cards">')
    for r in sorted(results, key=lambda x: -x.summary.improvement):
        overrides = ' '.join(f'<code>{html.escape(k)}={_fmt(v)}</code>'
                             for k, v in sorted(r.overrides.items())) or '<span class="tag">defaults</span>'
        parts.append(
            f'<div class="card"><h3>{html.escape(r.label)}</h3>'
            f'<p>{html.escape(r.doc)}</p>'
            f'<p>{overrides}</p>'
            f'<a href="{html.escape(r.name)}/filmstrip.png"><img src="{html.escape(r.name)}/filmstrip.png"></a>'
            f'<p style="margin-top:10px"><a href="{html.escape(r.name)}/diagnostics.png">diagnostics</a> · '
            f'<a href="{html.escape(r.name)}/metrics.csv">metrics.csv</a> · '
            f'<a href="{html.escape(r.name)}/config.json">config.json</a></p></div>')
    parts.append('</div>')

    Path(path).write_text('\n'.join(parts), encoding='utf-8')
    logger.info('Showcase written: %s', path)


def write_gallery_html(path, entries: List[Dict[str, str]]) -> None:
    """Top-level index over every study folder under `outputs/swarm/`."""
    parts = [_HTML_HEAD.format(
        title='Swarm laboratory',
        subtitle='Every calibration study, newest first. Each links to its own showcase page.')]
    parts.append('<div class="cards">')
    for entry in entries:
        parts.append(
            f'<div class="card"><h3><a href="{html.escape(entry["href"])}">'
            f'{html.escape(entry["title"])}</a></h3>'
            f'<p>{html.escape(entry.get("doc", ""))}</p>'
            f'<p><span class="tag">{html.escape(entry.get("runs", ""))} runs</span> '
            f'<span class="tag">{html.escape(entry.get("best", ""))}</span></p></div>')
    parts.append('</div>')
    Path(path).write_text('\n'.join(parts), encoding='utf-8')
    logger.info('Gallery written: %s', path)
