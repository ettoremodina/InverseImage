"""
Video of a swarm run -- the part of the verdict that no metric replaces.

A score says a configuration is better; it does not say the strokes look like
strokes, or that the picture arrives instead of thrashing. The tuner therefore
films its current best every few generations, and this module is what turns a
running `Simulation` into something to watch:

- `VideoRecorder` streams frames while a run happens, so a 2400-step film costs
  no more memory than one frame. It optionally lays the canvas between the
  stage-2 input and the target, which is the only framing in which "did this
  help" is answerable by eye.
- `save_generation_reel` films the *tuning* instead of the swarm: one frame per
  generation's best canvas, so a 40-generation search plays back in a few
  seconds and the moment the search found the good region is visible.

Encoder settings are duplicated from `rendering/utils.py` rather than imported
on purpose: that module imports pycairo at load time, and nothing in the
headless tuning path should need a vector graphics library to write an mp4.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import imageio
import numpy as np

from utils.log import get_logger

logger = get_logger(__name__)

VIDEO_CODEC = 'libx264'
VIDEO_QUALITY = 8
VIDEO_PIXELFORMAT = 'yuv420p'

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_BG = (18, 18, 20)
_FG = (235, 235, 235)
_MUTED = (150, 150, 155)
_GAP = 4


def _open_writer(path, fps: int):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(str(path), fps=fps, codec=VIDEO_CODEC,
                              quality=VIDEO_QUALITY, pixelformat=VIDEO_PIXELFORMAT,
                              macro_block_size=1)


def _even(value: int) -> int:
    """h264 with yuv420p needs even dimensions; rounding up beats a re-encode."""
    return value + (value % 2)


class VideoRecorder:
    """
    Streaming writer for one simulation.

    `add(sim)` composes and writes a frame; `close()` finalises the file. The
    HUD carries the two numbers that decide whether the run is worth watching
    to the end -- improvement on the subject, and whether anybody is alive.
    """

    def __init__(self, path, fps: int = 30, scale: int = 2, panels: bool = True,
                 hud: bool = True, title: str = ''):
        self.path = Path(path)
        self.scale = max(1, int(scale))
        self.panels = panels
        self.hud = hud
        self.title = title
        self.writer = _open_writer(path, fps)
        self.frames = 0

        self._base: Optional[np.ndarray] = None
        self._target: Optional[np.ndarray] = None

    # ------------------------------------------------------------ frame building

    def _references(self, sim):
        if self._base is None:
            self._base = sim.render_base_srgb().cpu().numpy()
            self._target = sim.render_target_srgb().cpu().numpy()
        return self._base, self._target

    @staticmethod
    def _improvement(sim) -> float:
        """Error reduction on the tissue -- the same masked figure the tuner scores."""
        mask = sim.fields.nutrient > sim.config.nutrient_threshold
        total = mask.sum()
        if total <= 0:
            return 0.0
        base_err = (sim.fields.base - sim.fields.target).norm(dim=-1)[mask].mean()
        now_err = (sim.fields.canvas - sim.fields.target).norm(dim=-1)[mask].mean()
        return float(1.0 - (now_err / base_err.clamp(min=1e-9)).item())

    def _compose(self, sim) -> np.ndarray:
        canvas = sim.render_canvas_srgb().cpu().numpy()

        if self.panels:
            base, target = self._references(sim)
            h = canvas.shape[0]
            gap = np.full((h, _GAP, 3), _BG[::-1], dtype=np.uint8)
            frame = np.concatenate([base, gap, canvas, gap, target], axis=1)
        else:
            frame = canvas

        if self.scale > 1:
            frame = cv2.resize(frame, (frame.shape[1] * self.scale, frame.shape[0] * self.scale),
                               interpolation=cv2.INTER_NEAREST)

        if self.hud:
            frame = self._add_hud(frame, sim)

        h, w = frame.shape[:2]
        if h % 2 or w % 2:
            frame = cv2.copyMakeBorder(frame, 0, _even(h) - h, 0, _even(w) - w,
                                       cv2.BORDER_CONSTANT, value=_BG[::-1])
        return frame

    def _add_hud(self, frame: np.ndarray, sim) -> np.ndarray:
        bar_h = 30 if not self.title else 46
        bar = np.full((bar_h, frame.shape[1], 3), _BG[::-1], dtype=np.uint8)

        population = int(sim.agents.alive.sum().item())
        line = (f'step {sim.step_count}   '
                f'{self._improvement(sim) * 100:+.2f}% vs stage 2   '
                f'pop {population}')

        y = 20
        if self.title:
            cv2.putText(bar, self.title, (10, y), _FONT, 0.5, _FG[::-1], 1, cv2.LINE_AA)
            y += 18
        cv2.putText(bar, line, (10, y), _FONT, 0.45, _MUTED[::-1], 1, cv2.LINE_AA)

        out = np.concatenate([bar, frame], axis=0)
        if self.panels:
            labels = ('stage 2 input', 'swarm', 'target')
            width = (frame.shape[1] - 2 * _GAP * self.scale) // 3
            for i, label in enumerate(labels):
                x = 8 + i * (width + _GAP * self.scale)
                cv2.putText(out, label, (x, bar_h + 16), _FONT, 0.4, _MUTED[::-1], 1, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------ public

    def add(self, sim) -> None:
        self.writer.append_data(self._compose(sim))
        self.frames += 1

    def close(self) -> Path:
        self.writer.close()
        logger.info('Animation: %s (%d frames)', self.path, self.frames)
        return self.path


def save_generation_reel(path, entries: Sequence[Tuple[str, np.ndarray]], fps: int = 8,
                         hold: int = 4, scale: int = 2, title: str = 'tuning progress') -> Path:
    """
    Film the search itself: one held shot per generation's best canvas.

    `entries` is `[(caption, rgb), ...]` in generation order. Played back at a
    few frames a second this answers a question the score curve cannot -- what
    the optimiser actually traded away when the number went up.
    """
    entries = list(entries)
    if not entries:
        raise ValueError('no generations to film')

    writer = _open_writer(path, fps)
    height, width = entries[0][1].shape[:2]

    for caption, rgb in entries:
        frame = cv2.resize(rgb, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)
        bar = np.full((34, frame.shape[1], 3), _BG[::-1], dtype=np.uint8)
        cv2.putText(bar, title, (10, 14), _FONT, 0.4, _MUTED[::-1], 1, cv2.LINE_AA)
        cv2.putText(bar, caption, (10, 29), _FONT, 0.45, _FG[::-1], 1, cv2.LINE_AA)

        composed = np.concatenate([bar, frame], axis=0)
        h, w = composed.shape[:2]
        if h % 2 or w % 2:
            composed = cv2.copyMakeBorder(composed, 0, _even(h) - h, 0, _even(w) - w,
                                          cv2.BORDER_CONSTANT, value=_BG[::-1])
        for _ in range(max(1, hold)):
            writer.append_data(composed)

    writer.close()
    logger.info('Progress reel: %s (%d generations)', path, len(entries))
    return Path(path)


def frames_to_video(path, frames: List[np.ndarray], fps: int = 30, scale: int = 1) -> Path:
    """Write a list of RGB frames as an mp4. For callers that already have them."""
    writer = _open_writer(path, fps)
    for frame in frames:
        if scale > 1:
            frame = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale),
                               interpolation=cv2.INTER_NEAREST)
        h, w = frame.shape[:2]
        if h % 2 or w % 2:
            frame = cv2.copyMakeBorder(frame, 0, _even(h) - h, 0, _even(w) - w,
                                       cv2.BORDER_CONSTANT, value=_BG[::-1])
        writer.append_data(frame)
    writer.close()
    return Path(path)
