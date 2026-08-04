"""
Easing curves for the timeline (PLAN 3.6).

Config stores the name, the timeline resolves it here. Every curve maps
[0, 1] -> [0, 1] and is monotonic, so a stage never runs backwards.
"""

from typing import Callable, Dict

import numpy as np


def linear(t: float) -> float:
    return t


def ease_in(t: float) -> float:
    return t * t


def ease_out(t: float) -> float:
    """Fast start, slow finish -- what tree growth actually looks like."""
    return 1.0 - (1.0 - t) ** 2


def ease_in_out(t: float) -> float:
    return 3.0 * t * t - 2.0 * t * t * t  # smoothstep


def ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


CURVES: Dict[str, Callable[[float], float]] = {
    'linear': linear,
    'ease_in': ease_in,
    'ease_out': ease_out,
    'ease_in_out': ease_in_out,
    'ease_out_cubic': ease_out_cubic,
}


def get_easing(name: str) -> Callable[[float], float]:
    """Look up a curve by name; unknown names fall back to linear with a warning."""
    curve = CURVES.get(name)
    if curve is None:
        from utils.log import get_logger
        get_logger(__name__).warning("Unknown easing '%s', falling back to linear", name)
        return linear
    return curve


def apply_easing(name: str, t: float) -> float:
    return float(get_easing(name)(float(np.clip(t, 0.0, 1.0))))
