"""
Subtree weight of the SCA branches (PLAN f, technique 1).

Today a branch is thick because it is close to the root. Two branches at the
same depth get the same width even when one holds half the crown and the other
a dead twig -- which is exactly what makes the tree read as a technical diagram
instead of as a plant.

The rule used instead is the one Leonardo observed and Murray later formalised:
a branch that splits conserves the cross-section of what it carries,

    w_parent^2 = sum(w_child^2)

so a branch is thick in proportion to *how much tree it holds up*.

Two entry points:

- `murray_weights` -- exact, from the hierarchy, used at export time where the
  parent pointers are still available;
- `infer_weights_from_polylines` -- reconstructs the hierarchy from the geometry
  of an already exported file, so the feature also works on render data written
  before this existed.
"""

from typing import Dict, List, Sequence

import numpy as np

from utils.log import get_logger

logger = get_logger(__name__)


def murray_weights(parents: Sequence[int], order: Sequence[int],
                   exponent: float = 2.0) -> np.ndarray:
    """
    Subtree weight of every node, normalised so the heaviest node is 1.

    Args:
        parents: parent index of each node, -1 for a root.
        order: node indices in an order where every child comes before its
            parent (deepest first).
        exponent: Murray exponent; 2 is the classic area-conserving rule.

    Returns:
        float array in (0, 1].
    """
    count = len(parents)
    weights = np.ones(count, dtype=np.float64)
    accumulated = np.zeros(count, dtype=np.float64)

    for i in order:
        # A leaf never accumulates anything and keeps the unit weight.
        weights[i] = max(1.0, accumulated[i]) ** (1.0 / exponent)
        parent = parents[i]
        if parent >= 0:
            accumulated[parent] += weights[i] ** exponent

    peak = float(weights.max()) if count else 1.0
    return weights / max(peak, 1e-9)


def depths_from_parents(parents: Sequence[int], order: Sequence[int]) -> np.ndarray:
    """
    Depth of every node, computed in one pass.

    `order` must list every parent before its children (shallowest first),
    which is the reverse of what `murray_weights` wants.
    """
    depths = np.zeros(len(parents), dtype=np.int64)
    for i in order:
        parent = parents[i]
        depths[i] = 0 if parent < 0 else depths[parent] + 1
    return depths


def infer_weights_from_polylines(polylines: List[Dict]) -> np.ndarray:
    """
    Rebuild the branch hierarchy from exported polylines and weight it.

    A polyline is a chain of segments with no branching inside it (that is how
    `build_polylines` cuts them), so its weight is constant along its length and
    one number per polyline is enough.

    The parent of a polyline is found by looking for the closest point of any
    *shallower* polyline to its own starting point. Exact coordinate matching
    would be wrong here: the export decimates the geometry, so the parent's copy
    of the junction point may well have been dropped.
    """
    if not polylines:
        return np.zeros(0)

    from scipy.spatial import cKDTree

    starts = np.array([p['points'][0] for p in polylines], dtype=np.float64)
    start_depths = np.array([p['depths'][0] for p in polylines], dtype=np.float64)

    all_points = np.concatenate([np.asarray(p['points'], dtype=np.float64) for p in polylines])
    owner = np.concatenate([
        np.full(len(p['points']), i, dtype=np.int64) for i, p in enumerate(polylines)
    ])

    tree = cKDTree(all_points)
    # A handful of neighbours is enough: the junction point is either the exact
    # same coordinate or one decimation step away.
    neighbours = min(len(all_points), 12)
    _, indices = tree.query(starts, k=neighbours)
    indices = np.atleast_2d(indices)

    parents = np.full(len(polylines), -1, dtype=np.int64)
    for i in range(len(polylines)):
        for j in indices[i]:
            candidate = int(owner[j])
            if candidate != i and start_depths[candidate] < start_depths[i]:
                parents[i] = candidate
                break

    order = np.argsort(start_depths)[::-1]
    weights = murray_weights(parents, order)

    orphans = int((parents < 0).sum())
    if orphans > 1:
        logger.debug('Subtree weights: %d polylines without a parent (roots or gaps)', orphans)

    return weights
