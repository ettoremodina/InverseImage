"""
Data exporters to convert simulation data into renderer-friendly format.
Keeps rendering module decoupled from simulation code.
"""

import json
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any

import numpy as np


def _simplify(points: np.ndarray, depths: np.ndarray, tolerance: float):
    """
    Douglas-Peucker decimation of one polyline, keeping each kept point's depth.

    SCA emits a point every growth step (1 source pixel), which is far denser
    than the output raster needs. Dropping points that sit within `tolerance`
    of the straight line through their neighbours is invisible at stroke widths
    of a few pixels but removes ~75% of the geometry.
    """
    n = len(points)
    if n <= 2 or tolerance <= 0:
        return points, depths

    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]

    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        start, end = points[a], points[b]
        span = end - start
        length = float(np.hypot(span[0], span[1]))
        between = points[a + 1:b] - start

        if length < 1e-9:
            distance = np.hypot(between[:, 0], between[:, 1])
        else:
            # |cross product| / |span| is the perpendicular distance
            distance = np.abs(span[0] * between[:, 1] - span[1] * between[:, 0]) / length

        worst = int(np.argmax(distance))
        if distance[worst] > tolerance:
            split = a + 1 + worst
            keep[split] = True
            stack.append((a, split))
            stack.append((split, b))

    return points[keep], depths[keep]


def build_polylines(branches: List[Dict], tolerance: float = 0.5) -> List[Dict]:
    """
    Collapse a flat list of one-step SCA segments into polylines.

    Space colonization emits one segment per growth step, and many branches
    retrace the exact same path, so the raw list is both huge and redundant
    (on a typical tree: 125k segments, 40% of them exact duplicates, which
    collapse to ~3k polylines). Rendering cost is dominated by the number of
    primitives, so this is the single biggest lever on render time.

    Three passes:
      1. Drop duplicate (start, end) segments, keeping the shallowest one so
         the growth animation still reveals the path at the same moment.
      2. Chain segments that follow each other with no branching in between.
      3. Decimate each chain to `tolerance` (in source pixels).

    Each point carries its own depth, so the growth animation can still reveal
    the tree progressively — and more smoothly than before, because a partly
    revealed segment is now interpolated instead of popping in whole.

    Returns a list of
    {"points": [[x, y], ...], "depths": [int, ...], "is_tip": bool}.
    """
    if not branches:
        return []

    # 1. Deduplicate, keeping the shallowest occurrence of each segment.
    unique: Dict[tuple, Dict] = {}
    for b in branches:
        key = (tuple(b['start']), tuple(b['end']))
        previous = unique.get(key)
        if previous is None or b['depth'] < previous['depth']:
            unique[key] = b
    segments = list(unique.values())

    # 2. Index by start point. A segment can only be chained onto its parent
    #    when that parent's end point has exactly one outgoing segment.
    by_start: Dict[tuple, List[int]] = {}
    out_degree = Counter()
    for i, s in enumerate(segments):
        start = tuple(s['start'])
        by_start.setdefault(start, []).append(i)
        out_degree[start] += 1

    has_incoming = {tuple(s['end']) for s in segments}

    consumed = [False] * len(segments)
    polylines = []

    for i, segment in enumerate(segments):
        if consumed[i]:
            continue
        # Skip segments that another segment will absorb as its continuation.
        start = tuple(segment['start'])
        if out_degree[start] == 1 and start in has_incoming:
            continue

        consumed[i] = True
        points = [segment['start'], segment['end']]
        depths = [segment['depth'], segment['depth'] + 1]
        current = segment

        while True:
            following = by_start.get(tuple(current['end']), ())
            if len(following) != 1:
                break
            j = following[0]
            if consumed[j] or segments[j]['depth'] != current['depth'] + 1:
                break
            consumed[j] = True
            current = segments[j]
            points.append(current['end'])
            depths.append(current['depth'] + 1)

        polylines.append(_finish(points, depths, current['is_tip'], tolerance))

    # Anything left is part of a cycle; emit it as a standalone polyline so no
    # geometry is silently dropped.
    for i, segment in enumerate(segments):
        if not consumed[i]:
            polylines.append(_finish(
                [segment['start'], segment['end']],
                [segment['depth'], segment['depth'] + 1],
                segment['is_tip'],
                tolerance,
            ))

    return polylines


def _finish(points: List, depths: List, is_tip: bool, tolerance: float) -> Dict:
    """Decimate one chain and package it as a serialisable polyline."""
    kept_points, kept_depths = _simplify(
        np.asarray(points, dtype=np.float64),
        np.asarray(depths, dtype=np.int64),
        tolerance,
    )
    return {
        "points": kept_points.tolist(),
        "depths": kept_depths.tolist(),
        "is_tip": bool(is_tip),
    }


def export_sca_data(tree, output_path: str, tolerance: float = 0.5):
    """
    Export SCA tree to JSON format for rendering.

    Format:
    {
        "source_width": int,
        "source_height": int,
        "polylines": [
            {
                "points": [[x, y], ...],
                "depth_start": int,   # depth of the first segment
                "is_tip": bool
            }
        ]
    }
    """
    def get_depth(branch) -> int:
        depth = 0
        current = branch
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth

    branches_data = [
        {
            "start": [branch.start_pos.x, branch.start_pos.y],
            "end": [branch.end_pos.x, branch.end_pos.y],
            "depth": get_depth(branch),
            "is_tip": branch.is_tip,
        }
        for branch in tree.branches
    ]

    polylines = build_polylines(branches_data, tolerance=tolerance)
    points = sum(len(p['points']) for p in polylines)
    print(f"  Simplified {len(branches_data)} segments into "
          f"{len(polylines)} polylines ({points} points)")

    data = {
        "source_width": tree.mask_width,
        "source_height": tree.mask_height,
        "polylines": polylines
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f)

    return data


def export_nca_frames(frames: List, output_path: str):
    """
    Export NCA animation frames to NPZ format for rendering.
    
    Args:
        frames: List of RGBA frames (torch tensors or numpy arrays)
                Each frame shape: [H, W, 4] with values in [0, 1]
        output_path: Path to save .npz file
    
    Format:
        - frames: np.ndarray of shape [N, H, W, 4], float32
        - source_height: int
        - source_width: int
        - num_frames: int
    """
    processed = []
    for frame in frames:
        if hasattr(frame, 'numpy'):
            frame = frame.numpy()
        frame = np.clip(frame, 0, 1).astype(np.float32)
        processed.append(frame)
    
    frames_array = np.stack(processed, axis=0)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        frames=frames_array,
        source_height=frames_array.shape[1],
        source_width=frames_array.shape[2],
        num_frames=frames_array.shape[0]
    )
    
    print(f"Exported {len(frames)} NCA frames to {output_path}")
    print(f"  Shape: {frames_array.shape}")
    
    return {
        "source_height": frames_array.shape[1],
        "source_width": frames_array.shape[2],
        "num_frames": len(frames)
    }


def load_sca_data(path: str) -> Dict[str, Any]:
    """
    Load SCA render data, converting the legacy flat-segment format on the fly.

    Files written before the polyline format still render; they just pay the
    one-off simplification cost at load time.
    """
    with open(path, 'r') as f:
        data = json.load(f)

    if "polylines" not in data:
        branches = data.pop("branches", [])
        print(f"  Legacy SCA data: simplifying {len(branches)} segments...")
        data["polylines"] = build_polylines(branches)
        print(f"  -> {len(data['polylines'])} polylines")

    return data


def load_nca_frames(path: str) -> Dict[str, Any]:
    """
    Load NCA frames from NPZ file.
    
    Returns:
        Dict with keys: frames, source_height, source_width, num_frames
    """
    data = np.load(path)
    return {
        "frames": data["frames"],
        "source_height": int(data["source_height"]),
        "source_width": int(data["source_width"]),
        "num_frames": int(data["num_frames"])
    }
