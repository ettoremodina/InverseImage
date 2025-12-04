# Space Colonization Algorithm (SCA)

A comprehensive guide to the 2D Space Colonization Algorithm implementation for generating tree-like branching structures within arbitrary shapes.

**Based on:** *"Modeling Trees with a Space Colonization Algorithm"* by Adam Runions, Brendan Lane, and Przemyslaw Prusinkiewicz (2007)

---

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Algorithm Steps](#algorithm-steps)
5. [Code Architecture](#code-architecture)
6. [Parameter Tuning Guide](#parameter-tuning-guide)
7. [Extensions: Directional Clustering](#extensions-directional-clustering)

---

## Overview

The Space Colonization Algorithm simulates how plants grow by following "attractors" - points representing nutrients or growth hormones. Branches grow toward these attractors, consuming them when close enough, until all attractors are consumed.

The key insight: **branches compete for attractors**, naturally creating organic branching patterns.

### Visual Process

```
Iteration 0:          Iteration N:           Final:
    · · · ·              · · · ·            
   · · · · ·            ·  /|\  ·              /|\
  · · · · · ·          · / | \ ·             / | \
 · · · · · · ·        ·/  |  \·            /  |  \
· · · · · · · ·       /   |   \           /   |   \
      |               |   |   |          /|   |   |\
      |               |   |   |         / |   |   | \
      ●               ●   ●   ●        ●  ●   ●   ●  ●

  (attractors)       (growing)         (complete tree)
```

---

## Core Concepts

### 1. Attractors

Attractors are points scattered within a target shape that represent "food sources" guiding growth.

```python
class Attractor:
    position: Vector2D  # (x, y) coordinates
    alive: bool         # False when consumed by a branch
```

**Biological analogy:** Auxin (plant growth hormone) concentrated in certain areas, attracting growing tips.

### 2. Branches

Branches are line segments forming a tree structure. Each branch connects to a parent (except the root) and may have children.

```python
class Branch:
    start_pos: Vector2D    # Where branch begins
    end_pos: Vector2D      # Where branch ends (the "tip")
    parent: Branch         # Parent branch (None for root)
    children: List[Branch] # Child branches
```

**Key property:** `is_tip` - A branch is a "tip" if it has no children. Only tips can grow.

### 3. The Tree

The Tree manages the entire system: all attractors, all branches, and the growth loop.

---

## Mathematical Foundations

### Vector Operations

All growth calculations use 2D vectors:

**Magnitude (length):**
$$||\vec{v}|| = \sqrt{x^2 + y^2}$$

**Normalization (unit vector):**
$$\hat{v} = \frac{\vec{v}}{||\vec{v}||}$$

**Distance between points:**
$$d(A, B) = ||B - A|| = \sqrt{(B_x - A_x)^2 + (B_y - A_y)^2}$$

### Growth Direction Calculation

For a branch tip $B$ influenced by attractors $A_1, A_2, ..., A_n$:

**Step 1:** Calculate direction to each attractor
$$\vec{d}_i = A_i - B$$

**Step 2:** Normalize each direction
$$\hat{d}_i = \frac{\vec{d}_i}{||\vec{d}_i||}$$

**Step 3:** Average the normalized directions
$$\vec{d}_{avg} = \sum_{i=1}^{n} \hat{d}_i$$

**Step 4:** Normalize the average
$$\hat{d}_{grow} = \frac{\vec{d}_{avg}}{||\vec{d}_{avg}||}$$

**Step 5:** Calculate new tip position
$$B_{new} = B + \hat{d}_{grow} \times \text{growth\_step}$$

### Why Normalize Before Averaging?

Without normalization, distant attractors would have more influence:

```
Without normalization:        With normalization:
    A₁ (far)                      A₁ (far)
    ↓ (long vector)               ↓ (unit vector)
    B ← A₂ (close)                B ← A₂ (close)
    
Result: biased toward A₁        Result: equal influence
```

---

## Algorithm Steps

### Initialization

```python
def _initialize(self):
    # 1. Load mask image (defines valid growth region)
    self.mask = load_mask(config.mask_image_path)
    
    # 2. Scatter attractors randomly within mask
    positions = sample_attractors(self.mask, config.num_attractors)
    self.attractors = [Attractor(pos) for pos in positions]
    
    # 3. Create root branch at bottom-center of mask
    root_pos = find_bottom_center(self.mask)
    root_branch = Branch(root_pos, root_pos + Vector2D(0, -1) * growth_step)
    self.branches.append(root_branch)
```

### Main Growth Loop

Each iteration performs these steps:

```
┌─────────────────────────────────────────────────────────────┐
│                     GROWTH ITERATION                        │
├─────────────────────────────────────────────────────────────┤
│  1. ASSOCIATE: Each attractor finds its closest branch tip  │
│                                                             │
│  2. KILL: Attractors within kill_distance are consumed      │
│                                                             │
│  3. INFLUENCE: Attractors within influence_radius guide     │
│                their closest tip                            │
│                                                             │
│  4. CLUSTER: Group attractor directions for branching       │
│                                                             │
│  5. GROW: Each influenced tip creates new branch(es)        │
│                                                             │
│  6. CLEANUP: Remove dead attractors                         │
│                                                             │
│  7. REBUILD: Update spatial index with new tips             │
└─────────────────────────────────────────────────────────────┘
```

### Step 1-3: Attractor Association

```python
def _associate_attractors(self):
    for attractor in self.attractors:
        # Find closest branch tip
        closest_tip = min(tips, key=lambda t: distance(attractor, t.end_pos))
        dist = distance(attractor.position, closest_tip.end_pos)
        
        # Kill if too close
        if dist < kill_distance:
            attractor.kill()
            continue
        
        # Influence if within radius
        if dist < influence_radius:
            branch_influences[closest_tip].append(attractor)
```

**Visual representation:**

```
         influence_radius
        ┌───────────────────┐
        │    ·  ·  ·        │
        │  ·    ·    ·      │
        │ ·  ╭─────╮  ·     │
        │ · │kill  │ ·      │
        │   │dist. │        │
        │   ╰──┬───╯        │  ← Branch tip
        └──────┴────────────┘
               │
          (kill zone - attractors consumed here)
```

### Step 4: Directional Clustering (Extension)

Standard SCA averages all attractor directions → single growth direction.

**Problem:** With many attractors in opposite directions, they cancel out:

```
Standard averaging:
     ·←─ A₁                A₁ + A₂ = 0 (cancel!)
         ↓
         B
         ↑
     ·←─ A₂
```

**Solution:** Cluster attractors by direction, create one branch per cluster:

```python
def _cluster_directions(self, branch, attractors):
    # Calculate direction to each attractor
    directions = [(a.position - branch.end_pos).normalize() for a in attractors]
    
    clusters = []
    for d in directions:
        placed = False
        for cluster in clusters:
            # Check if direction is similar to cluster average
            cluster_avg = average(cluster).normalize()
            similarity = dot(d, cluster_avg)  # cosine similarity
            
            if similarity > branch_angle_threshold:
                cluster.append(d)
                placed = True
                break
        
        if not placed:
            clusters.append([d])  # New cluster
    
    # Return one averaged direction per valid cluster
    return [average(c).normalize() for c in clusters if len(c) >= min_per_branch]
```

**Cosine Similarity:**
$$\cos(\theta) = \hat{d}_1 \cdot \hat{d}_2 = d_{1x} \cdot d_{2x} + d_{1y} \cdot d_{2y}$$

| Threshold | Angle Apart | Branching |
|-----------|-------------|-----------|
| 0.7 | ~45° | Less branching |
| 0.5 | ~60° | Moderate |
| 0.3 | ~70° | More branching |
| 0.0 | 90° | Maximum branching |

### Step 5: Branch Growth

```python
def _grow_branches(self, branch_influences):
    new_branches = []
    
    for branch, attractors in branch_influences.items():
        # Get growth directions (may be multiple due to clustering)
        directions = self._cluster_directions(branch, attractors)
        
        for direction in directions:
            new_end = branch.end_pos + direction * growth_step
            
            # Boundary check - stay inside mask
            if not self._is_inside_mask(new_end):
                continue
            
            new_branch = branch.create_child(direction, growth_step)
            new_branches.append(new_branch)
    
    return new_branches
```

---

## Code Architecture

```
sca/
├── __init__.py          # Package exports
├── config.py            # SCAConfig dataclass
├── vector.py            # Vector2D math operations
├── attractor.py         # Attractor class
├── branch.py            # Branch class with tree structure
├── tree.py              # Main algorithm (Tree class)
├── spatial.py           # KD-tree for efficient lookups
├── mask.py              # Image loading and sampling
└── visualization.py     # Matplotlib rendering
```

### Class Relationships

```
┌─────────────────────────────────────────────────────────────┐
│                         Tree                                │
│  ┌─────────────┐    ┌─────────────┐    ┌────────────────┐  │
│  │ attractors  │    │  branches   │    │ spatial_index  │  │
│  │ List[Attr.] │    │ List[Branch]│    │ BranchSpatial  │  │
│  └──────┬──────┘    └──────┬──────┘    └───────┬────────┘  │
│         │                  │                   │            │
│         ▼                  ▼                   ▼            │
│  ┌────────────┐     ┌────────────┐      ┌────────────┐     │
│  │ Attractor  │     │   Branch   │      │  cKDTree   │     │
│  │ - position │     │ - start    │      │ (scipy)    │     │
│  │ - alive    │     │ - end      │      │            │     │
│  └────────────┘     │ - parent   │      └────────────┘     │
│                     │ - children │                          │
│                     └────────────┘                          │
│                           │                                 │
│                     inherits from                           │
│                           ▼                                 │
│                    ┌────────────┐                           │
│                    │  Vector2D  │                           │
│                    │  - x, y    │                           │
│                    └────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Parameter Tuning Guide

### Key Parameters

| Parameter | Effect | Typical Range |
|-----------|--------|---------------|
| `num_attractors` | Density of final result | 3000-10000 |
| `influence_radius` | How far tips can "see" | 50-200 |
| `kill_distance` | When attractors are consumed | 2-10 |
| `growth_step` | Branch segment length | 1-5 |
| `branch_angle_threshold` | Branching frequency | 0.2-0.7 |

### Tuning Rules

**For denser results:**
- ↑ `num_attractors`
- ↓ `kill_distance` (branches get closer before consuming)
- ↓ `growth_step` (finer segments)

**For more branching:**
- ↓ `branch_angle_threshold` (more directional clusters)
- ↓ `min_attractors_per_branch`
- ↓ `influence_radius` (tips compete more locally)

**For smoother curves:**
- ↓ `growth_step`
- ↑ `num_attractors`

### Parameter Relationships

```
                    influence_radius
                   ┌───────────────────────────────┐
                   │                               │
                   │      kill_distance            │
                   │     ┌───────────┐             │
                   │     │           │             │
                   │     │     ●     │ ← tip       │
                   │     │           │             │
                   │     └───────────┘             │
                   │                               │
                   └───────────────────────────────┘

Ratio influence_radius / kill_distance:
  - High (20:1) → branches spread far before consuming → sparse tree
  - Low (5:1)   → branches consume quickly → dense tree
```

### Recommended Starting Points

**For 512x512 image:**
```python
num_attractors = 8000
influence_radius = 120.0
kill_distance = 3.0
growth_step = 2.0
branch_angle_threshold = 0.3
```

**For 256x256 image:** (halve distances)
```python
num_attractors = 4000
influence_radius = 60.0
kill_distance = 1.5
growth_step = 1.0
```

---

## Extensions: Directional Clustering

### The Problem with Standard SCA

Standard SCA: each tip creates exactly ONE new branch per iteration.

With a single root and large influence radius:
1. All attractors influence the single tip
2. Directions average to the centroid
3. Tip grows toward center, gets "stuck"

```
    · · · ·
   · ·   · ·     ←── attractors pull equally from all sides
  ·   ↖ ↗   ·
 ·    ← ● →  ·   ←── tip oscillates at center
  ·   ↙ ↘   ·
   · ·   · ·
    · · · ·
```

### Our Solution: Directional Clustering

When attractors point in **different directions** (angle > threshold), create **multiple branches**:

```
Before clustering:              After clustering:
                               
     A₁  A₂                        A₁  A₂    
      ↘  ↓                          ↘  ↓     
        ● ← A₃                        ●──→ A₃  (3 branches!)
      ↗  ↑                          ↗  ↑     
     A₄  A₅                        A₄  A₅    

Average: ≈ 0 (stuck)            Three distinct clusters
```

### Clustering Algorithm

```python
# For each direction d:
for d in directions:
    for cluster in clusters:
        similarity = dot(d, cluster_average)
        if similarity > threshold:
            cluster.append(d)  # Similar direction, add to cluster
            break
    else:
        clusters.append([d])   # New direction, new cluster
```

**Time complexity:** O(D × C) where D = directions, C = clusters

---

## Complete Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│                        main_sca.py                          │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│   config = SCAConfig()                                      │
│   tree = Tree(config)                                       │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│   Tree._initialize()                                        │
│   ├── Load mask image                                       │
│   ├── Sample N attractors within mask                       │
│   ├── Create root branch at bottom-center                   │
│   └── Build spatial index                                   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│   tree.grow()                                               │
│   │                                                         │
│   └── while attractors remain and iterations < max:         │
│       │                                                     │
│       ├── _associate_attractors()                           │
│       │   └── For each attractor: find closest tip          │
│       │       ├── If dist < kill_distance: kill attractor   │
│       │       └── If dist < influence_radius: add to tip    │
│       │                                                     │
│       ├── _grow_branches()                                  │
│       │   └── For each influenced tip:                      │
│       │       ├── _cluster_directions() → N directions      │
│       │       └── Create N child branches (if inside mask)  │
│       │                                                     │
│       ├── _cleanup_attractors() → remove dead               │
│       │                                                     │
│       └── spatial_index.rebuild() → update with new tips    │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│   visualize_tree(tree)                                      │
│   └── Draw all branch segments with matplotlib              │
└─────────────────────────────────────────────────────────────┘
```

---

## References

1. Runions, A., Lane, B., & Prusinkiewicz, P. (2007). *Modeling Trees with a Space Colonization Algorithm*. Eurographics Workshop on Natural Phenomena.

2. Original paper: Describes "open" venation for tree skeletons and "closed" venation for leaf veins.

3. Key insight from paper: The algorithm naturally handles obstacle avoidance and space-filling - branches grow toward available attractors, avoiding areas already colonized.
