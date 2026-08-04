# Usage Guide

Complete reference for running the pipeline: setup, the three stages, every render
command, and the knobs that control each one.

For the *theory* behind each stage see
[Space_Colonization_Algorithm.md](Space_Colonization_Algorithm.md),
[Growing_Neural_Cellular_Automata.md](Growing_Neural_Cellular_Automata.md) and
[Particle_Advection_Refinement.md](Particle_Advection_Refinement.md).

---

## 1. Setup

```bash
pip install -r requirements.txt
```

Device selection is automatic (`config/common.py:get_device`): MPS → CUDA → CPU.
Training on CPU works but is slow; rendering is CPU-bound either way.

### GPU (CUDA) on Windows

**The `torch` wheel on PyPI is CPU-only for Windows.** `pip install torch` there
gives you a working install where `torch.cuda.is_available()` is `False`, with no
error to tell you why. The CUDA builds live on a separate index:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Pick the CUDA build your **driver** supports — not the CUDA toolkit version
`nvcc --version` reports, which is irrelevant here since PyTorch ships its own
runtime. Check with `nvidia-smi`:

| Driver | Use |
| --- | --- |
| ≥ 527 | `cu126` — safe default, and the widest range of torch versions |
| ≥ 570 | `cu128` |
| ≥ 580 | `cu130` |

Verify afterwards:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

A `+cpu` suffix in the version means you got the wrong wheel. If you already
installed the CPU build, `pip uninstall torch` first — pip will not replace it on
its own, because the plain version requirement is already satisfied.

Make sure you are installing into the **same interpreter** you run the pipeline
with. `python` on PATH is often not your virtualenv; `python -c "import sys;
print(sys.executable)"` tells you which one you are actually using.

---

## 2. The single control point: `config/pipeline.py`

There is **no CLI configuration**. Every script calls `load_config()`, which returns
a default-constructed `PipelineConfig`. To change anything you edit the dataclass
defaults in [config/pipeline.py](../config/pipeline.py).

The one setting you always touch is the target image:

```python
@dataclass
class PipelineConfig:
    target_image: str = 'images/jellyfish.png'   # <-- your image here
    output_base: str = 'outputs'
```

The config validates this path on construction, so a wrong filename fails
immediately with a clear message instead of somewhere deep in the SCA stage.

Every output path is *derived* from that filename stem, so all stages agree on where
things live without any further wiring:

| Property | Path |
| --- | --- |
| `nca_model_path` | `outputs/nca/<name>_model.pt` |
| `nca_animation_path` | `outputs/nca/<name>_animation.gif` |
| `sca_render_data_path` | `outputs/sca/<name>_render_data.json` |
| `sca_seeds_path` | `outputs/sca/<name>_seeds.json` |
| `render_output_dir` | `outputs/rendering/` |

`PipelineConfig.__post_init__` also propagates shared values down into the
sub-configs (`nca`, `sca`, `nca_render`, `sca_render`), so setting `target_image`,
`render_size` or `random_seed` once is enough — do **not** set them again on the
sub-configs, they get overwritten.

The input image should be RGBA or RGB; `nca/data.py:prepare_image` resizes it to
`nca.target_size` (default 128) and adds an opaque alpha channel if missing. For the
SCA mask, transparency/darkness defines the shape, so a PNG with a transparent
background gives the cleanest skeleton.

---

## 3. Running the pipeline

The three stages must run in order — each consumes the previous stage's files.

### Stage 1 — Skeleton (SCA)

```bash
python train_sca.py
```

Grows the space-colonization tree inside the target shape and extracts the seed
points that the NCA will grow from.

Writes to `outputs/sca/`:
- `<name>_render_data.json` — branch geometry as simplified polylines, consumed
  by the renderer (see `sca.simplify_tolerance` below)
- `<name>_seeds.json` — seed positions, consumed by NCA training and rendering
- `<name>_tree.png`, `<name>_tree_with_seeds.png` — static previews
- `<name>_edges.png` — edge map (only when `attractor_placement='edge'`)
- `<name>_metadata.json`

### Stage 2 — Texture model (NCA)

```bash
python train_nca.py
```

Trains the cellular automaton to grow the target image. It automatically picks up
`outputs/sca/<name>_seeds.json` if it exists (via `load_seed_positions()`); without
it, growth starts from a single centre seed.

Progressive multi-resolution training kicks in when `nca.progressive_stages` has
more than one entry; otherwise it is a single-resolution run at `nca.target_size`.

Writes to `outputs/nca/`:
- `<name>_model.pt` — weights **plus the pickled config** (the renderer reads the
  config back out of the checkpoint)
- `<name>_loss.png` (or `<name>_model_loss_<size>x<size>.png` in progressive mode)
- `<name>_animation.gif` — quick preview
- `<name>_metadata.json`

It also exports `outputs/rendering/<name>_nca_frames.npz`, which is what the
particle stage feeds on.

### Stage 3 — Render

```bash
python render.py --mode timeline
```

See the next section for all modes.

---

## 4. Render commands

`render.py` takes `--mode`, defaulting to `timeline`.

```bash
# The real deliverable: one overlapped timeline, SCA + NCA (+ swarm slot)
python render.py --mode timeline

# One graded frame, for calibrating cellularity / lighting / grading
python render.py --mode still --time 9

# Just the skeleton growing
python render.py --mode sca

# Just the NCA growing, from a single centre seed
python render.py --mode nca

# Just the particle pass, reusing existing NCA frames
python render.py --mode particles

# Legacy sequential pipeline: SCA -> NCA -> particles, concatenated
python render.py --mode combined
```

What each mode produces in `outputs/rendering/`:

| Mode | Requires | Output |
| --- | --- | --- |
| `timeline` | both `train_sca.py` and `train_nca.py` | **`<name>_timeline.mp4`**, `<name>_nca_frames.npz` |
| `still` | the same, plus existing NCA frames | `<name>_still_<t>s.png` |
| `sca` | `train_sca.py` | `<name>_sca.mp4`, `<name>_sca_final.png` |
| `nca` | `train_nca.py` | `<name>_nca.mp4`, `<name>_nca_frames.npz` |
| `particles` | an existing `<name>_nca_frames.npz` | `<name>_particles.mp4` |
| `combined` (legacy) | both | `<name>_combined.mp4`, `<name>_particles.mp4`, `<name>_full_pipeline.mp4` |

### `timeline` vs `combined`

`combined` renders the stages one after the other and concatenates the clips with
ffmpeg. That is why the tree's sway freezes at the cut: the last stage runs over a
still background.

`timeline` is one loop over frames. Every stage has a window in seconds and an
easing curve, evaluated at every frame, so the stages **overlap**: the NCA enters
while the tree is still growing, and the scaffold dissolves under the tissue as it
gets covered. Nothing is concatenated and nothing freezes.

Stage 3 (the evolutionary swarm) is not implemented yet. Its window is already in
the timing config and renders as a hold; it plugs in through the `stages` argument
of `TimelineRenderer.render` without touching the loop.

Useful flags:

```bash
# Skip the NCA rollout and reuse the frames already on disk
python render.py --mode timeline --reuse-frames
```

Existing output files are deleted before writing (`remove_if_exists`), so reruns
never append to a stale video.

### Iterating on particles only

Re-running `--mode combined` regenerates the NCA frames every time. To tune particle
parameters without paying for that:

```bash
python run_particles.py
```

It reads the existing `<name>_nca_frames.npz` and writes
`outputs/rendering/<name>_particles_standalone.mp4`, leaving the pipeline outputs
untouched.

---

## 5. Tuning the output

All of these live in `config/`. Restart the relevant stage after changing them.

### Video timing — `config/timing_config.py` (timeline mode)

Each stage has a start, an end and an easing curve, in seconds. The intervals are
meant to **overlap**; `TimingConfig.validate()` warns when they do not, because
disjoint windows silently turn the video back into a sequence of phases.

```python
total_duration: float = 25.0    # length of the finished video, hold included
hold: float = 3.0               # final stillness, only the sway breathing

sca   = StageWindow(0.0,  7.0,  'ease_out')      # fast start, slows down
nca   = StageWindow(3.0,  16.0, 'ease_in_out')   # enters while the tree still grows
swarm = StageWindow(12.0, 22.0, 'linear')        # enters on mature tissue
scaffold_fade = StageWindow(10.0, 18.0, 'ease_in_out')
```

Available easings are in `rendering/easing.py`: `linear`, `ease_in`, `ease_out`,
`ease_in_out`, `ease_out_cubic`.

The number of NCA simulation steps follows the NCA window automatically:
`max(animation_steps, nca_duration * render_fps)`.

### Video timing — `config/pipeline.py` (legacy and standalone modes)

```python
total_video_duration_seconds: float = 20.0   # length of the SCA+NCA section
sca_percentage: float = 0.4                  # share of that spent on the skeleton
nca_percentage: float = 0.6                  # share spent on the texture
render_size: int = 512                       # output resolution (square)
render_fps: int = 20
render_supersample: int = 2                  # internal canvas = render_size * this
animation_steps: int = 100                   # NCA steps for the preview/`--mode nca`
```

These only drive `--mode combined`, `--mode sca` and `--mode nca`. The particle
section is timed separately by `particles.particle_duration_seconds`.

### Resolution and antialiasing — `render_supersample`

Everything is drawn on a canvas `render_size * render_supersample` wide and reduced
with an area average, which is what removes the shimmer from the thin SCA branches
and softens the cell edges. The NCA simulation stays at 128 and the video stays at
512, so the file size does not change — only the rasterisation time (~3-4x). Set it
to `1` to disable.

The camera crop lives on that same canvas, so the frame is never enlarged twice.

### Skeleton shape — `config/sca_config.py`

| Setting | Effect |
| --- | --- |
| `num_attractors` (10000) | more attractors → denser, finer branching |
| `attractor_placement` (`'edge'`) | `'edge'` hugs the image contours, `'random'` fills the mask uniformly |
| `edge_threshold` (0.1) | lower → more edges detected |
| `influence_radius` (50) / `kill_distance` (1.0) | reach of an attractor, and how close a branch must get to consume it |
| `growth_step` (1.0) | branch segment length — smaller is smoother and slower |
| `adaptive_influence` (True) | grows the influence radius when growth stalls, up to `max_influence_radius` |
| `seed_mode` (`'tips'`) / `max_seeds` (1000) | which tree nodes become NCA seeds — `'tips'` uses branch ends, `'all'` uses every node |
| `simplify_tolerance` (0.5) | geometry decimation applied when exporting the render data, in source pixels. Sub-pixel values are visually free and cut render time and file size several-fold; `0` keeps the raw one-segment-per-step geometry |

### Texture model — `config/nca_config.py`

| Setting | Effect |
| --- | --- |
| `target_size` (128) | simulation grid; the main cost driver for training |
| `channel_n` (16) / `hidden_size` (128) | model capacity |
| `n_epochs` (2000), `batch_size` (8), `steps_per_epoch` (50) | training budget |
| `use_pattern_pool` / `pool_size` | sample-pool persistence training — keep on for stable long-running growth |
| `progressive_stages` | list of `ResolutionStage(size, epochs, batch_size, accumulation_steps)`; more than one entry enables progressive training |
| `use_mixed_precision` (True) | faster on CUDA |

### Look — `config/render_config.py`

**`SCARenderConfig`** — `branch_color` → `branch_color_end` gradient (normally
overwritten by the palette, see below), `branch_base_width`/`branch_tip_width`
taper, `sway_magnitude`/`sway_frequency` for the idle animation. `color_steps` (64)
is how many bands the gradient is quantised into: each band is one batched stroke,
so lowering it speeds up rendering and raising it smooths the gradient. Below ~32
the banding shows.

Line quality:

| Setting | Effect |
| --- | --- |
| `branch_width_mode` (`'subtree'`) | `'subtree'` sizes a branch by how much tree it carries (Murray: `w_parent² = Σ w_child²`); `'depth'` is the old distance-from-root rule |
| `branch_width_gamma` (0.45) | visual spread of the subtree weights. A tree with thousands of leaves has a trunk weight of `sqrt(leaves)`, so raw weights pile up near zero; `1.0` = raw Murray |
| `branch_smoothing` (True) / `smoothing_subdivisions` (3) | Catmull-Rom interpolation of the polylines — SCA emits one straight segment per growth step |
| `prune_tips` (0.5), `prune_depth_start` (0.5), `prune_max_points` (8) | thins the deep, short, terminal twigs that turn the silhouette into fuzz. Long deep branches are always kept |

The subtree weights are computed at export time and stored in the render data. Files
written before that existed still work: the hierarchy is reconstructed from the
geometry at load (you get one log line saying so). Re-run `train_sca.py` to store
them properly.

**`NCARenderConfig`** — `alpha_threshold`, `temporal_smoothing` (0.25),
`initial_repeats`/`decay_rate` for the slow start, plus two sub-configs:

`cells` (`CellRenderConfig`) — every living cell is drawn as its own disc instead of
a square block of pixels, and the cellularity is a **function of time**: a young cell
is small with a visible gap, a mature cell overflows the grid step and merges with
its neighbours into a continuous surface. You see the colony while it grows and the
tissue at the end.

| Setting | Effect |
| --- | --- |
| `cellularity` (1.0) | how marked the colony look is at the start |
| `cellularity_floor` (0.1) | residue that survives at the end. Do not set to 0: a perfectly smooth surface reads as dead plastic |
| `cellularity_curve` (1.6) | >1 keeps the colony look alive longer |
| `radius_mature` (0.78) / `radius_young` (0.34) | cell radius as a fraction of the grid step. Above 0.5 the neighbours overlap, which is what closes the gaps |
| `radius_jitter` / `position_jitter` | deterministic per-cell irregularity, scaled by the current cellularity |
| `enabled` (True) | False falls back to the old nearest-neighbour blocks |

`lighting` (`LightingConfig`) — a normal map from the gradient of the smoothed alpha,
then lambert + rim light + bloom. `light_direction`, `ambient`, `diffuse`,
`rim_intensity`, `bloom_threshold`, `bloom_intensity`, all disableable.

**`ScaffoldFadeConfig`** — how the tree disappears once the flesh covers it.
`mode='alpha'` (default) drives the fade from the NCA coverage, so a branch only
vanishes where something actually grew on it; `mode='time'` fades the whole tree on
schedule.

### Palette — `config/palette_config.py`

The background and the SCA gradient are derived from the target image by k-means
(k = 5) over its non-transparent pixels, so the three stages sit in one chromatic
world instead of being hardcoded in two places that drift apart.

| Setting | Effect |
| --- | --- |
| `enabled` (True) | False keeps the literal colours in `render_config.py` |
| `background_rule` | `'dark_desaturated'` (the subject's own shadow) or `'complementary'` (more graphic separation) |
| `background_value` / `background_saturation` | how dark and how colourful the ground is |
| `tree_from_palette`, `tree_base_value`, `tree_tip_value`, `tree_saturation` | the SCA gradient, built on the dominant hue instead of brown → green |
| `background_color`, `branch_color`, `branch_color_end` | manual overrides; they always win |

### Grading — `config/grading_config.py`

One pass applied to **every** frame of **every** stage, right before it is written:
tone curve, chromatic aberration, vignette, grain. Each effect switches off on its
own, and `enabled=False` disables the whole pass.

The grain is the important one: it breaks up the pixel grid, so the low simulation
resolution reads as film grain instead of as a poor image. `grain_animated=True`
(default) draws new noise every frame; set it to `False` if h264 finds it too noisy.

### Camera — `config/camera_config.py`

Slow push-in plus a drift, cropped out of the supersampled canvas.
`zoom_start`/`zoom_end` are the fraction of the canvas visible; `drift_x`/`drift_y`
move the crop centre. `enabled=False` costs exactly nothing.

### Progressive seeding — `config/seeding_config.py`

Instead of lighting every NCA seed at t=0, each seed is injected when the tree has
grown down to it (`order='depth'`, birth time from the depth of the nearest branch
point). `spread` is the fraction of the rollout used for the seeding.

Note this is out of distribution for the current model, which was trained with all
seeds on from step 0 — that is deliberate, it is the experiment of PLAN 3.2. The two
ways it can break: grown regions perturbed when a neighbour lights up, and late
seeds that fail to take. `enabled=False` restores the old behaviour.

### Particles — `config/particle_config.py`

| Setting | Effect |
| --- | --- |
| `particle_count` (2000) | density of the flow |
| `particle_speed` (10.0) | advection step size |
| `particle_duration_seconds` (10.0) | length of the particle section |
| `particle_trail_fade` (1.0) | 1.0 keeps full trails; lower values fade them out |
| `particle_stretch_factor` (2.0) | motion-blur elongation along the velocity |
| `particle_radius` (1) | dot size in pixels |
| `particle_outline_width` (0) | extra radius drawn behind each head in `particle_outline_color`. Leave at 0: the outline area grows with the square of the radius, so even 1 px covers 4x more pixels than a radius-1 head and buries the image |

Particle colours are sampled from the **target image**, not from the NCA output, so
the source image's palette carries through to the final frame. Transparent areas of
the target are flattened onto the render background first, so particles that drift
outside the shape stay invisible instead of painting the PNG's black backing.

---

## 6. Utilities

```bash
# Paint an RGBA target image by hand (tkinter GUI, 512x512, transparent background)
python tools/image_painter.py

# Split outputs/rendering/<name>_full_pipeline.mp4 into the per-stage GIFs,
# final frames and triptych used by the README (writes into docs/assets/)
python tools/split_video.py
```

`split_video.py` derives its cut points from the same config values used to build
the video, so it must run against a video rendered with the current config.

---

## 7. Typical workflows

**New image, from scratch:**

```bash
python train_sca.py
python train_nca.py
python render.py --mode timeline
```

**Changed only the skeleton parameters** — the seeds moved, so the NCA must be
retrained:

```bash
python train_sca.py
python train_nca.py
python render.py --mode timeline
```

**Changed only render colours, timing or resolution** — no retraining needed:

```bash
python render.py --mode timeline --reuse-frames
```

**Calibrating the look** (cellularity, lighting, grading, palette) — never do this on
a video:

```bash
python render.py --mode still --time 9
```

**Changed only particle parameters** (legacy stage 3):

```bash
python run_particles.py
```

---

## 8. Troubleshooting

| Symptom | Cause |
| --- | --- |
| `FileNotFoundError: target_image not found` | `target_image` in `config/pipeline.py` points at a file that does not exist |
| `SCA render data not found` / `SCA seed positions not found` | run `train_sca.py` first |
| `NCA frames not found at ...` | run `train_nca.py`, or `render.py --mode nca`, before `--mode particles` |
| Unpickling error loading an old `.pt` | handled by the `sys.modules['nca.config'] = config.nca_config` shim at the top of `render.py`; keep that import first if you refactor |
| Growth is sparse / SCA stops early | raise `num_attractors`, or `influence_radius`; check `min_attractor_kill_ratio` and `stagnation_limit` |
| `Legacy SCA data: simplifying ... segments` at load | the render data predates the polyline format. It is converted in memory each time; re-run `train_sca.py` to write the compact format once |
| Merge fails with a concat error | all clips must share codec and resolution. They do when produced by this pipeline; mixing in an externally encoded file will break the stream copy |
| `TimingConfig: stage 'x' starts after the previous stage ends` | the stage windows in `config/timing_config.py` no longer overlap, so the timeline has silently become a sequence of phases |
| `SCA: render data carries no subtree weights` | the render data predates the Murray weights. The hierarchy is reconstructed from the geometry at load; re-run `train_sca.py` to store them |
| The timeline mp4 is much bigger than expected | animated grain. It roughly quadruples the bitrate (measured: 7.1 MB vs 1.8 MB per 100 frames at 512²). Set `grading.grain_animated = False` |
| Rendering is slow | the per-cell Cairo loop dominates. `render_supersample = 1` cuts it ~4x, `nca_render.cells.enabled = False` falls back to the old blocks |
