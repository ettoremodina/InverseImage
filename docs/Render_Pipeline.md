# Render Pipeline

How a frame is built, and why the pieces are split the way they are.

Implements phases 2 and 3 of [PLAN.md](PLAN.md). Stage 3 (the evolutionary swarm,
[Evolutionary_Swarm.md](Evolutionary_Swarm.md)) is not here yet; the slot it plugs
into is.

---

## 1. The shape of a frame

One loop, one frame at a time, in [rendering/timeline.py](../rendering/timeline.py):

```
    background          solid colour from the palette
        |
    + tree layer        SCA polylines, own transparent layer
        |               faded per-pixel where the flesh covers it
    + flesh layer       NCA cells, drawn as discs, then lit
        |
    + swarm layer       (stage 3 -- not implemented, slot reserved)
        |
    camera crop         a rectangle on the internal canvas
        |
    resolve             crop + downscale in ONE area-average resampling
        |
    grading             tone, aberration, vignette, grain
        |
    writer
```

Everything above `resolve` happens at the **internal canvas** resolution
(`render_size * render_supersample`, 1024 by default). Everything below is at video
resolution. There is exactly one enlargement and one reduction in the whole
pipeline — the camera does not add a second one, it just picks the rectangle that
the reduction reads from.

## 2. Why a timeline and not a sequence

The old `CombinedRenderer` rendered the stages one after the other and ffmpeg
concatenated the clips. Two things followed from that, both visible:

- at the cut the tree's sway stopped dead, because the last stage ran over a still
  background;
- the stages could never overlap, so the video was three exercises rather than one
  growth.

`TimelineRenderer` evaluates **every stage at every frame**. A stage is a window in
seconds plus an easing curve ([config/timing_config.py](../config/timing_config.py)),
and windows are meant to cross: the NCA enters while the tree is still growing, the
scaffold dissolves under the tissue, the swarm enters on tissue that is already
mature. `TimingConfig.validate()` warns when a stage starts after the previous one
has ended, because that silently turns the timeline back into a sequence.

### Adding stage 3

The loop knows nothing about the swarm. It asks for a layer:

```python
class StageLayer(Protocol):
    def layer(self, progress: float, time: float,
              beneath: np.ndarray) -> Optional[np.ndarray]:
        ...

timeline.render(sca_data, nca_data, path, stages={'swarm': my_swarm})
```

`progress` is the eased progress of that stage's own window, `beneath` is the frame
composited so far (the swarm needs it: it paints *over* the tissue and its fitness
is measured on what is already on the canvas). The return value is an RGBA layer
with straight alpha at canvas resolution, or None.

## 3. Layers

### Tree — [rendering/sca_renderer.py](../rendering/sca_renderer.py)

`TreeGeometry` flattens every polyline into contiguous arrays, so the per-frame sway
is one numpy expression, and groups them into `color_steps` bands. One band = one
colour + one line width = one Cairo path + one stroke. That is what keeps tens of
thousands of segments at a few dozen draw calls.

Each point carries two quantities that must not be confused:

- **`depths`** — drives the growth reveal. When a piece appears.
- **`values`** — drives colour and width. How it looks.

`branch_width_mode` selects where `values` comes from. In `'depth'` mode it is
`depth / max_depth`, the historical behaviour. In `'subtree'` mode it is
`1 - weight^gamma`, where `weight` is the Murray subtree weight
([rendering/tree_weights.py](../rendering/tree_weights.py)): a branch is thick in
proportion to how much tree it holds up, not to how far it is from the root. In
subtree mode the value is constant along a polyline — by construction a polyline
contains no branching — so no polyline is ever cut into bands.

The weights are computed at export time, where the hierarchy still exists, and
stored in the render data. For files written before that, the hierarchy is
reconstructed from the geometry with a KD-tree.

### Flesh — [rendering/cells.py](../rendering/cells.py) + [lighting.py](../rendering/lighting.py)

Each living cell is a disc. Its radius depends on its own maturity (alpha) and on a
global `cellularity` that decays over the NCA window:

```
gap    = cellularity * (1 - maturity)
radius = step * (radius_mature - gap * (radius_mature - radius_young))
```

A young cell is small and isolated; a mature cell has a radius past half the grid
step, so it interpenetrates its neighbours and the gaps close. You watch a colony
and end up with a surface — one parameter, no special-casing.

The cellularity never reaches zero (`cellularity_floor`). A perfectly regular
surface is dead; the residue, plus the lighting, reads as skin. The per-cell jitter
is seeded once per grid size, so a cell keeps its own irregularity for the whole
animation instead of twitching every frame.

The lighting builds a normal map from the gradient of the blurred alpha, applies
lambert + rim + bloom, and returns the layer with its alpha preserved (except where
the bloom adds a halo, which has to carry coverage of its own or it gets multiplied
away at composite time).

### Scaffold fade — `ScaffoldFadeConfig`

Two implementations, per the plan's working principle:

- `'alpha'` (default): the tree layer's alpha is multiplied by
  `1 - fade * strength * coverage`, where coverage is the NCA layer's own blurred
  alpha. A branch only vanishes where something actually grew on it.
- `'time'`: the whole tree fades on schedule. Trivial, kept for comparison.

## 4. Colour

Two modules, neither of which draws anything.

[color/palette.py](../color/palette.py) runs k-means on the non-transparent pixels
of the target and derives the background and the tree gradient from it. It is the
single place where the background colour is decided — the two hardcoded backgrounds
that used to live in `render_config.py` are now both written by
`PipelineConfig.apply_palette`.

[color/grading.py](../color/grading.py) is one pass over **every** frame of **every**
stage, right before it is written. Tone curve, chromatic aberration, vignette,
grain. Its real job is the last one: the grain breaks up the pixel grid, so a 128×128
simulation upscaled to 512 reads as film grain rather than as a poor image.

Animated grain compresses badly under h264 — expect a noticeably larger file. Switch
`grain_animated` off if that matters more than the look.

## 5. Progressive seeding

[nca/seeding.py](../nca/seeding.py) plants each NCA seed when the tree has grown down
to it, instead of lighting all of them at t=0. The birth time comes from the depth of
the nearest exported polyline point, so it works without re-running the SCA.

It hangs off a `hook` argument added to `CAModel.forward` / `generate_frames`, called
between two steps — the smallest possible change to the model, and backward
compatible.

**This is out of distribution for the current weights.** The model was trained with
every seed on from step 0, so a staggered seeding is a different problem from the one
it learned. That is the experiment of PLAN 3.2 step 1: render it and look for the two
failure modes — grown regions perturbed when a neighbour lights up, and late seeds
that fail to take. Retraining is decided *after* looking, not before.

## 6. Cost

At 512 video / 1024 canvas, per frame:

| Piece | Order of magnitude |
| --- | --- |
| tree strokes | a few dozen Cairo paths |
| cells | one arc + fill per living cell, ~10-20k at full growth |
| lighting | 3 gaussian blurs + 2 sobels at canvas size |
| resolve | one `INTER_AREA` resize |
| grading | a handful of whole-frame numpy ops |

The cell loop dominates. `cells.enabled = False` falls back to the old
nearest-neighbour blocks if a fast preview is needed; `render_supersample = 1` cuts
the rasterisation cost roughly fourfold.
