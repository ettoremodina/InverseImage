"""
What the finished picture is like -- the measurements the §12 diagnostics do not make.

`metrics.py` watches the *population*: who is alive, who is eating, how the
error curve moves. That is enough to tell a starving run from a healthy one and
nothing else. It cannot tell a run that repainted the subject from one that
smeared it, and it cannot tell brushwork from speckle, because both of those
are properties of the image rather than of the swarm.

This module reads the canvas instead, and it does so on three axes that the
design document names as the point of stage 3:

- **fidelity, on the subject only.** The lab target is an RGBA cut-out: ~80% of
  the frame is background where base and target already agree, so a global mean
  error is diluted about fivefold and a real 2% gain on the jellyfish reads as
  0.4%. Every number here is masked to `nutrient > threshold`, which is also
  exactly where the agents are allowed to live (§5.3).
- **detail** (§3). The swarm is handed an image that is missing its high
  frequency and the residual error "is almost all detail" -- so the honest
  question is not "is the error lower" but "is the high-frequency band back,
  and is it in the right places". `detail_ratio` answers the first,
  `gradient_alignment` the second: noise raises the first and lowers the
  second, which is precisely how a speckled run gives itself away.
- **brushwork** (§5.2, §5.4). The stroke has to read as a stroke.
  `stroke_coherence` is the structure-tensor anisotropy of the pigment layer:
  elongated marks score high, isotropic blobs and per-pixel noise score low.
  It is the one number in the file that is about the aesthetic rather than the
  reconstruction, and it is the reason the tuner can be told to prefer a
  painting over a photocopy.

Nothing here is read by the simulation. These are judge-side measurements, run
after the fact on the final state -- the target is already visible to this
layer by construction (it writes contact sheets against it), so no rule is bent.
"""

import math
from dataclasses import dataclass, fields as dataclass_fields
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from swarm.metrics import StepMetrics

_EPS = 1e-8


# ==================== small image ops (torch, device-agnostic) ====================

def _as_nchw(field: torch.Tensor) -> torch.Tensor:
    """(H, W) or (H, W, C) -> (1, C, H, W)."""
    if field.dim() == 2:
        return field.unsqueeze(0).unsqueeze(0)
    return field.permute(2, 0, 1).unsqueeze(0)


def _gaussian_kernel(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


def gaussian_blur(field: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable gaussian blur of an (H, W) or (H, W, C) field, shape preserved."""
    if sigma <= 0:
        return field
    x = _as_nchw(field)
    c = x.shape[1]
    k = _gaussian_kernel(sigma, field.device, field.dtype)
    r = (k.numel() - 1) // 2

    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode='replicate'), kx, groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode='replicate'), ky, groups=c)

    x = x.squeeze(0)
    return x.squeeze(0) if field.dim() == 2 else x.permute(1, 2, 0)


_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
_SOBEL_Y = _SOBEL_X.T.contiguous()


def gradients(field: torch.Tensor):
    """(gx, gy) of a scalar (H, W) field, same shape, Sobel with replicate padding."""
    x = _as_nchw(field)
    kx = _SOBEL_X.to(field.device, field.dtype).view(1, 1, 3, 3)
    ky = _SOBEL_Y.to(field.device, field.dtype).view(1, 1, 3, 3)
    padded = F.pad(x, (1, 1, 1, 1), mode='replicate')
    return (F.conv2d(padded, kx).squeeze(0).squeeze(0),
            F.conv2d(padded, ky).squeeze(0).squeeze(0))


def _masked_mean(field: torch.Tensor, mask: torch.Tensor) -> float:
    total = mask.sum()
    if total <= 0:
        return 0.0
    return float((field * mask).sum().item() / float(total.item()))


# ==================== the measurements ====================

@dataclass
class QualityMetrics:
    """
    The final canvas, in numbers. Every field is masked to the tissue.

    Ratios are stated against the reference that makes them readable:
    `*_ratio` fields are canvas-over-target (1.0 = same as the ground truth),
    `base_*` fields are the same measurement on the stage-2 input, so the
    tuner can score a run against what it was handed rather than against an
    absolute that depends on the image.
    """

    # --- fidelity (§5.5, masked to the subject) ---
    baseline_error: float          # ‖base - target‖ on tissue: what stage 2 achieved for free
    final_error: float             # ‖canvas - target‖ on tissue
    improvement: float             # 1 - final/baseline; < 0 means the swarm hurt the picture

    # --- detail (§3: "lo sciame mangia dettaglio") ---
    detail_ratio: float            # high-frequency RMS, canvas / target. base is well under 1
    base_detail_ratio: float       # the same for the untouched input, for reference
    gradient_alignment: float      # cosine between canvas and target gradient fields, [-1, 1]
    base_gradient_alignment: float

    # --- brushwork (§5.2, §5.4) ---
    stroke_coherence: float        # structure-tensor anisotropy of the pigment layer, [0, 1]
    coverage: float                # share of tissue actually repainted
    pigment_mass: float            # mean ‖canvas - base‖ on tissue: how much paint is on

    # --- colour (§13 "deriva cromatica") ---
    chroma_ratio: float            # mean OKLab chroma, canvas / target
    lightness_bias: float          # mean L canvas - mean L target; drift towards grey/dark

    # --- temporal (§5.9: alive, but not seething) ---
    flicker: float                 # mean per-step ‖Δcanvas‖ over the settled tail


QUALITY_FIELDS = tuple(f.name for f in dataclass_fields(QualityMetrics))


def _detail_ratio(canvas_l: torch.Tensor, target_l: torch.Tensor, mask: torch.Tensor,
                  sigma: float) -> float:
    """
    RMS of the high-pass band, canvas over target.

    The band is `x - blur(x)`, i.e. everything the NCA's downsample-upsample
    threw away. 1.0 means the canvas carries as much fine structure as the
    ground truth; the stage-2 input sits far below, and a noisy run overshoots.
    """
    hf_canvas = canvas_l - gaussian_blur(canvas_l, sigma)
    hf_target = target_l - gaussian_blur(target_l, sigma)
    energy_canvas = _masked_mean(hf_canvas * hf_canvas, mask)
    energy_target = _masked_mean(hf_target * hf_target, mask)
    return math.sqrt(energy_canvas / max(energy_target, _EPS))


def _gradient_alignment(canvas_l: torch.Tensor, target_l: torch.Tensor,
                        mask: torch.Tensor) -> float:
    """
    Global cosine between the two gradient fields.

    Detail in the right places correlates with the target's own edges; detail
    sprayed at random does not. Paired with `detail_ratio` this separates
    "restored the texture" from "added noise", which no single scalar can.
    """
    gcx, gcy = gradients(canvas_l)
    gtx, gty = gradients(target_l)
    dot = _masked_mean(gcx * gtx + gcy * gty, mask)
    norm_c = math.sqrt(max(_masked_mean(gcx * gcx + gcy * gcy, mask), _EPS))
    norm_t = math.sqrt(max(_masked_mean(gtx * gtx + gty * gty, mask), _EPS))
    return dot / (norm_c * norm_t)


def _stroke_coherence(pigment: torch.Tensor, mask: torch.Tensor, sigma: float = 1.6) -> float:
    """
    Structure-tensor anisotropy of the pigment layer, energy-weighted.

    For the smoothed tensor J = [[Jxx, Jxy], [Jxy, Jyy]], the eigenvalue
    contrast `(λ1 - λ2) / (λ1 + λ2)` is 1 where the layer varies along one
    direction only -- a stroke -- and 0 where it varies equally in all of them
    -- a blob, or per-pixel noise. Weighting by `λ1 + λ2` keeps flat, unpainted
    regions from voting.
    """
    gx, gy = gradients(pigment)
    jxx = gaussian_blur(gx * gx, sigma)
    jyy = gaussian_blur(gy * gy, sigma)
    jxy = gaussian_blur(gx * gy, sigma)

    trace = jxx + jyy
    contrast = torch.sqrt(((jxx - jyy) ** 2 + 4.0 * jxy * jxy).clamp(min=0.0))
    coherence = contrast / trace.clamp(min=_EPS)

    weight = trace * mask
    total = weight.sum()
    if total <= 0:
        return 0.0
    return float(((coherence * weight).sum() / total).item())


def _tail_flicker(history: List[StepMetrics], tail_fraction: float = 0.25) -> float:
    """Mean per-step canvas change over the settled tail (§5.9's λ_min reading)."""
    if not history:
        return 0.0
    values = [m.canvas_delta for m in history if m.canvas_delta is not None]
    if not values:
        return 0.0
    start = max(0, int(len(values) * (1.0 - tail_fraction)))
    tail = values[start:] or values[-1:]
    return sum(tail) / len(tail)


def measure(canvas: torch.Tensor, base: torch.Tensor, target: torch.Tensor,
            mask: torch.Tensor, history: List[StepMetrics] = None,
            detail_sigma: float = 1.5, coverage_threshold: float = 0.01) -> QualityMetrics:
    """
    Measure a finished canvas against the input it was handed and the target.

    All tensors are OKLab (H, W, 3) except `mask`, an (H, W) boolean of where
    the tissue is. `history` is the §12 per-step record, used only for the
    temporal reading; pass None to leave `flicker` at zero.
    """
    mask_f = mask.to(canvas.dtype)

    err_base = (base - target).norm(dim=-1)
    err_canvas = (canvas - target).norm(dim=-1)
    baseline_error = _masked_mean(err_base, mask_f)
    final_error = _masked_mean(err_canvas, mask_f)

    canvas_l, base_l, target_l = canvas[..., 0], base[..., 0], target[..., 0]
    pigment = (canvas - base).norm(dim=-1)

    chroma_canvas = canvas[..., 1:].norm(dim=-1)
    chroma_target = target[..., 1:].norm(dim=-1)

    return QualityMetrics(
        baseline_error=baseline_error,
        final_error=final_error,
        improvement=1.0 - final_error / max(baseline_error, _EPS),

        detail_ratio=_detail_ratio(canvas_l, target_l, mask_f, detail_sigma),
        base_detail_ratio=_detail_ratio(base_l, target_l, mask_f, detail_sigma),
        gradient_alignment=_gradient_alignment(canvas_l, target_l, mask_f),
        base_gradient_alignment=_gradient_alignment(base_l, target_l, mask_f),

        stroke_coherence=_stroke_coherence(pigment, mask_f),
        coverage=_masked_mean((pigment > coverage_threshold).to(canvas.dtype), mask_f),
        pigment_mass=_masked_mean(pigment, mask_f),

        chroma_ratio=_masked_mean(chroma_canvas, mask_f) / max(_masked_mean(chroma_target, mask_f), _EPS),
        lightness_bias=_masked_mean(canvas_l, mask_f) - _masked_mean(target_l, mask_f),

        flicker=_tail_flicker(history or []),
    )


def measure_simulation(sim, history: Optional[List[StepMetrics]] = None) -> QualityMetrics:
    """`measure` applied to a live `Simulation`, with the tissue mask it already holds."""
    mask = sim.fields.nutrient > sim.config.nutrient_threshold
    return measure(sim.fields.canvas, sim.fields.base, sim.fields.target, mask,
                   history if history is not None else sim.history)


def quality_to_dict(quality: QualityMetrics) -> Dict[str, float]:
    return {name: getattr(quality, name) for name in QUALITY_FIELDS}
