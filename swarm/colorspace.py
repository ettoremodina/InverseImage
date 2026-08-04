"""
sRGB <-> OKLab conversion, vectorized in torch.

The whole swarm lives in OKLab (Evolutionary_Swarm.md §10): genes, the canvas,
the error metric and the pigment decay all measure distance in this space, so
`mutation_sigma` and `tolerance` mean the same thing everywhere in colour, and
there is exactly one conversion per frame instead of one per agent per step.

Reference: Bjoern Ottosson, https://bottosson.github.io/posts/oklab/
"""

import torch

# ==================== sRGB <-> linear ====================

def srgb_to_linear(c: torch.Tensor) -> torch.Tensor:
    """Inverse gamma. Input and output in [0, 1]."""
    c = c.clamp(0.0, 1.0)
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: torch.Tensor) -> torch.Tensor:
    """Gamma. Input assumed non-negative; output clamped to [0, 1]."""
    c = c.clamp(min=0.0)
    srgb = torch.where(c <= 0.0031308, c * 12.92, 1.055 * c.clamp(min=1e-8) ** (1.0 / 2.4) - 0.055)
    return srgb.clamp(0.0, 1.0)


# ==================== linear sRGB <-> OKLab ====================
# Matrices from Ottosson's reference implementation. Operate on the last
# dimension (size 3); any leading shape is broadcast through.

_M1 = torch.tensor([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])

_M2 = torch.tensor([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
])

_M1_INV = torch.linalg.inv(_M1)
_M2_INV = torch.linalg.inv(_M2)


def linear_to_oklab(rgb: torch.Tensor) -> torch.Tensor:
    """(..., 3) linear sRGB, non-negative -> (..., 3) OKLab."""
    m1 = _M1.to(rgb.device, rgb.dtype)
    m2 = _M2.to(rgb.device, rgb.dtype)
    lms = rgb.clamp(min=0.0) @ m1.T
    lms_ = torch.sign(lms) * lms.abs().clamp(min=1e-12) ** (1.0 / 3.0)
    return lms_ @ m2.T


def oklab_to_linear(lab: torch.Tensor) -> torch.Tensor:
    """(..., 3) OKLab -> (..., 3) linear sRGB. Not gamut-clamped."""
    m1_inv = _M1_INV.to(lab.device, lab.dtype)
    m2_inv = _M2_INV.to(lab.device, lab.dtype)
    lms_ = lab @ m2_inv.T
    lms = lms_ ** 3
    return lms @ m1_inv.T


# ==================== sRGB <-> OKLab, end to end ====================

def srgb_to_oklab(srgb: torch.Tensor) -> torch.Tensor:
    """(..., 3) sRGB in [0, 1] -> (..., 3) OKLab."""
    return linear_to_oklab(srgb_to_linear(srgb))


def oklab_to_srgb(lab: torch.Tensor) -> torch.Tensor:
    """(..., 3) OKLab -> (..., 3) sRGB in [0, 1], clamped to gamut."""
    return linear_to_srgb(oklab_to_linear(lab))


def clamp_gamut(lab: torch.Tensor) -> torch.Tensor:
    """
    Snap an OKLab colour back to the sRGB gamut.

    OKLab contains coordinates with no sRGB equivalent (Evolutionary_Swarm.md
    §10); without this, mutation drifts genes into imaginary colours that read
    as free error reduction ("clamp_gamut" the doc calls for explicitly).
    Round-trips through sRGB, which is the gamut boundary we actually care
    about here.
    """
    return srgb_to_oklab(oklab_to_srgb(lab))
