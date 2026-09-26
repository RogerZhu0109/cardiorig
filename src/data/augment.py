"""
Augmentation of (ED slice, SVF) training pairs, on device.

A random similarity transform T(q) = A (q - b) about the image centre (rotation,
isotropic scale, shift) is applied consistently: M'(p) = M(A^-1 p + b), and the SVF is
pushed forward, v'(p) = A v(A^-1 p + b). Then exp(v') = T o exp(v) o T^-1, so
M' o exp(v') is the transformed target and the pair stays an exact training example.
No flips: they would reverse the direction of LV twist. Intensity augmentation
(gamma, contrast, brightness, noise) acts on the ED image only. CineMA feature maps, if
given, are resampled with the same transform (an approximation: ViT features are not exactly
equivariant); the cls token is left unchanged.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def random_similarity(n: int, size: int, rot_deg: float, scale: float, shift_px: float,
                      generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """A (n, 2, 2) and b (n, 2) in normalised coordinates (align_corners=True), x first."""
    u = torch.rand(n, 4, generator=generator) * 2 - 1
    ang, s = u[:, 0] * math.radians(rot_deg), 1 + u[:, 1] * scale
    c, si = torch.cos(ang), torch.sin(ang)
    A = s[:, None, None] * torch.stack([torch.stack([c, -si], -1), torch.stack([si, c], -1)], 1)
    b = u[:, 2:] * shift_px * 2 / (size - 1)
    return A, b


def apply_similarity(ed: torch.Tensor, v: torch.Tensor, A: torch.Tensor, b: torch.Tensor, feat=None):
    A, b = A.to(ed.device), b.to(ed.device)
    theta = torch.cat([torch.linalg.inv(A), b[:, :, None]], dim=2)
    grid = F.affine_grid(theta, list(ed.shape), align_corners=True)
    ed = F.grid_sample(ed, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    v = F.grid_sample(v, grid, mode="bilinear", padding_mode="border", align_corners=True)
    if feat is not None:
        g = F.affine_grid(theta, list(feat.shape), align_corners=False)  # same field of view, coarse grid
        feat = F.grid_sample(feat, g, mode="bilinear", padding_mode="border", align_corners=False)
        return ed, torch.einsum("bij,bjhw->bihw", A, v), feat
    return ed, torch.einsum("bij,bjhw->bihw", A, v)


def intensity(ed: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    u = (torch.rand(len(ed), 4, generator=generator) * 2 - 1).to(ed.device)[:, :, None, None, None]
    ed = ed.clamp(0, 1) ** torch.exp(0.3 * u[:, 0])  # gamma in [0.74, 1.35]
    ed = ed * (1 + 0.1 * u[:, 1]) + 0.05 * u[:, 2]  # contrast, brightness
    noise = torch.randn(ed.shape, generator=generator).to(ed.device)
    return (ed + 0.01 * (1 + u[:, 3]) * noise).clamp(0, 1)  # noise std in [0, 0.02]


def augment(ed, v, generator=None, rot_deg=30.0, scale=0.1, shift_px=8.0, feat=None):
    """Returns (ed, v) or, with feat, (ed, v, feat)."""
    A, b = random_similarity(len(ed), ed.shape[-1], rot_deg, scale, shift_px, generator)
    out = apply_similarity(ed, v, A, b, feat)
    return (intensity(out[0], generator), *out[1:])
