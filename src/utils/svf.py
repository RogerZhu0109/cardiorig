"""
Stationary velocity field (SVF) utilities for 2D deformations.

Convention: fields are (B, 2, H, W) displacements in pixel units, channel 0 = x
(column), channel 1 = y (row).  A displacement u defines phi(p) = p + u(p), and
warp(img, u) = img o phi, i.e. the output at p samples img at p + u(p).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _identity(h: int, w: int, like: torch.Tensor) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(h, device=like.device, dtype=like.dtype),
        torch.arange(w, device=like.device, dtype=like.dtype),
        indexing="ij",
    )
    return torch.stack([xs, ys])[None]  # (1, 2, H, W)


def warp(img: torch.Tensor, disp: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Sample img (B, C, H, W) at p + disp(p); border padding."""
    h, w = disp.shape[-2:]
    loc = _identity(h, w, disp) + disp
    grid = torch.stack([2 * loc[:, 0] / (w - 1) - 1, 2 * loc[:, 1] / (h - 1) - 1], dim=-1)
    return F.grid_sample(img, grid, mode=mode, padding_mode="border", align_corners=True)


def compose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Displacement of phi_a o phi_b: p -> p + b(p) + a(p + b(p))."""
    return b + warp(a, b)


def integrate(v: torch.Tensor, steps: int = 7) -> torch.Tensor:
    """exp(v) by scaling and squaring; returns the displacement of phi = exp(v)."""
    u = v / (2 ** steps)
    for _ in range(steps):
        u = compose(u, u)
    return u


def jacobian_det(disp: torch.Tensor) -> torch.Tensor:
    """det(D phi) of phi = id + disp, (B, H, W), central differences."""
    dux_dy, dux_dx = torch.gradient(disp[:, 0], dim=(-2, -1))
    duy_dy, duy_dx = torch.gradient(disp[:, 1], dim=(-2, -1))
    return (1 + dux_dx) * (1 + duy_dy) - dux_dy * duy_dx
