"""
Dense registration ("teacher"): per-pair SVF optimisation, ED slice -> every frame.

    min_v  -LNCC(ED o exp(v/2), I_t o exp(-v/2)) + lam * ||grad v||^2

Symmetric (midpoint) objective as in SyN; exp(v) maps ED onto I_t. v is optimised directly
(no network) coarse-to-fine with Adam, all pairs of a patient in one batch. Pairs are
independent: each v only enters its own loss term. exp() is the scaling and squaring of
src/utils/svf.py, which the models also use.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.utils.svf import integrate, warp


def lncc(a: torch.Tensor, b: torch.Tensor, win: int = 9, eps: float = 1e-5) -> torch.Tensor:
    """Local squared NCC per pair, (B,)."""
    def mean(x):
        return F.avg_pool2d(x, win, stride=1, padding=win // 2, count_include_pad=False)
    ma, mb = mean(a), mean(b)
    cross = mean(a * b) - ma * mb
    va, vb = mean(a * a) - ma ** 2, mean(b * b) - mb ** 2
    return (cross ** 2 / (va * vb + eps)).mean(dim=(1, 2, 3))


def diffusion(v: torch.Tensor) -> torch.Tensor:
    """Mean squared forward difference of v, (B,)."""
    dy = (v[..., 1:, :] - v[..., :-1, :]).pow(2).mean(dim=(1, 2, 3))
    dx = (v[..., :, 1:] - v[..., :, :-1]).pow(2).mean(dim=(1, 2, 3))
    return dx + dy


def register(
    moving: torch.Tensor,
    fixed: torch.Tensor,
    lam: float = 0.5,
    levels: tuple[int, ...] = (4, 2, 1),
    iters: int = 150,
    lr: float = 0.1,
    win: int = 21,
) -> tuple[torch.Tensor, dict]:
    """
    moving, fixed: (B, 1, H, W).  Returns the SVF v (B, 2, H, W) in full-resolution
    pixel units such that warp(moving, integrate(v)) ~ fixed, and a loss log.
    The field Jacobian is scale-invariant, so lam means the same at every level.
    """
    v, log = None, {}
    for s in levels:
        m = F.avg_pool2d(moving, s) if s > 1 else moving
        f = F.avg_pool2d(fixed, s) if s > 1 else fixed
        if v is None:
            v = torch.zeros(m.shape[0], 2, *m.shape[-2:], device=m.device)
        else:
            v = F.interpolate(v, size=m.shape[-2:], mode="bilinear", align_corners=True) * (v_scale / s)
        v_scale = s
        v = v.detach().requires_grad_(True)
        opt = torch.optim.Adam([v], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        for _ in range(iters):
            sim = lncc(warp(m, integrate(v / 2)), warp(f, integrate(-v / 2)), win)
            reg = diffusion(v)
            loss = (-sim + lam * reg).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        log[s] = dict(sim=sim.mean().item(), reg=reg.mean().item())
    return v.detach(), log


def register_batched(moving: torch.Tensor, fixed: torch.Tensor, chunk: int = 64, **kw) -> torch.Tensor:
    """register() over chunks of pairs to bound memory; pairs are independent, so results match."""
    return torch.cat([register(moving[i:i + chunk], fixed[i:i + chunk], **kw)[0] for i in range(0, len(moving), chunk)])
