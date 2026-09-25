"""
Generator objectives and sampling in normalised SVF space, w = v / sigma.

fm       conditional flow matching with the linear (rectified-flow) path
         w_t = (1 - t) w0 + t w1, w0 ~ N(0, I);  u(w_t, t, c) regresses w1 - w0.
         Sampling: Euler from t = 0 to 1 in `steps` network evaluations (NFE).
regress  same network with w = 0, t = 0, regressing w1 directly (deterministic
         baseline). With the optimal field, one Euler step of fm from any w0 gives
         E[w1 | c], the regression target; fm differs only through multi-step samples.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def loss(model, objective: str, ed, tau, w1):
    b = len(w1)
    if objective == "regress":
        zero = torch.zeros(b, device=w1.device)
        return F.mse_loss(model(torch.zeros_like(w1), zero, ed, tau), w1)
    w0 = torch.randn_like(w1)
    t = torch.rand(b, device=w1.device)
    wt = (1 - t)[:, None, None, None] * w0 + t[:, None, None, None] * w1
    return F.mse_loss(model(wt, t, ed, tau), w1 - w0)


@torch.no_grad()
def sample(model, objective: str, ed, tau, sigma: float, steps: int = 4, generator: torch.Generator | None = None):
    """SVF v (B, 2, H, W) in pixels. Noise is drawn on CPU from `generator` so seeds match across devices."""
    b, (h, w) = len(ed), ed.shape[-2:]
    if objective == "regress":
        zero = torch.zeros(b, device=ed.device)
        return sigma * model(torch.zeros(b, 2, h, w, device=ed.device), zero, ed, tau)
    x = torch.randn(b, 2, h, w, generator=generator).to(ed.device)
    for i in range(steps):
        t = torch.full((b,), i / steps, device=ed.device)
        x = x + model(x, t, ed, tau) / steps
    return sigma * x
