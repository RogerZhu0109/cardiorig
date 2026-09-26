"""
Conditional U-Net u(w, t, ed, tau) -> (B, 2, H, W).

The noisy SVF w (2 ch) and the ED slice (1 ch) are concatenated at the input.
Flow time t (sinusoidal) and cardiac phase tau (cyclic Fourier features, so
tau = 0.98 sits next to tau = 0) are embedded, summed, and enter every residual
block as a FiLM scale/shift.
Optional (cond=True): frozen CineMA features of the ED stack, a per-slice (768, 12, 12) map
added at the bottleneck and a global cls token added to the embedding; both projections are
zero-initialised, so training starts from the unconditioned network.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    x = 1000.0 * t[:, None] * freqs[None]
    return torch.cat([x.sin(), x.cos()], dim=1)


def phase_features(tau: torch.Tensor, n_freq: int) -> torch.Tensor:
    x = 2 * math.pi * tau[:, None] * torch.arange(1, n_freq + 1, device=tau.device)[None]
    return torch.cat([x.sin(), x.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, emb: int):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.film = nn.Linear(emb, 2 * cout)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        nn.init.zeros_(self.c2.weight), nn.init.zeros_(self.c2.bias)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.film(e)[:, :, None, None].chunk(2, dim=1)
        h = self.c2(F.silu(self.n2(h) * (1 + scale) + shift))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, ch: tuple[int, ...] = (32, 64, 128, 256), emb: int = 256, n_freq: int = 4, cond: bool = False):
        super().__init__()
        self.emb, self.n_freq, self.cond = emb, n_freq, cond
        if cond:
            self.feat_proj = nn.Sequential(nn.GroupNorm(1, 768), nn.Conv2d(768, ch[-1], 1))
            self.cls_mlp = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, emb), nn.SiLU(), nn.Linear(emb, emb))
            for m in (self.feat_proj[-1], self.cls_mlp[-1]):
                nn.init.zeros_(m.weight), nn.init.zeros_(m.bias)
        self.t_mlp = nn.Sequential(nn.Linear(emb, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.tau_mlp = nn.Sequential(nn.Linear(2 * n_freq, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.inp = nn.Conv2d(3, ch[0], 3, padding=1)
        last = len(ch) - 1
        self.down = nn.ModuleList(ResBlock(ch[max(i - 1, 0)], c, emb) for i, c in enumerate(ch))
        self.pool = nn.ModuleList(nn.Conv2d(c, c, 3, stride=2, padding=1) for c in ch[:-1])
        self.mid = nn.ModuleList([ResBlock(ch[-1], ch[-1], emb), ResBlock(ch[-1], ch[-1], emb)])
        self.unpool = nn.ModuleList(nn.Conv2d(ch[i + 1], ch[i + 1], 3, padding=1) for i in range(last))
        self.up = nn.ModuleList(ResBlock((ch[i + 1] if i < last else ch[i]) + ch[i], ch[i], emb) for i in range(len(ch)))
        self.out = nn.Sequential(nn.GroupNorm(8, ch[0]), nn.SiLU(), nn.Conv2d(ch[0], 2, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight), nn.init.zeros_(self.out[-1].bias)

    def forward(self, w, t, ed, tau, feat=None, cls=None):
        e = self.t_mlp(time_embedding(t, self.emb)) + self.tau_mlp(phase_features(tau, self.n_freq))
        if self.cond:
            e = e + self.cls_mlp(cls)
        h, skips = self.inp(torch.cat([w, ed], dim=1)), []
        for i, block in enumerate(self.down):
            h = block(h, e)
            skips.append(h)
            if i < len(self.pool):
                h = self.pool[i](h)
        if self.cond:
            h = h + F.interpolate(self.feat_proj(feat), size=h.shape[-2:], mode="bilinear", align_corners=False)
        for block in self.mid:
            h = block(h, e)
        for i in reversed(range(len(self.up))):
            if i < len(self.unpool):
                h = self.unpool[i](F.interpolate(h, scale_factor=2, mode="nearest"))
            h = self.up[i](torch.cat([h, skips[i]], dim=1), e)
        return self.out(h)
