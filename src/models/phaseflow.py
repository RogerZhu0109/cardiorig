"""
Reimplementation of PhaseFlow (Wang et al., arXiv 2610.09397) from the paper; no public code.
ED frame -> frozen VAE latent -> phase-conditioned rectified flow -> frame-wise deformation decoder
-> SVFs that warp the ED frame. T = 30 frames, 128 x 128, latent (4, T, 16, 16). Sections cited
in comments refer to the paper.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet import time_embedding

T, ALPHA, LATENT = 30, 0.1, 4


class SegUNet(nn.Module):
    """Plain 2D U-Net for the per-fold LV areas of Sec. 3.2 (4 classes)."""

    def __init__(self, ch: tuple[int, ...] = (32, 64, 128, 256)):
        super().__init__()

        def block(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU(),
                                 nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU())
        self.down = nn.ModuleList(block(a, b) for a, b in zip((1, *ch[:-1]), ch))
        self.up = nn.ModuleList(block(ch[i] + ch[i + 1], ch[i]) for i in range(len(ch) - 1))
        self.out = nn.Conv2d(ch[0], 4, 1)

    def forward(self, x):
        skips = []
        for i, b in enumerate(self.down):
            x = b(x if i == 0 else F.avg_pool2d(x, 2))
            skips.append(x)
        for i in reversed(range(len(self.up))):
            x = self.up[i](torch.cat([skips[i], F.interpolate(x, scale_factor=2, mode="bilinear")], 1))
        return self.out(x)


def nonlinear_phase(area: np.ndarray) -> np.ndarray:
    """Eq. 1: cumulative |area change|, normalised to [0, 2 pi] over the frames; linear if the area is flat."""
    step = np.abs(np.diff(area))
    if step.sum() < 1e-6:
        return 2 * np.pi * np.arange(len(area)) / (len(area) - 1)
    return 2 * np.pi * np.r_[0, np.cumsum(step)] / step.sum()


class Res3D(nn.Module):
    """3D residual block; with `emb`, AdaGN scale/shift from the global condition."""

    def __init__(self, cin: int, cout: int, emb: int = 0):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv3d(cin, cout, 3, padding=1)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv3d(cout, cout, 3, padding=1)
        self.ada = nn.Linear(emb, 2 * cout) if emb else None
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e=None):
        h = self.n2(self.c1(F.silu(self.n1(x))))
        if self.ada is not None:
            scale, shift = self.ada(e)[:, :, None, None, None].chunk(2, 1)
            h = h * (1 + scale) + shift
        return self.skip(x) + self.c2(F.silu(h))


class FrameAttention(nn.Module):
    """2D self-attention inside each frame."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm, self.attn = nn.GroupNorm(8, ch), nn.MultiheadAttention(ch, 8, batch_first=True)

    def forward(self, x):
        b, c, t, h, w = x.shape
        q = self.norm(x).permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c)
        return x + self.attn(q, q, q, need_weights=False)[0].reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)


class VAE3D(nn.Module):
    """Spatial 8x compression, time preserved, per-frame 2D attention at the bottleneck (Appendix E).
    Smaller than the source (64 base channels) and L1 + KL only: this model supplies latents, not pixels."""

    def __init__(self, ch: tuple[int, ...] = (64, 128, 256, 256)):
        super().__init__()
        self.inp = nn.Conv3d(1, ch[0], 3, padding=1)
        self.enc = nn.ModuleList(Res3D(a, b) for a, b in zip((ch[0], *ch[:-1]), ch))
        self.down = nn.ModuleList(nn.Conv3d(c, c, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)) for c in ch[:-1])
        self.enc_attn, self.dec_attn = FrameAttention(ch[-1]), FrameAttention(ch[-1])
        self.to_z = nn.Sequential(nn.GroupNorm(8, ch[-1]), nn.SiLU(), nn.Conv3d(ch[-1], 2 * LATENT, 1))
        self.from_z = nn.Conv3d(LATENT, ch[-1], 1)
        self.dec = nn.ModuleList(Res3D(ch[i + 1], ch[i]) for i in reversed(range(len(ch) - 1)))
        self.out = nn.Sequential(nn.GroupNorm(8, ch[0]), nn.SiLU(), nn.Conv3d(ch[0], 1, 3, padding=1))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.inp(x)
        for i, block in enumerate(self.enc):
            h = block(h)
            if i < len(self.down):
                h = self.down[i](h)
        return self.to_z(self.enc_attn(h)).chunk(2, 1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_attn(self.from_z(z))
        for block in self.dec:
            h = block(F.interpolate(h, scale_factor=(1, 2, 2)))
        return self.out(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        kl = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).mean()
        return self.decode(z), kl


class Attention(nn.Module):
    """Temporal self-attention across frames, then cross-attention of each frame to its phase tokens."""

    def __init__(self, ch: int, token: int):
        super().__init__()
        self.n1, self.temporal = nn.GroupNorm(8, ch), nn.MultiheadAttention(ch, 8, batch_first=True)
        self.n2, self.cross = nn.GroupNorm(8, ch), nn.MultiheadAttention(ch, 8, kdim=token, vdim=token, batch_first=True)

    def forward(self, x, tokens):
        b, c, t, h, w = x.shape
        q = self.n1(x).permute(0, 3, 4, 2, 1).reshape(b * h * w, t, c)
        x = x + self.temporal(q, q, q, need_weights=False)[0].reshape(b, h, w, t, c).permute(0, 4, 3, 1, 2)
        q = self.n2(x).permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c)
        k = tokens.reshape(b * t, *tokens.shape[2:])
        return x + self.cross(q, k, k, need_weights=False)[0].reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)


class VelocityUNet(nn.Module):
    """3D U-Net, channels [128, 256] (Appendix E). tau and slice position go through AdaGN,
    per-frame phase through cross-attention with 4 tokens."""

    def __init__(self, ch: tuple[int, int] = (128, 256), emb: int = 512, token: int = 256, tokens: int = 4):
        super().__init__()
        self.emb, self.n_tokens, self.token = emb, tokens, token
        self.tau_mlp = nn.Sequential(nn.Linear(emb, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.slice_mlp = nn.Sequential(nn.Linear(emb, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.phase = nn.Sequential(nn.Linear(2, token), nn.SiLU(), nn.Linear(token, tokens * token))
        self.inp = nn.Conv3d(LATENT, ch[0], 3, padding=1)
        self.e1 = nn.ModuleList([Res3D(ch[0], ch[0], emb), Res3D(ch[0], ch[0], emb)])
        self.a1 = Attention(ch[0], token)
        self.down = nn.Conv3d(ch[0], ch[1], (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.e2 = nn.ModuleList([Res3D(ch[1], ch[1], emb), Res3D(ch[1], ch[1], emb)])
        self.a2 = Attention(ch[1], token)
        self.up = nn.Conv3d(ch[1], ch[0], 3, padding=1)
        self.d1 = nn.ModuleList([Res3D(2 * ch[0], ch[0], emb), Res3D(ch[0], ch[0], emb)])
        self.a3 = Attention(ch[0], token)
        self.out = nn.Sequential(nn.GroupNorm(8, ch[0]), nn.SiLU(), nn.Conv3d(ch[0], LATENT, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight), nn.init.zeros_(self.out[-1].bias)

    def forward(self, z, tau, phase, slice_pos):
        """z (B, 4, T, 16, 16); tau, slice_pos (B,); phase (B, T) in radians."""
        e = self.tau_mlp(time_embedding(tau, self.emb)) + self.slice_mlp(time_embedding(slice_pos, self.emb))
        tokens = self.phase(torch.stack([phase.sin(), phase.cos()], -1)).reshape(*phase.shape, self.n_tokens, self.token)
        h = self.inp(z)
        for b in self.e1:
            h = b(h, e)
        skip = h = self.a1(h, tokens)
        h = self.down(h)
        for b in self.e2:
            h = b(h, e)
        h = self.up(F.interpolate(self.a2(h, tokens), scale_factor=(1, 2, 2)))
        for i, b in enumerate(self.d1):
            h = b(torch.cat([h, skip], 1) if i == 0 else h, e)
        return self.out(self.a3(h, tokens))


class DeformDecoder(nn.Module):
    """Frame-wise transposed convolutions, latent velocity -> 2-channel SVF in pixels (Sec. 3.4)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.ConvTranspose2d(LATENT, 64, 4, 2, 1), nn.SiLU(), nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(),
                                 nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.SiLU(), nn.Conv2d(16, 2, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight), nn.init.zeros_(self.net[-1].bias)
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, v):
        """v (B, 4, T, h, w) -> SVFs (B, T, 2, H, W)."""
        b, c, t, h, w = v.shape
        u = self.net(v.transpose(1, 2).reshape(b * t, c, h, w)) * self.scale
        return u.reshape(b, t, *u.shape[1:])


def bending(u: torch.Tensor) -> torch.Tensor:
    """Eq. 8, second-order finite differences of displacements (N, 2, H, W)."""
    dxx = u[..., :, 2:] - 2 * u[..., :, 1:-1] + u[..., :, :-2]
    dyy = u[..., 2:, :] - 2 * u[..., 1:-1, :] + u[..., :-2, :]
    dxy = u[..., 1:, 1:] - u[..., 1:, :-1] - u[..., :-1, 1:] + u[..., :-1, :-1]
    return dxx.pow(2).mean() + dyy.pow(2).mean() + 2 * dxy.pow(2).mean()
