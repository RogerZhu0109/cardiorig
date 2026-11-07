"""
Reimplementation of the temporal CVAE of Krebs et al., TMI 2021 (arXiv 2011.01741), from the
paper; no public code. Frame pairs (I0, It) -> motion matrix z (D x T) under a Cauchy-kernel GP
prior -> SVFs conditioned on I0. ED-only simulation sets every frame feature to zero.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

T_MAX, LATENT = 35, 32
SIGMA_L, SIGMA_K, LENGTH = 0.0045, 1.005, 7.0
OUTPUT_PX = 16.0  # the paper gives a tanh output but not its scale
SMOOTH_PX, SMOOTH_T = 2.0, 1.5  # 3 mm at 1.5 mm/px; frames
JITTER = 1e-3  # diagonal term of the kernel, which is otherwise numerically singular


def cauchy_kernel(t: int = T_MAX, device=None) -> torch.Tensor:
    d = torch.arange(t, device=device, dtype=torch.float32)
    return SIGMA_K ** 2 / (1 + (d[:, None] - d[None]) ** 2 / LENGTH ** 2) + JITTER * torch.eye(t, device=device)


def kl_divergence(mu: torch.Tensor, log_sigma: torch.Tensor, inv_k: torch.Tensor) -> torch.Tensor:
    """KL of N(mu_d, sigma_d^2 K) from N(0, K) per dimension d, summed. mu (B, D, T), log_sigma (B, D)."""
    t = mu.shape[-1]
    return 0.5 * ((2 * log_sigma).exp() * t + torch.einsum("bdi,ij,bdj->bd", mu, inv_k, mu) - t - 2 * t * log_sigma).sum(1)


def gaussian_kernel(sigma: float) -> torch.Tensor:
    r = math.ceil(3 * sigma)
    k = (-(torch.arange(-r, r + 1, dtype=torch.float32) ** 2) / (2 * sigma ** 2)).exp()
    return k / k.sum()


def smooth(v: torch.Tensor, spatial: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
    """Gaussian smoothing of SVFs (T, 2, H, W) along t, y and x."""
    t, c, h, w = v.shape
    r = len(temporal) // 2
    x = v.permute(2, 3, 1, 0).reshape(h * w * c, 1, t)
    x = F.conv1d(F.pad(x, (r, r), mode="replicate"), temporal[None, None]).reshape(h, w, c, t).permute(3, 2, 0, 1)
    r = len(spatial) // 2
    x = F.pad(x, (r, r, r, r), mode="replicate").reshape(t * c, 1, h + 2 * r, w + 2 * r)
    x = F.conv2d(F.conv2d(x, spatial[None, None, None]), spatial[None, None, :, None])
    return x.reshape(t, c, h, w)


class TCNBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.drop, self.linear = nn.Dropout1d(dropout), nn.Conv1d(ch, ch, 1)

    def forward(self, x):
        return x + self.linear(self.drop(F.relu(self.conv(x))))


class Krebs(nn.Module):
    def __init__(self):
        super().__init__()
        act = nn.LeakyReLU(0.2)
        chain = []
        for cin, cout, stride in ((2, 16, 2), (16, 32, 2), (32, 32, 2), (32, 4, 1)):
            chain += [nn.Conv2d(cin, cout, 3, stride=stride, padding=1), act]
        self.encoder = nn.Sequential(*chain, nn.Flatten(), nn.Linear(16 * 16 * 4, 2 * LATENT))
        self.first = nn.Conv1d(2 * LATENT, 2 * LATENT, 1)
        self.blocks = nn.ModuleList(TCNBlock(2 * LATENT, d) for d in (1, 2, 4, 8))
        self.dense = nn.Linear(LATENT, 16 * 16 * 4)
        self.up = nn.ModuleList(nn.ConvTranspose2d(c, 32, 3, stride=2, padding=1, output_padding=1) for c in (5, 33, 33))
        self.conv, self.out = nn.Conv2d(33, 16, 3, padding=1), nn.Conv2d(16, 2, 3, padding=1)
        self.act = act
        self.register_buffer("kernel", cauchy_kernel(), persistent=False)
        self.register_buffer("spatial", gaussian_kernel(SMOOTH_PX), persistent=False)
        self.register_buffer("temporal", gaussian_kernel(SMOOTH_T), persistent=False)

    def posterior(self, gamma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Feature matrix (B, 2D, T) -> mean (B, D, T), log sigma (B, D)."""
        h = self.first(gamma)
        skip = 0
        for block in self.blocks:
            h = block(h)
            skip = skip + h
        return skip[:, :LATENT], skip[:, LATENT:].mean(2)

    def encode(self, i0: torch.Tensor, frames: torch.Tensor, slots: torch.Tensor, drop: float = 0.0) -> torch.Tensor:
        """Feature matrix (1, 2D, T_MAX) of one sequence: frames (N, 1, H, W) sit at slots, the rest is zero."""
        g = self.encoder(torch.cat([i0.expand(len(frames), -1, -1, -1), frames], 1))
        if drop:
            g = g * (torch.rand(len(g), 1, device=g.device) >= drop)
        return torch.zeros(1, 2 * LATENT, T_MAX, device=g.device, dtype=g.dtype).index_copy(2, slots, g.T[None])

    def sample(self, mu, log_sigma, generator=None):
        chol = torch.linalg.cholesky(self.kernel)
        eps = torch.randn(mu.shape, generator=generator).to(mu.device)
        return mu + log_sigma.exp()[:, :, None] * (eps @ chol.T)

    def decode(self, z: torch.Tensor, i0: torch.Tensor) -> torch.Tensor:
        """Motion matrix (B, D, T) and I0 (B, 1, H, W) -> smoothed SVFs (B, T, 2, H, W) in pixels."""
        b, _, t = z.shape
        i = i0.repeat_interleave(t, 0)
        h = self.act(self.dense(z.transpose(1, 2).reshape(b * t, LATENT))).reshape(b * t, 4, 16, 16)
        for up in self.up:
            h = self.act(up(torch.cat([h, F.interpolate(i, h.shape[-2:], mode="bilinear", align_corners=True)], 1)))
        v = OUTPUT_PX * torch.tanh(self.out(self.act(self.conv(torch.cat([h, i], 1)))))
        v = torch.stack([smooth(x, self.spatial, self.temporal) for x in v.reshape(b, t, *v.shape[1:])])
        return v

    @torch.no_grad()
    def simulate(self, i0: torch.Tensor, generator=None) -> torch.Tensor:
        """ED-only motion: all frame features zero. Mean latent, or a draw when a generator is given."""
        mu, log_sigma = self.posterior(torch.zeros(len(i0), 2 * LATENT, T_MAX, device=i0.device))
        return self.decode(self.sample(mu, log_sigma, generator) if generator is not None else mu, i0)
