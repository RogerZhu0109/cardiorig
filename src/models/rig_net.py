"""
Rig network: ED stack + ED mask -> mean and log-variance of the compact rig numbers.

A small CNN runs on every slice. Its feature map is pooled into anatomical tokens, one per LV
segment, RV sector and translation, using each token's support (src/rig.py token_maps) over the
whole stack. A transformer mixes the tokens, and heads shared across tokens of one kind output
that token's numbers, so every LV segment trains the same head.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet import ResBlock
from src.rig import RV_SECTORS, SHARED, TOKENS


class RigNet(nn.Module):
    def __init__(self, curves: int = 4, ch: tuple[int, ...] = (32, 64, 128), dim: int = 256, layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.curves = curves
        self.cls_mlp = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.feat_proj = nn.Sequential(nn.GroupNorm(1, 768), nn.Conv2d(768, ch[-1], 1))
        self.inp = nn.Conv2d(4, ch[0], 3, padding=1)
        self.blocks = nn.ModuleList(ResBlock(ch[max(i - 1, 0)], c, dim) for i, c in enumerate(ch))
        self.pool = nn.ModuleList(nn.Conv2d(c, c, 3, stride=2, padding=1) for c in ch[:-1])
        self.to_token = nn.Linear(ch[-1], dim)
        self.token_id = nn.Parameter(torch.zeros(TOKENS, dim))
        layer = nn.TransformerEncoderLayer(dim, 8, 4 * dim, dropout, "gelu", batch_first=True, norm_first=True)
        self.mixer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False) if layers else nn.Identity()
        self.norm = nn.LayerNorm(dim)
        self.lv_head = nn.Linear(dim, 3 * curves * 2)
        self.rv_head = nn.Linear(dim, 2 * curves * 2)
        self.global_head = nn.Linear(dim, 2 * curves * 2)
        for head in (self.lv_head, self.rv_head, self.global_head):
            nn.init.zeros_(head.weight), nn.init.zeros_(head.bias)

    def forward(self, x, maps, owner, feat, cls):
        """x (slices, 4, H, W), maps (slices, TOKENS, H, W), owner (slices,) patient index,
        feat (slices, 768, 12, 12), cls (patients, 768). Returns mean, log_var (patients, SHARED, curves)."""
        n = len(cls)
        e = self.cls_mlp(cls)
        slice_e = e[owner]
        h = self.inp(x)
        for i, block in enumerate(self.blocks):
            h = block(h, slice_e)
            if i < len(self.pool):
                h = self.pool[i](h)
        h = h + F.interpolate(self.feat_proj(feat), size=h.shape[-2:], mode="bilinear", align_corners=False)
        m = F.adaptive_avg_pool2d(maps, h.shape[-2:])
        pooled = torch.einsum("schw,sthw->stc", h, m)  # feature sum per slice and token
        weight = m.sum((2, 3))
        total = pooled.new_zeros(n, TOKENS, h.shape[1]).index_add(0, owner, pooled)
        mass = weight.new_zeros(n, TOKENS).index_add(0, owner, weight)
        tokens = self.to_token(total / mass.clamp_min(1e-3)[:, :, None]) + self.token_id + e[:, None]
        tokens = self.norm(self.mixer(tokens))
        out = torch.cat([self.global_head(tokens[:, -1:]).flatten(1),
                         self.lv_head(tokens[:, :16]).flatten(1),
                         self.rv_head(tokens[:, 16:16 + RV_SECTORS]).flatten(1)], 1)
        # motions are ordered per token here, in the same order as the slots of src/rig.py
        out = out.reshape(n, SHARED, self.curves, 2)
        return out[..., 0], out[..., 1].clamp(-8, 4)


def beta_nll(mean, log_var, target, weight, beta: float = 0.5):
    """Gaussian negative log-likelihood, each number weighted by its detached variance ** beta."""
    var = log_var.exp()
    nll = 0.5 * (log_var + (target - mean) ** 2 / var)
    return (weight * nll * var.detach() ** beta).sum() / weight.sum().clamp_min(1)
