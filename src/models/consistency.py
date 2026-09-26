"""
Image-consistency objective for fine-tuning the generator on real frames: the teacher's
per-pair objective (src/teacher.py), evaluated on the generator's SVF instead of optimised per pair,

    -LNCC(ED o exp(v/2), I_tau o exp(-v/2)) + lam * ||grad v||^2,   v in pixels.

Same symmetric form, window and regulariser as the teacher, so a difference between the two
comes from amortisation (one network over all pairs) and lam, not from a different similarity.
"""
from __future__ import annotations

import torch

from src.teacher import diffusion, lncc
from src.utils.svf import integrate, warp


def consistency(v: torch.Tensor, ed: torch.Tensor, target: torch.Tensor, win: int = 21):
    """Per-pair (sim, reg), each (B,); the loss is -sim + lam * reg."""
    sim = lncc(warp(ed, integrate(v / 2)), warp(target, integrate(-v / 2)), win)
    return sim, diffusion(v)
