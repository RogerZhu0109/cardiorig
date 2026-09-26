import math

import torch

from src.data.augment import apply_similarity, random_similarity
from src.utils.svf import integrate, warp


def smooth(n, h, scale, seed):
    g = torch.Generator().manual_seed(seed)
    return scale * torch.nn.functional.interpolate(torch.randn(n, h, 6, 6, generator=g), size=(64, 64),
                                                   mode="bicubic", align_corners=True)


def test_transformed_pair_is_consistent():
    """warp(T M, exp(T_* v)) == T warp(M, exp(v)) in the interior."""
    img, v = smooth(4, 1, 1.0, 0), smooth(4, 2, 3.0, 1)
    A, b = random_similarity(4, 64, rot_deg=30, scale=0.1, shift_px=4, generator=torch.Generator().manual_seed(2))
    img_t, v_t = apply_similarity(img, v, A, b)
    lhs = warp(img_t, integrate(v_t))
    rhs, _ = apply_similarity(warp(img, integrate(v)), torch.zeros_like(v), A, b)
    inner = (slice(None), slice(None), slice(16, 48), slice(16, 48))
    err = (lhs - rhs)[inner].abs().mean() / img[inner].abs().mean()
    assert err < 0.02, err


def test_rotation_turns_vectors():
    """A uniform field rotated by 90 degrees points the other way."""
    v = torch.zeros(1, 2, 32, 32)
    v[:, 0] = 1.0  # +x everywhere
    ang = math.pi / 2
    A = torch.tensor([[[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]]])
    _, v_t = apply_similarity(torch.zeros(1, 1, 32, 32), v, A, torch.zeros(1, 2))
    assert torch.allclose(v_t[0, :, 16, 16], torch.tensor([0.0, 1.0]), atol=1e-5)
