import math

import torch
import torch.nn.functional as F

from src.utils.svf import compose, integrate, jacobian_det, warp

H = W = 64


def smooth_field(max_disp: float, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(1, 2, 8, 8, generator=g)
    v = F.interpolate(v, size=(H, W), mode="bicubic", align_corners=True)
    return v * max_disp / v.abs().max()


def interior(t: torch.Tensor, m: int = 12) -> torch.Tensor:
    return t[..., m:-m, m:-m]


def test_zero_field_is_identity():
    assert integrate(torch.zeros(1, 2, H, W)).abs().max() == 0


def test_warp_direction():
    img = torch.zeros(1, 1, H, W)
    img[0, 0, 30, 20] = 1.0
    disp = torch.zeros(1, 2, H, W)
    disp[:, 0] = 3.0  # sample at x + 3 -> content moves left by 3
    out = warp(img, disp)
    assert out[0, 0, 30, 17] == 1.0 and out.sum() == 1.0


def test_constant_field_is_translation():
    v = torch.zeros(1, 2, H, W)
    v[:, 0], v[:, 1] = 2.5, -1.5
    assert torch.allclose(integrate(v), v, atol=1e-5)


def test_rotation_field():
    # v(p) = w * J (p - c) is the generator of a rotation by angle w about c.
    w = 0.3
    ys, xs = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    c = (W - 1) / 2
    dx, dy = xs - c, ys - c
    v = torch.stack([-w * dy, w * dx])[None]
    rot = torch.stack([
        math.cos(w) * dx - math.sin(w) * dy - dx,
        math.sin(w) * dx + math.cos(w) * dy - dy,
    ])[None]
    err = interior(integrate(v, steps=10) - rot, m=16).abs().max()
    assert err < 0.05, err


def test_inverse_is_negative_field():
    v = smooth_field(4.0)
    residual = compose(integrate(v), integrate(-v))
    assert interior(residual).abs().max() < 0.05


def test_no_folding_for_large_smooth_field():
    v = smooth_field(8.0, seed=1)
    u = integrate(v)
    assert u.abs().max() > 4  # the deformation is genuinely large
    assert jacobian_det(u).min() > 0


def test_translation_jacobian_is_one():
    v = torch.full((1, 2, H, W), 1.7)
    assert torch.allclose(jacobian_det(integrate(v)), torch.ones(1, H, W), atol=1e-5)
