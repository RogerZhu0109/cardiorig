import numpy as np
import torch

from scripts.evaluate_published import curve_metrics
from scripts.train_krebs import slots_of
from src.data.cache import ordered, resampled
from src.models.krebs import LATENT, T_MAX, Krebs, cauchy_kernel, gaussian_kernel, kl_divergence, smooth
from src.models.phaseflow import T, DeformDecoder, VelocityUNet, bending, nonlinear_phase


def test_ordered_puts_ed_first():
    tau = np.array([0.5, 0.75, 0.0, 0.25])
    assert ordered(np.arange(4), tau).tolist() == [2, 3, 0, 1]


def test_resampled_keeps_ends_and_values():
    x = np.arange(5, dtype=np.float32)
    out = resampled(x, 9)
    assert out.tolist() == [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4]


def test_nonlinear_phase_is_monotone_and_full_cycle():
    p = nonlinear_phase(np.array([10.0, 7, 4, 4, 7, 10]))
    assert p[0] == 0 and np.isclose(p[-1], 2 * np.pi) and np.all(np.diff(p) >= 0)
    assert np.isclose(p[2], np.pi)  # half the area change is done at ES


def test_kl_is_zero_at_prior_and_positive_elsewhere():
    inv_k = torch.linalg.inv(cauchy_kernel())
    zero = torch.zeros(1, LATENT)
    assert kl_divergence(torch.zeros(1, LATENT, T_MAX), zero, inv_k).abs().item() < 1e-5
    assert kl_divergence(torch.ones(1, LATENT, T_MAX), zero, inv_k).item() > 0
    assert kl_divergence(torch.zeros(1, LATENT, T_MAX), zero + 0.5, inv_k).item() > 0


def test_smooth_keeps_constant_fields():
    v = torch.ones(6, 2, 32, 32) * 3
    out = smooth(v, gaussian_kernel(2.0), gaussian_kernel(1.5))
    assert torch.allclose(out, v, atol=1e-4)


def test_slots_cover_the_latent_columns():
    s = slots_of(20)
    assert s[0] == 0 and s[-1] == T_MAX - 1 and len(set(s.tolist())) == 20


def test_simulation_ignores_frames_and_depends_on_i0():
    m = Krebs().eval()
    i0 = torch.rand(2, 1, 128, 128)
    assert m.simulate(i0).shape == (2, T_MAX, 2, 128, 128)
    assert not torch.allclose(m.simulate(i0)[0], m.simulate(i0)[1])


def test_phaseflow_output_shapes_and_zero_start():
    net, dec = VelocityUNet(ch=(16, 32), emb=32, token=16), DeformDecoder()
    z = torch.zeros(1, 4, T, 16, 16)
    v = net(z, torch.ones(1), torch.zeros(1, T), torch.zeros(1))
    assert v.shape == z.shape and v.abs().max() == 0
    assert dec(v).shape == (1, T, 2, 128, 128)


def test_bending_vanishes_on_affine_fields():
    ys, xs = torch.meshgrid(torch.arange(16.0), torch.arange(16.0), indexing="ij")
    assert bending(torch.stack([xs * 0.1, ys * 0.2 + xs])[None]).abs().item() < 1e-8


def test_curve_metrics_perfect_prediction():
    v = 100 + 20 * np.cos(np.linspace(0, 6, 30))
    m = curve_metrics(v.copy(), v)
    assert m["ef_err"] == 0 and np.isclose(m["vol_corr"], 1) and m["vol_r2"] == 1 and m["vol_mae"] == 0
