import numpy as np
import torch
import torch.nn.functional as F

from src.rig import SHARED, basis, edit, project, rays, roi, septal_angle, sector_of_bins, shared_slots, thickening, time_basis
from src.utils.svf import integrate, warp


def toy_mask():
    ys, xs = np.mgrid[:128, :128]
    lv = np.hypot(xs - 64, ys - 64)
    rv = np.hypot(xs - 38, ys - 64)
    m = np.zeros((128, 128), dtype=np.int64)
    m[(rv < 20) & (lv >= 24)] = 1
    m[(lv >= 14) & (lv < 24)] = 2
    m[lv < 14] = 3
    return m


def test_projection_recovers_decoded_field():
    m = toy_mask()
    b = basis(m, septal_angle(m[None]), 6)
    assert len(b) == 28
    v = np.tensordot(np.random.default_rng(0).normal(size=len(b)), b, 1)
    region = roi(m)
    assert np.abs(project(b, v, region) - v)[:, region].max() < 1e-3


def test_contraction_shrinks_lv_and_thickens_wall():
    m = toy_mask()
    b = basis(m, septal_angle(m[None]), 6)
    c = np.zeros(len(b))
    c[2::3][:6], c[3::3][:6] = 3.0, 1.0  # endo, epi radial per sector: endo moves more
    v = torch.from_numpy(np.tensordot(c, b, 1)[None]).float()
    onehot = F.one_hot(torch.from_numpy(m), 4).permute(2, 0, 1)[None].float()
    pred = warp(onehot, integrate(v)).argmax(1)[0].numpy()
    assert (pred == 3).sum() < 0.8 * (m == 3).sum()
    k = rays(m)
    assert 1.0 < thickening(integrate(-v)[0].numpy(), k).mean() < 3.0


def test_sectors_cover_all_bins():
    assert set(sector_of_bins(1.0, 6)) == set(range(6)) and set(sector_of_bins(1.0, 4)) == set(range(4))


def test_shared_slots_and_curves():
    assert len(shared_slots(28, "mid")) == 28 and shared_slots(22, "apical").max() == 57
    assert len(set(shared_slots(28, "basal")) & set(shared_slots(22, "apical"))) == 10  # translation + RV
    assert np.allclose(time_basis(np.array([0.0]))[0], 0)


def test_edit_scales_only_target_radial_curves():
    numbers = np.random.default_rng(0).normal(size=(SHARED, 4))
    assert np.array_equal(edit(numbers, {3: 1.0}), numbers)
    out = edit(numbers, {3: 0.0, 15: 0.5})
    assert np.all(out[[11, 12]] == 0) and np.allclose(out[[47, 48]], 0.5 * numbers[[47, 48]])
    keep = np.setdiff1d(np.arange(SHARED), [11, 12, 47, 48])
    assert np.array_equal(out[keep], numbers[keep]) and np.array_equal(numbers, np.random.default_rng(0).normal(size=(SHARED, 4)))
