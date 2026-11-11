import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from scripts.register_rig import code_fields, fit_images
from src.rig import decode
from src.utils.svf import integrate, warp
from tests.test_rig import toy_mask


def test_image_fit_recovers_a_known_contraction():
    mask = np.stack([toy_mask()] * 2)
    fields, slots, size = code_fields(mask, "rig")
    truth = np.zeros((size, 4), np.float32)
    truth[2:50:3, 0], truth[3:50:3, 0] = 1.5, 0.5  # endo and epi radial of every LV segment, 1 - cos curve
    rng = np.random.default_rng(0)
    ed = gaussian_filter(np.array([0.1, 0.8, 0.3, 1.0])[mask] + 0.05 * rng.normal(size=mask.shape), (0, 1.5, 1.5)).astype(np.float32)
    tau = np.array([0.25, 0.5, 0.75])

    def moved(x: np.ndarray, numbers: np.ndarray, t: float) -> torch.Tensor:
        return warp(torch.from_numpy(x).float(), integrate(torch.from_numpy(decode(numbers, t, fields, slots)).float()))

    frames = np.stack([moved(ed[:, None], truth, t)[:, 0].numpy() for t in tau])
    fitted = fit_images(ed, frames, tau, fields, slots, size, "cpu", levels=(2, 1), iters=60)

    onehot = F.one_hot(torch.from_numpy(mask), 4).permute(0, 3, 1, 2).numpy()
    lv = [moved(onehot, n, 0.5).argmax(1) == 3 for n in (truth, fitted, np.zeros_like(truth))]
    dice = [2 * (x & lv[0]).sum().item() / (x.sum() + lv[0].sum()).item() for x in lv[1:]]
    assert dice[1] < 0.85 < 0.95 < dice[0]  # no motion vs fitted


def test_numbers_read_as_wall_motion():
    from scripts.analyze_readout import segments
    mask = np.stack([toy_mask()] * 3)
    fields, slots, size = code_fields(mask, "rig")
    numbers = np.zeros((size, 4), np.float32)
    numbers[2:50:3, 0], numbers[3:50:3, 0] = 1.5, 0.5  # at tau = 0.5: endo 3 px = 4.5 mm inward, epi 1 px
    v = decode(numbers, 0.5, fields, slots).astype(np.float32)
    onehot = F.one_hot(torch.from_numpy(mask), 4).permute(0, 3, 1, 2).float()
    d = dict(pid="toy", group="NOR", ed=0, es=1, tau=np.array([0.0, 0.5]), ed_mask=mask, svf=np.stack([np.zeros_like(v), v]),
             es_mask=warp(onehot, integrate(torch.from_numpy(v))).argmax(1).numpy())
    s = segments(d, dict(rig=numbers), "cpu")
    assert len(s) == 16 and np.allclose(s.endo_rig, 4.5) and np.allclose(s.thick_rig, 3.0)
    assert (np.abs(s.endo_teacher - s.endo_label) < 0.75).all() and (np.abs(s.endo_label / s.endo_rig - 1) < 0.25).all()
    assert (s.thick_label > 1).all() and (s.thick_teacher > 1).all()
