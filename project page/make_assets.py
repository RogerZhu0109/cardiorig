"""
Assets of the project page.
  assets/data.js      one mid-ventricular slice of one patient for the interactive figures: ED image, real frames,
                      the rig's basis fields and their factors, and the 58 x 4 rig parameters (predicted and fitted).
                      The patient is the one of the paper's teaser (NOR patient with the median held-out rig ES Dice);
                      the predicted parameters are the held-out prediction of the 5-fold run.
  assets/cycle_*.gif  real cine next to the cycle generated from the ED frame, for the two ACDC test patients of
                      the cycle figure (outputs/cycles, final seed-0 model).
  assets/method.png   the method figure of the paper.
Usage (from the repository root): .venv/bin/python "project page/make_assets.py"
"""
import base64
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion, zoom

REPO = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "assets"
sys.path.insert(0, str(REPO))
from src.rig import BIN_CENTRES, DECAY, TAU, _periodic_hats, fit_compact, patient_fields, positions, rays, sector_centres, septal_angle  # noqa: E402

N, Y0, X0 = 96, 18, 18  # crop of the 128 x 128 slice shown on the page


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def crop(a: np.ndarray) -> np.ndarray:
    return a[..., Y0:Y0 + N, X0:X0 + N]


def page_data():
    r = pd.read_csv(REPO / "outputs/edits/reports.csv")
    r = r[(r.group == "NOR") & (r.method == "rig")].assign(dice=lambda x: x[["dice_RV", "dice_MYO", "dice_LV"]].mean(axis=1))
    pid = r.sort_values("dice").pid.iloc[len(r) // 2]
    with np.load(REPO / f"data/processed/acdc/training/{pid}.npz") as f:
        d = {k: f[k] for k in f}
    pred = np.load(REPO / "outputs/cv/rig_predictions.npz")
    predicted = dict(zip(pred["pids"].tolist(), pred["numbers"]))[pid]
    fitted = fit_compact(d)

    mask = d["ed_mask"]
    mids = [z for z, pos in sorted(positions(mask).items()) if pos == "mid"]
    z = mids[len(mids) // 2]
    fields, slots, _ = patient_fields(mask)
    septum = septal_angle(mask)
    basis = fields[z]
    assert np.abs(basis).max() <= 1.0 and len(basis) == 28

    # factors of the LV fields, as in src/rig.py basis(); checked against the fields themselves
    k = rays(mask[z])
    r_, th = k["r"], k["th"]
    endo = np.interp(th, BIN_CENTRES, k["endo"], period=TAU)
    epi = np.interp(th, BIN_CENTRES, k["epi"], period=TAU)
    depth = np.clip((r_ - endo) / (epi - endo), 0, 1)
    cavity = np.clip(r_ / endo, 0, 1)
    decay = np.exp(-np.maximum(r_ - epi, 0) / DECAY)
    hats = _periodic_hats(th, 6, septum + sector_centres(6))
    e_r, e_t = np.stack([np.cos(th), np.sin(th)]), np.stack([-np.sin(th), np.cos(th)])
    for s in range(6):
        assert np.allclose(basis[2 + 3 * s], hats[s] * cavity * (1 - depth) * e_r)
        assert np.allclose(basis[3 + 3 * s], hats[s] * depth * decay * e_r)
        assert np.allclose(basis[4 + 3 * s], hats[s] * cavity * decay * e_t)

    lo, hi = np.percentile(d["cine"][int(d["ed"]), z].astype(np.float32), [1, 99.5])
    order = np.argsort(d["tau"], kind="stable")
    frames = np.clip((d["cine"][order, z].astype(np.float32) - lo) / (hi - lo), 0, 1)
    u8 = lambda a: np.round(255 * a).astype(np.uint8)  # noqa: E731

    data = dict(
        pid=pid, n=N, spacing=1.5, decay=DECAY,
        tau=[round(float(t), 4) for t in d["tau"][order]], tau_es=round(float(d["tau"][int(d["es"])]), 4),
        frames=b64(u8(crop(frames))),  # (frames, N, N), the first is ED
        mask=b64(crop(mask[z])),  # 1 RV, 2 LV wall, 3 LV cavity
        basis=b64(np.round(127 * crop(basis)).astype(np.int8)),  # (28, 2, N, N), x then y
        slots=slots[z].tolist(),  # row of the 58 x 4 parameters that drives each field
        hats=b64(u8(crop(hats))), cavity=b64(u8(crop(cavity))), depth=b64(u8(crop(depth))), decay_map=b64(u8(crop(decay))),
        cx=round(k["cx"] - X0, 3), cy=round(k["cy"] - Y0, 3), septum=round(septum, 4),
        first_sector=round(septum + sector_centres(6), 4),
        endo=np.round(k["endo"], 3).tolist(), epi=np.round(k["epi"], 3).tolist(),
        predicted=np.round(predicted, 4).tolist(), fitted=np.round(fitted, 4).tolist(),
    )
    (OUT / "data.js").write_text("window.RIG = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"data.js: {pid}, slice {z}, {(OUT / 'data.js').stat().st_size / 1e6:.2f} MB")


def outline(region: np.ndarray, width: int = 2) -> np.ndarray:
    return region & ~binary_erosion(region, iterations=width)


def cycle_gif(pid: str, name: str, scale: int = 4, ms: int = 60):
    with np.load(REPO / f"outputs/cycles/{pid}.npz") as f:
        d = {k: f[k] for k in f}
    ys, xs = np.nonzero(d["ed_mask"] > 0)
    cy, cx = int(round(ys.mean())), int(round(xs.mean()))
    half = int(max(np.abs(ys - cy).max(), np.abs(xs - cx).max()) + 9)
    win = (slice(cy - half, cy + half), slice(cx - half, cx + half))
    size, gap, top = 2 * half * scale, 12, 30
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 17)
    colours = {"RV": (234, 132, 221), "Epicardium": (139, 207, 139), "LV": (255, 215, 0)}
    shapes = {"RV": lambda m: m == 1, "Epicardium": lambda m: (m == 2) | (m == 3), "LV": lambda m: m == 3}

    def pane(img):
        up = np.clip(zoom(img[win], scale, order=3), 0, 1)
        return np.repeat(np.round(255 * up).astype(np.uint8)[..., None], 3, -1)

    frames = []
    for t in range(len(d["tau"])):
        real, gen = pane(d["real"][t]), pane(d["rig_img"][t])
        for key, shape in shapes.items():
            region = zoom(shape(d["rig_lab"][t])[win].astype(np.float32), scale, order=1) >= 0.5
            gen[outline(region)] = colours[key]
        canvas = np.full((top + size + 8, 2 * size + gap, 3), 255, np.uint8)
        canvas[top:top + size, :size] = real
        canvas[top:top + size, size + gap:] = gen
        canvas[top + size + 2:top + size + 6, :int((2 * size + gap) * (t + 1) / len(d["tau"]))] = (15, 77, 146)
        im = Image.fromarray(canvas)
        draw = ImageDraw.Draw(im)
        draw.text((2, 5), "Real cine", fill=(26, 26, 26), font=font)
        draw.text((size + gap + 2, 5), "Generated from the first frame", fill=(15, 77, 146), font=font)
        frames.append(im)
    frames[0].save(OUT / f"cycle_{name}.gif", save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=True)
    print(f"cycle_{name}.gif: {pid}, {frames[0].size}, {(OUT / f'cycle_{name}.gif').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    page_data()
    cycle_gif("patient144", "normal")
    cycle_gif("patient105", "hcm")
    subprocess.run(["pdftoppm", "-r", "220", "-png", "-singlefile", str(REPO / "figures/pdf/method.pdf"), str(OUT / "method")], check=True)
