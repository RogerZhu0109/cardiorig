"""
Paper Fig. 1 (teaser): one ED frame -> predicted rig parameters -> the cardiac cycle.
Patient picked by a fixed rule, not by look: the NOR patient with the median held-out rig ES Dice
(outputs/edits/reports.csv, as in the segment-edit figures). Parameters are the held-out prediction of the
5-fold run (outputs/cv/rig_predictions.npz), not a fit. Mid-ventricular slice.
Generated frames carry the ED endocardium dashed; the ES frame has one arrow per segment, true scale.
CVPR full width (6.875 in); Arial, TrueType fonts embedded.
Usage: python figures/fig_teaser.py  ->  figures/pdf/teaser.pdf
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Wedge  # noqa: E402
from scipy.ndimage import binary_dilation  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures" / "pdf"
sys.path.insert(0, str(REPO))
from src.rig import OFFSET, SECTORS, TAU, decode, patient_fields, positions, sector_centres, septal_angle, time_basis, token_maps  # noqa: E402
from src.utils.svf import integrate, warp  # noqa: E402

SEG_C = ["#B64342", "#E9A6A1", "#E0B43C", "#8BCF8B", "#42949E", "#3775BA"]
RV_C = ["#9A4D8E", "#B97DB0", "#D3A9CC", "#E8D2E4"]
INK, GRAY = "#1A1A1A", "#6E6E6E"
CMAP = LinearSegmentedColormap.from_list("theta", ["#3775BA", "#F7F7F7", "#B64342"])
PHASES = [0.15, None, 0.6]  # None = end-systole
plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "font.size": 7,
                     "mathtext.fontset": "stixsans", "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})

# ------------------------------------------------------------------ data
r = pd.read_csv(REPO / "outputs/edits/reports.csv")
r = r[(r.group == "NOR") & (r.method == "rig")].assign(dice=lambda x: x[["dice_RV", "dice_MYO", "dice_LV"]].mean(axis=1))
PID = r.sort_values("dice").pid.iloc[len(r) // 2]
with np.load(REPO / f"data/processed/acdc/training/{PID}.npz") as f:
    d = {k: f[k] for k in f}
pred = np.load(REPO / "outputs/cv/rig_predictions.npz")
numbers = dict(zip(pred["pids"].tolist(), pred["numbers"]))[PID]

ED, mask = int(d["ed"]), d["ed_mask"]
tau_es = float(d["tau"][int(d["es"])])
phases = [tau_es if p is None else p for p in PHASES]
where = positions(mask)
mids = [z for z, pos in sorted(where.items()) if pos == "mid"]
Z = mids[len(mids) // 2]
fields, slots, _ = patient_fields(mask)
septum = septal_angle(mask)


def norm(img):
    lo, hi = np.percentile(img, [1, 99.5])
    return np.clip((img - lo) / (hi - lo), 0, 1)


image = norm(d["cine"][ED, Z].astype(np.float32))
tm = token_maps(fields, slots)[Z]
first = OFFSET["mid"]
labels = np.zeros(mask[Z].shape, int)
labels[mask[Z] == 2] = 1 + tm[first:first + 6].argmax(0)[mask[Z] == 2]
labels[mask[Z] == 1] = 7 + tm[16:20].argmax(0)[mask[Z] == 1]
labels[mask[Z] == 3] = 11
onehot = F.one_hot(torch.from_numpy(labels), 12).permute(2, 0, 1)[None].float()

UP = 8


def window(of, half):
    ys, xs = np.nonzero(of)
    cy, cx = int(round(ys.mean())), int(round(xs.mean()))
    return slice(cy - half, cy + half), slice(cx - half, cx + half)


def svf(tau):
    return torch.from_numpy(decode(numbers, tau, fields, slots)[Z][None]).float()


def at(tau):
    u = integrate(svf(tau))
    return warp(torch.from_numpy(image[None, None]).float(), u)[0, 0].numpy(), warp(onehot, u)[0]


def smooth_labels(prob, crop):
    lab = F.interpolate(prob[None, :, crop[0], crop[1]], scale_factor=UP, mode="bilinear")
    return F.avg_pool2d(lab, 2 * UP + 1, 1, UP, count_include_pad=False).argmax(1)[0].numpy()


ED_LAB = {}


def show(ax, img, prob, crop, tint=0.55, lw=0.6, ed_contour=False):
    n = crop[0].stop - crop[0].start
    box = (-0.5, n - 0.5, n - 0.5, -0.5)
    ax.imshow(img[crop], cmap="gray", vmin=0, vmax=1, interpolation="lanczos", extent=box)
    lab = smooth_labels(prob, crop)
    rgba = np.zeros((*lab.shape, 4))
    for i, c in enumerate(SEG_C + RV_C, start=1):
        rgba[lab == i] = (*to_rgb(c), tint if i <= 6 else 0.6 * tint)
    ax.imshow(rgba, interpolation="bilinear", extent=box)
    g = (np.arange(n * UP) + 0.5) / UP - 0.5
    ax.contour(g, g, lab == 11, levels=[0.5], colors="white", linewidths=lw)
    ax.contour(g, g, (lab >= 1) & (lab <= 6) | (lab == 11), levels=[0.5], colors="white", linewidths=lw)
    if ed_contour:  # ED endocardium, so the contraction is visible against it
        ed = smooth_labels(onehot[0], crop)
        ax.contour(g, g, ed == 11, levels=[0.5], colors=INK, linewidths=0.7, linestyles=[(0, (2.2, 1.4))])
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#3A3A3A")


def segment_arrows(tau, crop):
    """Per mid segment: ED endo border point (angular middle of the segment) and its motion at tau."""
    v = svf(tau)
    fwd = integrate(-v)[0].numpy()  # tissue at p moves to p + fwd(p) (inverse of the sampling map)
    border = binary_dilation(labels == 11) & (labels >= 1) & (labels <= 6)
    ys, xs = np.nonzero(labels == 11)
    cy, cx = ys.mean(), xs.mean()
    out = []
    for s in range(6):
        py, px = np.nonzero(border & (labels == 1 + s))
        ang = np.arctan2(py - cy, px - cx)
        mid = np.angle(np.exp(1j * ang).mean())
        k = np.argmin(np.abs(np.angle(np.exp(1j * (ang - mid)))))
        dx, dy = fwd[0, py, px].mean(), fwd[1, py, px].mean()  # segment-mean motion of the endo border
        out.append((px[k] - crop[1].start, py[k] - crop[0].start, dx, dy, SEG_C[s]))
    return out


# ------------------------------------------------------------------ canvas (inches)
FW, FH = 6.875, 2.02
Y0 = 0.36
fig = plt.figure(figsize=(FW, FH))
ov = fig.add_axes([0, 0, 1, 1], zorder=20)
ov.set_xlim(0, FW)
ov.set_ylim(0, FH)
ov.axis("off")
ov.patch.set_alpha(0)


def axes(x, y, w, h, **kw):
    return fig.add_axes([x / FW, y / FH, w / FW, h / FH], **kw)


def arrow(x0, x1, y, label):
    ov.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>,head_length=3.2,head_width=1.8", mutation_scale=1,
                                 lw=0.9, color=INK, shrinkA=0, shrinkB=0))
    ov.text((x0 + x1) / 2, y + 0.07, label, ha="center", va="bottom", fontsize=7, color=INK)


# (1) the ED frame with the rig's segments
A = 1.6
ax = axes(0.04, Y0, A, A)
show(ax, image, onehot[0], window(mask[Z] > 0, 36))
ov.text(0.04 + A / 2, 0.2, "one ED frame", ha="center", va="center", fontsize=7.5)
arrow(1.71, 2.07, Y0 + A / 2, "predict")

# (2) predicted parameters at ES as bullseyes; the mid ring (the slice shown) outlined in the segment colours
c_es = numbers @ time_basis(np.array([tau_es]))[0]
vmax = np.abs(c_es[2:50]).max()
BX, BD = 2.17, 0.38
for j, name in enumerate(["endo", "epi", "tangential"]):
    y0 = Y0 + A - BD - j * (BD + 0.15)
    ax = axes(BX, y0, BD, BD)
    rings = {"basal": (0.68, 1.0), "mid": (0.36, 0.68), "apical": (0.04, 0.36)}
    for pos, (r0, r1) in rings.items():
        n = SECTORS[pos]
        for s in range(n):
            mid = np.degrees(septum + sector_centres(n) + s * TAU / n)
            value = c_es[2 + 3 * (OFFSET[pos] + s) + j]
            ax.add_patch(Wedge((0, 0), r1, mid - 180 / n, mid + 180 / n, width=r1 - r0,
                               fc=CMAP(0.5 + 0.5 * value / vmax), ec="white", lw=0.5))
    r0, r1 = rings["mid"]
    for s in range(6):
        mid = np.degrees(septum + sector_centres(6) + s * TAU / 6)
        ax.add_patch(Wedge((0, 0), r1 - 0.035, mid - 30 + 4, mid + 30 - 4, width=r1 - r0 - 0.07,
                           fc="none", ec=SEG_C[s], lw=1.1, zorder=3))
    ax.set_xlim(-1.03, 1.03)
    ax.set_ylim(1.03, -1.03)
    ax.set_aspect("equal")
    ax.axis("off")
    ov.text(BX + BD / 2, y0 - 0.07, name, ha="center", va="center", fontsize=6.5, color=GRAY)
ov.text(BX + BD / 2, 0.2, "232 parameters", ha="center", va="center", fontsize=7.5)
arrow(2.66, 3.02, Y0 + A / 2, "decode")

# (3) the cycle: three generated frames above the endo curves of the six mid segments
CX, CW = 3.22, FW - 3.22 - 0.12
CH = 0.55  # curve height
n, gap = len(phases), 0.1
fw = min((CW - (n - 1) * gap) / n, A - CH - 0.08)
fx0 = CX + (CW - (n * fw + (n - 1) * gap)) / 2
fy = Y0 + A - fw
axc = axes(CX, Y0, CW, CH)
tight = window(mask[Z] >= 2, 30)
t = np.linspace(0, 1, 200)
curves = numbers[2 + 3 * first:2 + 3 * (first + 6):3] @ time_basis(t).T
axc.axhline(0, color=GRAY, lw=0.5)
for c, col in zip(curves, SEG_C):
    axc.plot(t, c, color=col, lw=1.2)
axc.set_xlim(0, 1)
lo, hi = curves.min(), curves.max()
axc.set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
axc.set_xticks([0, tau_es, 1])
axc.set_xticklabels(["ED", "ES", "ED"])
axc.set_yticks([])
axc.tick_params(length=2, width=0.6, pad=2)
axc.set_ylabel("endo motion", fontsize=6.5, color=GRAY, labelpad=3)
axc.set_xlabel(r"cardiac phase $\tau$", fontsize=7, labelpad=-6)
for sp in ("top", "right"):
    axc.spines[sp].set_visible(False)

names = {0: "systole", 1: "end-systole", 2: "diastole"}
for i, tau in enumerate(phases):
    ax = axes(fx0 + i * (fw + gap), fy, fw, fw)
    show(ax, *at(tau), tight, lw=0.5, ed_contour=True)
    if tau == tau_es:
        for x, y, dx, dy, col in segment_arrows(tau, tight):
            ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                        arrowprops=dict(arrowstyle="-|>,head_length=0.15,head_width=0.08", color="white", lw=0.9,
                                        shrinkA=0, shrinkB=0, mutation_scale=10,
                                        path_effects=[pe.Stroke(linewidth=1.8, foreground=INK), pe.Normal()]),
                        zorder=5)
        for sp in ax.spines.values():
            sp.set_color(INK)
            sp.set_linewidth(1.0)
    ax.text(0.04, 0.96, names[i], transform=ax.transAxes, ha="left", va="top", fontsize=6, color="white")
    axc.axvline(tau, color=GRAY, lw=0.5, ls=(0, (2, 2)), zorder=0)
    fig.add_artist(ConnectionPatch((tau, 1), (0.5, 0), coordsA=axc.get_xaxis_transform(), coordsB=ax.transAxes,
                                   color=GRAY, lw=0.5))

OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "teaser.pdf")
print("saved", OUT / "teaser.pdf", PID, "slice", Z, "tau_es", round(tau_es, 3))
