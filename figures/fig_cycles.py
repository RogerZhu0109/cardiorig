"""
Supplementary figure: the cardiac cycle generated from the ED frame by each predictor, and the LV area curves.
Subjects are the two ACDC test patients of Fig. 4 (median rig ES Dice among NOR and HCM); predictions are the
seed-0 final models (scripts/export_cycles.py, 30 resampled frames). Images: middle mid-ventricular slice of the
NOR patient; solid contours are the warped ED labels, dashed white the ED endocardium. Curves: LV area over the
stack relative to ED; the reference is the segmentation network on the real frames, as in the volume metrics.
CVPR full width (6.875 in); Arial, TrueType fonts embedded.
Usage: python figures/fig_cycles.py  ->  figures/pdf/cycles.pdf
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures" / "pdf"
SUBJECTS = [("patient144", "normal"), ("patient105", "HCM")]
INK, GRAY, BLUE = "#1A1A1A", "#8A8A8A", "#0F4D92"
METHODS = [("krebs_mean", "Krebs et al.", "#B64342"), ("phaseflow_tau0", "PhaseFlow", "#42949E"),
           ("current", "Dense-field", "#9A4D8E"), ("rig", "CardioRig", BLUE)]
SHAPES = {"LV": lambda m: m == 3, "Epicardium": lambda m: (m == 2) | (m == 3), "RV": lambda m: m == 1}
COLOR = {"LV": "#FFD700", "Epicardium": "#8BCF8B", "RV": "#EA84DD"}
DASH = (0, (2.2, 1.4))
plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "font.size": 7,
                     "mathtext.fontset": "stixsans", "axes.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42, "ps.fonttype": 42})

data = []
for pid, _ in SUBJECTS:
    with np.load(REPO / f"outputs/cycles/{pid}.npz") as f:
        data.append({k: f[k] for k in f})

# ------------------------------------------------------------------ layout (inches)
d = data[0]
T, es = len(d["tau"]), int(d["es"])
frames = [0, es // 2, es] + [int(round(es + k * (T - es) / 4)) for k in (1, 2, 3)]
W, LEFT, P, PAD, TOP = 6.875, 0.2, 0.6, 0.02, 0.17
rows = [("real", "Real", INK)] + [(f"{m}_img", name, c) for m, name, c in METHODS]
H = TOP + len(rows) * P + (len(rows) - 1) * PAD + 0.02
fig = plt.figure(figsize=(W, H))

ys, xs = np.nonzero(d["ed_mask"] > 0)
cy, cx = int(round(ys.mean())), int(round(xs.mean()))
half = max(np.abs(ys - cy).max(), np.abs(xs - cx).max()) + 7
win = (slice(cy - half, cy + half), slice(cx - half, cx + half))
lo, hi = np.percentile(d["real"][0], [1, 99.5])
ed_endo = gaussian_filter((d["ed_mask"] == 3).astype(float), 0.6)[win]

for i, (key, name, colour) in enumerate(rows):
    y0 = H - TOP - (i + 1) * P - i * PAD
    for j, t in enumerate(frames):
        ax = fig.add_axes([(LEFT + j * (P + PAD)) / W, y0 / H, P / W, P / H])
        ax.imshow(np.clip((d[key][t] - lo) / (hi - lo), 0, 1)[win], cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
        if key != "real":
            for shape, of in SHAPES.items():
                m = gaussian_filter(of(d[key.replace("_img", "_lab")][t]).astype(float), 0.6)[win]
                if m.max() >= 0.5:
                    ax.contour(m, [0.5], colors=COLOR[shape], linewidths=0.8)
        if t > 0:
            ax.contour(ed_endo, [0.5], colors="white", linewidths=0.6, linestyles=[DASH])
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if i == 0:
            ax.set_title("ED" if t == 0 else "ES" if t == es else rf"$\tau$ = {d['tau'][t]:.2f}", fontsize=7, pad=2.5)
    fig.text((LEFT - 0.1) / W, (y0 + P / 2) / H, name, rotation=90, ha="center", va="center", fontsize=7, color=colour,
             fontweight="bold" if key == "rig_img" else "normal")

# ------------------------------------------------------------------ LV area curves
CX, CW = LEFT + 6 * (P + PAD) + 0.52, 2.25
CH = (H - TOP - 0.32 - 0.22) / 2
for k, (d, (pid, group)) in enumerate(zip(data, SUBJECTS)):
    ax = fig.add_axes([CX / W, (0.32 + (1 - k) * (CH + 0.22)) / H, CW / W, CH / H])
    tau = d["tau"]
    ax.axvline(tau[int(d["es"])], color="#CFCECE", lw=0.6, zorder=0)
    ax.plot(tau, d["truth_area"] / d["truth_area"][0], color=INK, lw=1.4, label="real frames")
    ax.plot(tau, d["teacher_area"] / d["teacher_area"][0], color=GRAY, lw=1.0, ls=DASH, label="dense registration")
    for m, name, colour in METHODS:
        ax.plot(tau, d[f"{m}_area"] / d[f"{m}_area"][0], color=colour, lw=1.5 if m == "rig" else 1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_xticks([0, tau[int(d["es"])], 1], ["ED", "ES", "ED"] if k else [])
    ax.tick_params(pad=2)
    ax.set_ylabel("LV area / ED area", labelpad=2)
    ax.text(0.98, 0.05, group, transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    if k:
        ax.set_xlabel(r"cardiac phase $\tau$", labelpad=-6, x=0.65)
    else:
        ax.legend(handles=[Line2D([], [], color=INK, lw=1.4, label="real frames"),
                           Line2D([], [], color=GRAY, lw=1.0, ls=DASH, label="dense registration")],
                  loc="lower left", fontsize=6.5, frameon=False, handlelength=2.0, borderpad=0, labelspacing=0.3)

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "cycles.pdf")
print("saved", OUT / "cycles.pdf", f"height {H:.2f} in, frames {frames}, window {2 * half} px")
