"""
Paper Fig. 4: ES predicted from the ED frame alone, on ACDC test and zero-shot on M&Ms.
Subjects picked by a fixed rule, not by look: the median rig ES Dice (mean over seeds) among ACDC test NOR and
HCM patients, and among M&Ms Philips, GE and Canon subjects (outputs/test/test.csv, outputs/mnms/mnms.csv).
Predictions are the seed-0 final models (scripts/export_qualitative.py). Slice: the middle mid-ventricular one,
as in Fig. 1. Solid contours are the warped ED labels, dashed white the real ES labels; the number is the
subject's ES Dice over all slices (as in Tab. 2).
CVPR full width (6.875 in); Arial, TrueType fonts embedded.
Usage: python figures/fig_qualitative.py  ->  figures/pdf/qualitative.pdf
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures" / "pdf"
sys.path.insert(0, str(REPO))
from src.rig import positions  # noqa: E402
from src.utils.metrics import STRUCTURES, dice  # noqa: E402

ROWS = [("patient144", "ACDC", "NOR"), ("patient105", "ACDC", "HCM"),
        ("J9L6N9", "M&Ms", "Philips"), ("E7L0N6", "M&Ms", "GE"), ("L8M2U8", "M&Ms", "Canon")]
METHODS = [("krebs_mean", "Krebs et al."), ("phaseflow_tau0", "PhaseFlow"), ("current", "Dense-field"),
           ("rig", "CardioRig"), ("teacher", "Dense registration")]
COLS = [("ed", "ED (input)"), ("es", "ES (real)")] + METHODS
SHAPES = {"LV": lambda m: m == 3, "Epicardium": lambda m: (m == 2) | (m == 3), "RV": lambda m: m == 1}
COLOR = {"LV": "#FFD700", "Epicardium": "#8BCF8B", "RV": "#EA84DD"}
BLUE, INK = "#0F4D92", "#1A1A1A"
MM = 1.5
plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "font.size": 7,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})
OUTLINE = [pe.withStroke(linewidth=1.6, foreground="black")]


def load(pid):
    with np.load(REPO / f"outputs/qualitative/{pid}.npz") as f:
        d = {k: f[k] for k in f}
    mids = [z for z, pos in sorted(positions(d["ed_mask"]).items()) if pos == "mid"]
    d["Z"] = mids[len(mids) // 2]
    d["dice"] = {m: np.mean(list(dice(d[f"{m}_lab"], d["es_mask"]).values())) for m, _ in METHODS}
    return d


def contours(ax, mask, dashed):
    for name, of in SHAPES.items():
        m = gaussian_filter(of(mask).astype(float), 0.6)
        if m.max() < 0.5:
            continue
        if dashed:
            ax.contour(m, [0.5], colors="white", linewidths=0.6, linestyles=[(0, (2.2, 1.4))])
        else:
            ax.contour(m, [0.5], colors=COLOR[name], linewidths=0.9)


data = [load(pid) for pid, *_ in ROWS]
# one window size for all rows (same scale), centred on each slice's ED heart
boxes = []
for d in data:
    heart = np.zeros_like(d["ed_mask"][d["Z"]], bool)
    for k in ["ed_mask", "es_mask"] + [f"{m}_lab" for m, _ in METHODS]:
        heart |= d[k][d["Z"]] > 0
    ys, xs = np.nonzero(d["ed_mask"][d["Z"]] > 0)
    cy, cx = int(round(ys.mean())), int(round(xs.mean()))
    hy, hx = np.nonzero(heart)
    boxes.append((cy, cx, max(np.abs(hy - cy).max(), np.abs(hx - cx).max())))
HALF = max(b[2] for b in boxes) + 6

# ------------------------------------------------------------------ layout (inches)
W, LEFT, PAD, GAP, TOP, BOTTOM = 6.875, 0.30, 0.025, 0.09, 0.30, 0.22
P = (W - LEFT - 0.01 - 6 * PAD - 2 * GAP) / 7
x0 = [LEFT + j * (P + PAD) + (GAP if j >= 2 else 0) + (GAP if j >= 6 else 0) for j in range(7)]


def draw(keep, name):
    """One figure with the rows keep (indices into ROWS); all figures share the window size."""
    rows = [(ROWS[i], data[i], boxes[i]) for i in keep]
    H = TOP + len(rows) * P + (len(rows) - 1) * PAD + BOTTOM
    fig = plt.figure(figsize=(W, H))
    for i, ((pid, dataset, group), d, (cy, cx, _)) in enumerate(rows):
        Z = d["Z"]
        win = (slice(cy - HALF, cy + HALF), slice(cx - HALF, cx + HALF))
        lo, hi = np.percentile(d["ed"][Z], [1, 99.5])
        y0 = H - TOP - (i + 1) * P - i * PAD
        for j, (key, title) in enumerate(COLS):
            ax = fig.add_axes([x0[j] / W, y0 / H, P / W, P / H])
            img = d[key][Z] if key in ("ed", "es") else d[f"{key}_img"][Z]
            ax.imshow(np.clip((img - lo) / (hi - lo), 0, 1)[win], cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
            if key == "ed":
                contours(ax, d["ed_mask"][Z][win], dashed=False)
            else:
                if key != "es":
                    contours(ax, d[f"{key}_lab"][Z][win], dashed=False)
                contours(ax, d["es_mask"][Z][win], dashed=True)
            if key not in ("ed", "es"):
                ax.text(0.04, 0.04, f"{d['dice'][key]:.3f}", transform=ax.transAxes, color="white", fontsize=6.5,
                        ha="left", va="bottom", path_effects=OUTLINE, fontweight="bold" if key == "rig" else "normal")
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if key == "rig":
                for s in ax.spines.values():
                    s.set_visible(True), s.set_color(BLUE), s.set_linewidth(1.4)
            if i == 0:
                ax.set_title(title, fontsize=7, pad=2.5, color=BLUE if key == "rig" else INK,
                             fontweight="bold" if key == "rig" else "normal")
            if i == 0 and j == 0:  # 10 mm scale bar
                n = 2 * HALF
                ax.plot([4, 4 + 10 / MM], [n - 4, n - 4], color="white", lw=1.2, solid_capstyle="butt")
                ax.text(4 + 5 / MM, n - 6, "10 mm", color="white", fontsize=5.5, ha="center", va="bottom", path_effects=OUTLINE)
        fig.text((LEFT - 0.12) / W, (y0 + P / 2) / H, f"{dataset} {group}", rotation=90, ha="center", va="center", fontsize=7)

    # group labels above the headers
    for (a, b), label in [((0, 1), "Input and truth"), ((2, 5), "Predicted from the ED frame"), ((6, 6), "Reference")]:
        xa, xb, y = x0[a] / W, (x0[b] + P) / W, (H - 0.035) / H
        fig.text((xa + xb) / 2, y, label, ha="center", va="top", fontsize=7, color="#4D4D4D")
        if b > a:
            fig.add_artist(Line2D([xa + 0.004, xb - 0.004], [y - 0.075 / H] * 2, color="#9A9A9A", lw=0.6))

    handles = [Line2D([], [], color=COLOR[n], lw=1.2, label=dict(LV="LV endocardium").get(n, n)) for n in SHAPES]
    handles.append(Line2D([], [], color="#4D4D4D", lw=0.9, ls=(0, (2.2, 1.4)), label="Real ES labels"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, fontsize=6.5,
               handlelength=2.2, columnspacing=1.6, borderaxespad=0.15)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(REPO / f"outputs/qualitative/{name}.png", dpi=300)
    print("saved", OUT / f"{name}.pdf", f"panel {P:.3f} in, height {H:.2f} in, window {2 * HALF} px ({2 * HALF * MM:.0f} mm)")


OUT.mkdir(parents=True, exist_ok=True)
# main paper: the normal ACDC patient and the two largest M&Ms vendors unseen in training; supplement: all five rows
draw([0, 2, 3], "qualitative")
draw(range(len(ROWS)), "qualitative_all")
