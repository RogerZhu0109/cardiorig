"""
Paper Fig. 6: what the rig parameters say, 100 ACDC training patients, rig fitted to the dense registration
(outputs/readout/segments.csv, codes.npz; scripts/analyze_readout.py).
(a) Mean endo-radial parameter at ES per disease group on the 16 LV segments (basal ring outside, septum left).
(b) Held-out accuracy of a linear classifier of the disease group on the rig parameters and on the B-spline code,
    with the 95% CIs of analyze_readout.py (same folds and random stream).
(c) Endo motion of each segment read from the parameters against the same measurement from the ED and ES labels.
CVPR column width (3.25 in); Arial, TrueType fonts embedded.
Usage: python figures/fig_readable.py  ->  figures/pdf/readable.pdf
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.patches import Wedge  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures" / "pdf"
sys.path.insert(0, str(REPO))
from scripts.analyze_readout import classify  # noqa: E402
from src.rig import OFFSET, SECTORS, TAU, sector_centres  # noqa: E402

BLUE, RED, GRAY, INK = "#0F4D92", "#B64342", "#6E6E6E", "#1A1A1A"
CMAP = LinearSegmentedColormap.from_list("contraction", ["#F7F7F7", "#B64342"])  # the red half of the teaser's map
GROUPS = {"NOR": "normal", "DCM": "DCM", "HCM": "HCM", "MINF": "infarct", "RV": "abnormal RV"}
RINGS = {"basal": (0.68, 1.0), "mid": (0.36, 0.68), "apical": (0.04, 0.36)}
SEPTUM = np.pi  # drawn on the left
plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "font.size": 7,
                     "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

seg = pd.read_csv(REPO / "outputs/readout/segments.csv").dropna(subset=["endo_label"])
with np.load(REPO / "outputs/readout/codes.npz") as f:
    codes = {k: f[k] for k in f.files}

FW, FH = 3.25, 1.98
fig = plt.figure(figsize=(FW, FH))
axes = lambda x, y, w, h: fig.add_axes([x / FW, y / FH, w / FW, h / FH])  # noqa: E731

# (a) group means on bullseyes
mean = seg.pivot_table(index="group", columns="seg", values="endo_rig")
norm = Normalize(0, 12)
D, GAP, X0, Y0 = 0.48, 0.065, 0.2, 1.42
for i, (group, name) in enumerate(GROUPS.items()):
    ax = axes(X0 + i * (D + GAP), Y0, D, D)
    for pos, (r0, r1) in RINGS.items():
        n = SECTORS[pos]
        for s in range(n):
            mid = np.degrees(SEPTUM + sector_centres(n) + s * TAU / n)
            ax.add_patch(Wedge((0, 0), r1, mid - 180 / n, mid + 180 / n, width=r1 - r0,
                               fc=CMAP(norm(mean.loc[group, OFFSET[pos] + s])), ec="white", lw=0.5))
    ax.set_xlim(-1.03, 1.03)
    ax.set_ylim(1.03, -1.03)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.text((X0 + i * (D + GAP) + D / 2) / FW, (Y0 - 0.03) / FH, name, ha="center", va="top", fontsize=6.5)
cax = axes(X0 + 5 * (D + GAP) + 0.02, Y0 + 0.04, 0.05, D - 0.08)
bar = fig.colorbar(plt.cm.ScalarMappable(norm, CMAP), cax=cax, ticks=[0, 6, 12])
bar.outline.set_visible(False)
bar.ax.tick_params(length=1.5, width=0.5, pad=1.5, labelsize=6)
bar.ax.set_title("mm", fontsize=6, pad=2)

# (b) linear read-out of the disease group
pid, group = codes["pid"], codes["group"]
idx = np.random.default_rng(0).integers(len(pid), size=(10000, len(pid)))
ax = axes(0.42, 0.3, 1.0, 0.82)
for i, (key, colour) in enumerate((("rig", BLUE), ("spline", RED))):
    hit = (classify(codes[key], pid, group)[0] == group).astype(float)
    lo, hi = np.percentile(hit[idx].mean(1), [2.5, 97.5])
    ax.bar(1.35 * i, hit.mean(), width=0.7, color=colour, ec=INK, lw=0.6)
    ax.plot([1.35 * i] * 2, [lo, hi], color=INK, lw=0.8)
    print(f"{key}: accuracy {hit.mean():.2f} [{lo:.2f}, {hi:.2f}]")
ax.axhline(0.2, color=GRAY, lw=0.7, ls=(0, (3, 2)))
ax.text(2.55, 0.215, "chance", color=GRAY, fontsize=6, ha="right", va="bottom")
ax.set_xlim(-0.6, 2.6)
ax.set_ylim(0, 1)
ax.set_xticks([0, 1.35], ["CardioRig", "B-spline"])
ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
ax.tick_params(axis="x", length=0, pad=3)
ax.tick_params(axis="y", pad=2)
ax.set_ylabel("accuracy", labelpad=2)

# (c) segment endo motion: parameters against labels
ax = axes(2.0, 0.3, 1.17, 0.82)
lim = (-6, 18.5)
outside = int(((seg.endo_rig < lim[0]) | (seg.endo_rig > lim[1]) | (seg.endo_label < lim[0]) | (seg.endo_label > lim[1])).sum())
ax.plot(lim, lim, color=GRAY, lw=0.7, ls=(0, (3, 2)), zorder=0)
ax.scatter(seg.endo_rig, seg.endo_label, s=2.2, color=BLUE, alpha=0.3, lw=0)
r = np.corrcoef(seg.endo_rig, seg.endo_label)[0, 1]
ax.text(0.04, 0.98, f"r = {r:.2f}", transform=ax.transAxes, ha="left", va="top")
ax.set_xlim(lim)
ax.set_ylim(-4, 18.5)
ax.set_xticks([0, 8, 16])
ax.set_yticks([0, 8, 16])
ax.tick_params(pad=2)
ax.set_xlabel("from parameters (mm)", labelpad=2)
ax.set_ylabel("from labels (mm)", labelpad=2)

for letter, x, y in (("a", 0.02, FH - 0.03), ("b", 0.02, 1.2), ("c", 1.66, 1.2)):
    fig.text(x / FW, y / FH, letter, fontsize=8, fontweight="bold", color=INK, ha="left", va="top")

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "readable.pdf")
print("saved", OUT / "readable.pdf", f"r = {r:.3f}, {len(seg)} patient-segments, {outside} outside the axes")
print(mean.loc[list(GROUPS)].round(1).to_string())
