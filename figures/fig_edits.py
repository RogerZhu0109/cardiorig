"""
Supplementary figure: segment edits.
(a) Septal and lateral akinesia (endo and epi radial parameters of five segments set to zero) on the held-out
    prediction of one normal heart, the patient of Fig. 1 (median held-out rig ES Dice among NOR). Middle
    mid-ventricular slice at ES: solid contours are the edited rig, dashed white the unedited rig.
(b) Inward endocardial motion of the septal and lateral walls of that slice over the cycle.
(c) STRAUS, infarct report applied to the true healthy motion: error on the flagged segments at ES for the 12
    healthy/infarct pairs (outputs/straus/edits.csv), with the mean.
CVPR full width (6.875 in); Arial, TrueType fonts embedded.
Usage: python figures/fig_edits.py  ->  figures/pdf/edits.pdf
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures" / "pdf"
sys.path.insert(0, str(REPO))
from scripts.analyze_edits import Patient, lv_radius  # noqa: E402
from src.data.cache import fold_patients, load  # noqa: E402
from src.rig import OFFSET, SECTORS, edit, sector_of_bins, time_basis  # noqa: E402
from src.utils.svf import integrate, warp  # noqa: E402

SEPTAL, LATERAL = [0, 5, 6, 11, 12], [2, 3, 8, 9, 14]  # segments either side of the septal direction / opposite
BLUE, RED, GRAY, LIGHT, INK = "#3775BA", "#B64342", "#6E6E6E", "#B5B5B5", "#1A1A1A"
MM, DASH = 1.5, (0, (2.2, 1.4))
plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "font.size": 7,
                     "mathtext.fontset": "stixsans", "axes.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42, "ps.fonttype": 42})

# ------------------------------------------------------------------ data
r = pd.read_csv(REPO / "outputs/edits/reports.csv")
r = r[(r.group == "NOR") & (r.method == "rig")].assign(dice=lambda x: x[["dice_RV", "dice_MYO", "dice_LV"]].mean(axis=1))
PID = r.sort_values("dice").pid.iloc[len(r) // 2]
fold = next(f for f in range(5) if PID in fold_patients(f))
d = next(x for x in load(REPO / "data/processed/acdc", "val", svf=False, fold=fold) if str(x["pid"]) == PID)
pred = np.load(REPO / "outputs/cv/rig_predictions.npz")
p = Patient(d, dict(zip(pred["pids"].tolist(), pred["numbers"]))[PID], "cpu")
mids = [z for z, pos in p.where.items() if pos == "mid"]
Z = mids[len(mids) // 2]
k = p.rays[Z]
tau_es = float(d["tau"][int(d["es"])])
image = d["cine"][int(d["ed"])][Z].astype(np.float32)
SETS = {"no edit": p.numbers, "septal": edit(p.numbers, {s: 0.0 for s in SEPTAL}), "lateral": edit(p.numbers, {s: 0.0 for s in LATERAL})}


@torch.no_grad()
def at(numbers: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Image and labels of slice Z at phase tau under one set of rig parameters."""
    p.phi = time_basis(np.array([tau]), numbers.shape[1] // 2)[0]
    u = integrate(p.decode([numbers])[0][Z:Z + 1])
    return (warp(torch.from_numpy(image[None, None]), u)[0, 0].numpy(), warp(p.onehot[Z:Z + 1], u).argmax(1)[0].numpy())


def panel(ax, img, title, colour=INK):
    ax.imshow(img[box], cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
    ax.set_title(title, fontsize=7, pad=2.5, color=colour)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def contours(ax, lab, **kw):
    for m in (lab == 3, lab >= 2):
        ax.contour(gaussian_filter(m.astype(float), 0.6)[box], [0.5], **kw)


# ------------------------------------------------------------------ canvas (inches)
W, P, PAD, TOP = 6.875, 1.2, 0.03, 0.16
H = TOP + P + 0.04
fig = plt.figure(figsize=(W, H))
axes = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])  # noqa: E731
cx, cy = int(round(k["cx"])), int(round(k["cy"]))
box = (slice(cy - 32, cy + 32), slice(cx - 32, cx + 32))

# (a) the edited walls on the ED frame, then each edit at ES
ax = axes(0.02, 0.04, P, P)
panel(ax, image, "ED, edited segments")
seg, myo = p.segment_of_pixels(Z), d["ed_mask"][Z] == 2
tint = np.zeros((*image.shape, 4))
tint[myo & np.isin(seg, SEPTAL)] = (*to_rgb(BLUE), 0.75)
tint[myo & np.isin(seg, LATERAL)] = (*to_rgb(RED), 0.75)
ax.imshow(tint[box])
_, base = at(SETS["no edit"], tau_es)
for i, (name, colour) in enumerate((("septal", BLUE), ("lateral", RED)), start=1):
    ax = axes(0.02 + i * (P + PAD), 0.04, P, P)
    img, lab = at(SETS[name], tau_es)
    panel(ax, img, f"ES, {name} akinesia", colour)
    contours(ax, lab, colors=[colour], linewidths=1.1)
    contours(ax, base, colors="white", linewidths=0.7, linestyles=[DASH])

# (b) inward endo motion of the two walls over the cycle
taus = np.linspace(0, 1, 31)
sector = sector_of_bins(p.septum, SECTORS[p.where[Z]]) + OFFSET[p.where[Z]]
r0 = lv_radius(d["ed_mask"][Z], k)
radius = {name: np.stack([lv_radius(at(numbers, t)[1], k) for t in taus]) for name, numbers in SETS.items()}
X1, W1, H1 = 0.02 + 3 * (P + PAD) + 0.42, 1.0, 0.44
for j, (wall, segments) in enumerate((("septal", SEPTAL), ("lateral", LATERAL))):
    ax = axes(X1, 0.3 + (1 - j) * (H1 + 0.1), W1, H1)
    bins = np.isin(sector, segments) & k["has_wall"]
    ax.axvline(tau_es, color="#CFCECE", lw=0.6, zorder=0)
    for name, colour, lw in (("no edit", LIGHT, 2.4), ("septal", BLUE, 1.1), ("lateral", RED, 1.1)):
        ax.plot(taus, MM * (r0[bins] - radius[name][:, bins]).mean(1), color=colour, lw=lw, ls=DASH if name == "lateral" else "-")
    ax.text(0.98, 0.96, f"{wall} wall", transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.5, 8.5)
    ax.set_yticks([0, 4, 8])
    ax.set_xticks([0, tau_es, 1], ["ED", "ES", "ED"] if j else [])
    ax.tick_params(pad=2)
fig.text((X1 - 0.27) / W, (0.3 + H1 + 0.05) / H, "inward motion (mm)", rotation=90, ha="center", va="center")

# (c) STRAUS: error on the flagged segments under each edit of the true healthy motion
e = pd.read_csv(REPO / "outputs/straus/edits.csv").query("base == 'truth'")
wide = e.pivot(index=["anatomy", "condition"], columns="edit", values="epe_flagged")
EDITS = [("none", "none", LIGHT), ("report", "report", BLUE), ("misplaced", "wrong wall", GRAY), ("global", "all walls", GRAY)]
ax = axes(X1 + W1 + 0.52, 0.3, W - (X1 + W1 + 0.52) - 0.04, 2 * H1 + 0.1)
jitter = np.random.default_rng(0).uniform(-0.16, 0.16, len(wide))
for i, (key, _, colour) in enumerate(EDITS):
    ax.scatter(i + jitter, wide[key], s=7, color=colour, alpha=0.75, lw=0)
    ax.plot([i - 0.28, i + 0.28], [wide[key].mean()] * 2, color=INK, lw=1.0)
ax.set_xlim(-0.55, 3.55)
ax.set_ylim(0, 7)
ax.set_yticks([0, 3, 6])
ax.set_xticks(range(4), [label for _, label, _ in EDITS], rotation=25, ha="right", rotation_mode="anchor")
ax.tick_params(axis="x", length=0, pad=2)
ax.tick_params(axis="y", pad=2)
ax.set_ylabel("error (mm)", labelpad=2)

for letter, x in (("a", 0.03), ("b", X1 - 0.36), ("c", X1 + W1 + 0.2)):
    fig.text(x / W, (H - 0.02) / H, letter, fontsize=8, fontweight="bold", color=INK, ha="left", va="top")

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "edits.pdf")
print("saved", OUT / "edits.pdf", PID, "slice", Z, f"height {H:.2f} in")
print(wide.mean().round(2).to_dict(), "report lower than none in", int((wide.report < wide.none).sum()), "of", len(wide))
