"""
Paper Fig. 3: how much the rig loses by being small.
(a) ES Dice against the number of parameters per patient, 100 ACDC training patients. Open markers: rig variants
    fitted to the dense registration (outputs/rig/size.csv). Filled markers: rig and B-spline code fitted to the
    images (outputs/rig_registration/acdc.csv). Line: the dense registration.
(b) End-point error against the true motion over the cycle, three healthy STRAUS hearts (outputs/straus/frames.csv):
    mean line, band from the lowest to the highest heart.
CVPR column width (3.25 in); Arial, TrueType fonts embedded.
Usage: python figures/fig_compact.py  ->  figures/pdf/compact.pdf
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures" / "pdf"
BLUE, RED, GRAY, LIGHT, INK = "#0F4D92", "#B64342", "#6E6E6E", "#A8A8A8", "#1A1A1A"
VARIANTS = ["compact", "per slice, smooth curves", "shared slices, per frame", "per slice, per frame"]
plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "font.size": 7,
                     "mathtext.fontset": "stixsans", "axes.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42, "ps.fonttype": 42})

dice = lambda df: df.assign(dice=df[["dice_RV", "dice_MYO", "dice_LV"]].mean(axis=1))  # noqa: E731
size = dice(pd.read_csv(REPO / "outputs/rig/size.csv")).groupby("variant")[["size", "dice"]].mean()
fit = dice(pd.read_csv(REPO / "outputs/rig_registration/acdc.csv")).groupby("method").dice.mean()
frames = pd.read_csv(REPO / "outputs/straus/frames.csv").query("condition == 'healthy'")
n_frames = frames.frame.max() + 1  # frame 0 is ED

fig, (a, b) = plt.subplots(1, 2, figsize=(3.25, 1.42), gridspec_kw=dict(left=0.125, right=0.985, bottom=0.235, top=0.9, wspace=0.42))

# (a) Dice against size
a.axhline(fit["teacher"], color=GRAY, lw=0.8, ls=(0, (3, 2)))
a.text(8500, fit["teacher"] + 0.001, "dense registration", color=GRAY, ha="right", va="bottom", fontsize=6)
a.plot(232, fit["rig_registration"], ls="none", marker="o", ms=4.6, color=BLUE, label="CardioRig, images")
a.plot(size.loc[VARIANTS, "size"], size.loc[VARIANTS, "dice"], ls="none", marker="o", ms=4.2, mfc="white", mec=BLUE, mew=1.0,
       label="CardioRig, registration")
a.plot(288, fit["spline_registration"], ls="none", marker="s", ms=4.2, color=RED, label="B-spline, images")
a.legend(loc="lower right", bbox_to_anchor=(1.04, -0.04), fontsize=6, frameon=False, handletextpad=0.1, labelspacing=0.3,
         borderpad=0)
a.set_xscale("log")
a.set_xlim(160, 9000)
a.set_ylim(0.733, 0.804)
a.set_xticks([232, 947, 6051], ["232", "947", "6,051"])
a.set_xticks([], minor=True)
a.set_yticks([0.76, 0.78, 0.80])
a.set_xlabel("parameters per patient", labelpad=2)
a.set_ylabel("ES Dice", labelpad=2)

# (b) error against the true motion over the cycle
LINES = [("identity", "no motion", LIGHT), ("teacher", "dense registration", GRAY), ("rig_fit_truth", "CardioRig", BLUE)]
for method, label, colour in LINES:
    per = frames[frames.method == method].groupby(["frame", "anatomy"]).epe.mean().unstack()
    tau = per.index / n_frames
    b.fill_between(tau, per.min(axis=1), per.max(axis=1), color=colour, alpha=0.22, lw=0)
    b.plot(tau, per.mean(axis=1), color=colour, lw=1.2, label=label)
b.legend(loc="upper right", bbox_to_anchor=(1.04, 1.1), fontsize=6, frameon=False, handlelength=1.2, handletextpad=0.5,
         labelspacing=0.3, borderpad=0)
b.set_xlim(0, 1)
b.set_ylim(0, 8.6)
b.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
b.set_yticks([0, 4, 8])
b.set_xlabel(r"cardiac phase $\tau$", labelpad=2)
b.set_ylabel("error (mm)", labelpad=2)

for ax, letter in ((a, "a"), (b, "b")):
    ax.tick_params(pad=2)
    ax.text(-0.3, 1.04, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", color=INK, ha="left", va="bottom")

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "compact.pdf")
print("saved", OUT / "compact.pdf")
print(size.loc[VARIANTS].round(4).to_string(), "\n", fit.round(4).to_dict())
