"""
Paper Fig. 5: paired ES Dice differences, rig minus each other predictor, with 95% CIs, on the ACDC test set,
on M&Ms and on each M&Ms vendor (outputs/test/test.csv, outputs/mnms/mnms.csv). The CI is the two-level
bootstrap of scripts/evaluate_test.py (patients shared, each method's seeds independently; same random stream,
so the numbers are those of Tab. 2). As fixed in advance, PhaseFlow on M&Ms is judged on the subjects whose
pathology is an ACDC class.
CVPR column width (3.25 in); Arial, TrueType fonts embedded.
Usage: python figures/fig_forest.py  ->  figures/pdf/forest.pdf
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures" / "pdf"
GRAY = "#6E6E6E"
OTHERS = [("current", "dense-field", "#9A4D8E"), ("phaseflow_tau0", "PhaseFlow", "#42949E"), ("krebs_mean", "Krebs et al.", "#B64342")]
VENDORS = {"A": "Siemens", "B": "Philips", "C": "GE", "D": "Canon"}
plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "font.size": 7,
                     "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                     "xtick.major.width": 0.6, "xtick.major.size": 2.5, "pdf.fonttype": 42, "ps.fonttype": 42})


def matrix(df: pd.DataFrame, method: str) -> np.ndarray:
    return df[df.method == method].pivot(index="pid", columns="seed", values="dice").sort_index().to_numpy()


def two_level(df: pd.DataFrame, a: str, b: str, n_boot: int = 10000) -> tuple[float, float, float]:
    """Mean of a - b and its 95% CI, as scripts/evaluate_test.py two_level."""
    A, B = matrix(df, a), matrix(df, b)
    rng, n = np.random.default_rng(0), len(A)
    boots = np.empty(n_boot)
    for r in range(n_boot):
        p = rng.integers(n, size=n)
        boots[r] = A[p][:, rng.integers(A.shape[1], size=A.shape[1])].mean() - B[p][:, rng.integers(B.shape[1], size=B.shape[1])].mean()
    return (A.mean(1) - B.mean(1)).mean(), *np.percentile(boots, [2.5, 97.5])


def read(path: str) -> pd.DataFrame:
    df = pd.read_csv(REPO / path)
    return df.assign(dice=df[["dice_RV", "dice_MYO", "dice_LV"]].mean(axis=1))


acdc, mnms = read("outputs/test/test.csv"), read("outputs/mnms/mnms.csv")
rows = [("ACDC test", acdc, False), ("M&Ms", mnms, True)] + [(name, mnms[mnms.vendor == v], True) for v, name in VENDORS.items()]

fig, ax = plt.subplots(figsize=(3.25, 1.4), gridspec_kw=dict(left=0.19, right=0.985, bottom=0.24, top=0.92))
ax.axvline(0, color=GRAY, lw=0.6, zorder=0)
for i, (name, df, external) in enumerate(rows):
    for j, (method, label, colour) in enumerate(OTHERS):
        sub = df[df.pathology.isin(["DCM", "HCM", "NOR"])] if external and method == "phaseflow_tau0" else df
        mean, lo, hi = two_level(sub, "rig", method)
        y = i + 0.14 * (j - 1)
        ax.plot([lo, hi], [y, y], color=colour, lw=1.1, solid_capstyle="butt")
        ax.plot(mean, y, "o", color=colour, ms=2.8)
        print(f"{name:10s} rig - {method:15s} {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]  n={sub.pid.nunique()}")
        if i == 0:
            ax.text(mean, -0.5, label, color=colour, ha="center", va="bottom", fontsize=6.5)
ax.set_ylim(len(rows) - 0.5, -0.5)
ax.set_yticks(range(len(rows)), [r[0] for r in rows])
ax.tick_params(axis="y", length=0, pad=3)
for tick in ax.get_yticklabels()[2:]:
    tick.set_color(GRAY)
ax.set_xlim(-0.022, 0.098)
ax.set_xticks([-0.02, 0, 0.02, 0.04, 0.06, 0.08], ["−0.02", "0", "0.02", "0.04", "0.06", "0.08"])
ax.set_xlabel("ES Dice of CardioRig minus the other predictor", labelpad=2)

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "forest.pdf")
print("saved", OUT / "forest.pdf")
