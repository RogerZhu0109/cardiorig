"""
Sanity check for ACDCPatient.processed(): the heart-centred crop keeps every
labelled pixel, plus a figure of ED / ES mid slices for 2 patients per group.
Usage: python scripts/inspect_crops.py --root data/raw/ACDC --out outputs/inspect
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.acdc import GROUPS, ACDCPatient, list_patients  # noqa: E402


def main(args):
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lost, dropped, shown = [], 0, {g: [] for g in GROUPS}
    for p in list_patients(root):
        pat = ACDCPatient(p)
        d = pat.processed(size=args.size)
        full = sum((pat.resample_inplane(pat.mask(t), order=0) > 0).sum() for t in (pat.info.ed, pat.info.es))
        kept = (d["ed_mask"] > 0).sum() + (d["es_mask"] > 0).sum()
        if kept != full:
            lost.append((pat.pid, int(full - kept)))
        dropped += pat.shape[2] - len(d["z"])
        if pat.info.split == "training" and len(shown[d["group"]]) < 2:
            shown[d["group"]].append(d)
    print(f"crop {args.size}x{args.size}: patients losing labelled pixels: {lost or 'none'}")
    print(f"slices dropped (no ED foreground): {dropped}")

    fig, ax = plt.subplots(4, 5, figsize=(12, 10))
    for k, d in enumerate(ds for g in GROUPS for ds in shown[g]):
        z = len(d["z"]) // 2
        for r, (name, t, m) in enumerate((("ED", d["ed"], d["ed_mask"]), ("ES", d["es"], d["es_mask"]))):
            a = ax[2 * (k // 5) + r, k % 5]
            a.imshow(d["cine"][t, z], cmap="gray", vmin=0, vmax=1)
            a.contour(m[z], levels=[0.5, 1.5, 2.5], colors=["c", "y", "r"], linewidths=0.7)
            a.set_title(f"{d['pid']} {d['group']} {name}", fontsize=8)
            a.axis("off")
    fig.suptitle(f"heart-centred {args.size}x{args.size} crops @1.5 mm, mid slice")
    fig.tight_layout(); fig.savefig(out / "crops.png", dpi=120)
    print(f"figure -> {out}/crops.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/raw/ACDC")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--out", default="outputs/inspect")
    main(p.parse_args())
