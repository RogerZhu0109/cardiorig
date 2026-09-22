"""
Full-sequence sanity check of the teacher (ED -> every frame) for a few patients:
folding over all frames, the ED -> ED residual, frame-to-frame SVF change, and the
LV / MYO area of the warped ED mask over the cycle (should be smooth, min near ES).
Usage: python scripts/inspect_teacher.py --patients patient001 patient022 --out outputs/teacher
"""
import argparse
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.acdc import ACDCPatient  # noqa: E402
from src.teacher import register_batched  # noqa: E402
from src.utils.svf import integrate, jacobian_det, warp  # noqa: E402


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(args.patients), figsize=(5 * len(args.patients), 3.5), squeeze=False)
    for pid, ax in zip(args.patients, axes[0]):
        d = ACDCPatient(Path(args.root) / "training" / pid).processed()
        (T, Z), s = d["cine"].shape[:2], d["cine"].shape[-1]
        cine = torch.from_numpy(d["cine"]).to(args.device)
        t0 = time.time()
        v = register_batched(cine[d["ed"]].repeat(T, 1, 1).reshape(T * Z, 1, s, s), cine.reshape(T * Z, 1, s, s), lam=args.lam)
        dt = time.time() - t0
        u = integrate(v)
        jd = jacobian_det(u)
        onehot = F.one_hot(torch.from_numpy(d["ed_mask"]).long(), 4).permute(0, 3, 1, 2).float().to(args.device)
        pred = warp(onehot.repeat(T, 1, 1, 1), u).argmax(1).reshape(T, Z, s, s).cpu().numpy()

        v = v.reshape(T, Z, 2, s, s)
        order = np.argsort(d["tau"])  # ED first, then along the cycle
        step = [(v[b] - v[a]).norm(dim=1).mean().item() for a, b in zip(order[:-1], order[1:])]
        mag = v.norm(dim=2).mean(dim=(1, 2, 3)).cpu().numpy()
        print(f"{pid} ({d['group']}): {T * Z} pairs in {dt:.0f}s | non-positive det J: {(jd <= 0).sum().item()} px "
              f"| ED->ED mean |v| {mag[d['ed']]:.3f} px | mean |v| {mag.mean():.2f} px "
              f"| frame-to-frame mean |dv| {np.mean(step):.3f} px (max {np.max(step):.3f})")

        px_ml = 1.5 * 1.5 * 1e-3  # mm^2 per px -> cm^2, summed over slices
        tau = d["tau"][order]
        for k, name in ((3, "LV"), (2, "MYO")):
            ax.plot(tau, (pred[order] == k).sum(axis=(1, 2, 3)) * px_ml, "o-", ms=3, label=f"{name} (warped ED mask)")
            ax.plot(d["tau"][d["es"]], (d["es_mask"] == k).sum() * px_ml, "x", ms=9, label=f"{name} ES ground truth")
        ax.axvline(d["tau"][d["es"]], ls=":", c="gray")
        ax.set_title(f"{pid} {d['group']}", fontsize=9); ax.set_xlabel("tau"); ax.set_ylabel("area summed over slices (cm^2)")
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "teacher_curves.png", dpi=120)
    print(f"figure -> {out}/teacher_curves.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/raw/ACDC")
    p.add_argument("--patients", nargs="+", default=["patient001", "patient022", "patient061"])
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--out", default="outputs/teacher")
    main(p.parse_args())
