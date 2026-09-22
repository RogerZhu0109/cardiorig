"""
Tune the teacher's regularisation weight on ED -> ES of a fixed set of training
patients (the only frames with labels).  Metrics per patient: Dice of the warped
ED mask vs the ES mask (identity = no-warp floor), % non-positive Jacobian,
mean |det J - 1|.
Usage: python scripts/tune_teacher.py --lams 0.1 0.3 1 3 10 --out outputs/teacher
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.acdc import TUNE_PATIENTS, ACDCPatient  # noqa: E402
from src.teacher import register_batched  # noqa: E402
from src.utils.metrics import STRUCTURES, dice  # noqa: E402
from src.utils.svf import integrate, jacobian_det, warp  # noqa: E402


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = [ACDCPatient(Path(args.root) / "training" / p).processed() for p in TUNE_PATIENTS]
    ed = torch.from_numpy(np.concatenate([d["cine"][d["ed"]] for d in data]))[:, None].to(args.device)
    es = torch.from_numpy(np.concatenate([d["cine"][d["es"]] for d in data]))[:, None].to(args.device)
    splits = np.cumsum([len(d["z"]) for d in data])[:-1]

    rows = []
    for d in data:
        rows.append(dict(method="identity", lam="", pid=d["pid"], group=d["group"],
                         **dice(d["ed_mask"], d["es_mask"]), negJ_pct=0.0, absJ1=0.0))
    for lam in args.lams:
        t0 = time.time()
        v = register_batched(ed, es, lam=lam, iters=args.iters)
        u = integrate(v)
        jd = jacobian_det(u)
        onehot = torch.cat([F.one_hot(torch.from_numpy(d["ed_mask"]).long(), 4) for d in data])
        pred = warp(onehot.permute(0, 3, 1, 2).float().to(args.device), u).argmax(1).cpu().numpy()
        jd = jd.cpu().numpy()
        for d, p, j in zip(data, np.split(pred, splits), np.split(jd, splits)):
            rows.append(dict(method="svf_opt", lam=lam, pid=d["pid"], group=d["group"],
                             **dice(p, d["es_mask"]), negJ_pct=100 * (j <= 0).mean(), absJ1=np.abs(j - 1).mean()))
        print(f"lam {lam}: {len(ed)} pairs in {time.time() - t0:.0f}s")

    with open(out / "tune.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\n{'method':10s} {'lam':>5s} " + " ".join(f"{s:>5s}" for s in STRUCTURES.values()) + "  mean  negJ%  |J|-1")
    for key in dict.fromkeys((r["method"], r["lam"]) for r in rows):
        sel = [r for r in rows if (r["method"], r["lam"]) == key]
        ds = [np.nanmean([r[s] for r in sel]) for s in STRUCTURES.values()]
        print(f"{key[0]:10s} {str(key[1]):>5s} " + " ".join(f"{x:5.3f}" for x in ds)
              + f" {np.mean(ds):5.3f} {np.mean([r['negJ_pct'] for r in sel]):6.3f} {np.mean([r['absJ1'] for r in sel]):6.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/raw/ACDC")
    p.add_argument("--lams", type=float, nargs="+", default=[0.1, 0.3, 1.0, 3.0, 10.0])
    p.add_argument("--iters", type=int, default=150)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--out", default="outputs/teacher")
    main(p.parse_args())
