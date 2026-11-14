"""
External reference for the teacher: ANTs SyN (CC) on the same ED -> ES slices and
metrics as scripts/tune_teacher.py.  Not used to produce targets.

Runs in a separate environment with antspyx and nibabel.
Usage: <ants-venv>/bin/python scripts/syn_reference.py --flow_sigmas 1 2 3 --radius 10 --out outputs/teacher
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import ants
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.acdc import TUNE_PATIENTS, ACDCPatient  # noqa: E402
from src.utils.metrics import STRUCTURES, dice, lv_areas, lv_captured  # noqa: E402


def syn(ed, es, ed_mask, flow_sigma, radius):
    fixed, moving = ants.from_numpy(es), ants.from_numpy(ed)
    r = ants.registration(fixed, moving, type_of_transform="SyNOnly", syn_metric="CC", syn_sampling=radius,
                          reg_iterations=(100, 70, 50), grad_step=0.1, flow_sigma=flow_sigma, total_sigma=0)
    mask = ants.apply_transforms(fixed, ants.from_numpy(ed_mask.astype(np.float32)), r["fwdtransforms"],
                                 interpolator="genericLabel").numpy()
    jd = ants.create_jacobian_determinant_image(fixed, r["fwdtransforms"][0]).numpy()
    return mask.round().astype(np.uint8), jd


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = [ACDCPatient(Path(args.root) / "training" / p).processed() for p in TUNE_PATIENTS]
    rows = []
    for fs in args.flow_sigmas:
        t0 = time.time()
        for d in data:
            res = [syn(d["cine"][d["ed"], z], d["cine"][d["es"], z], d["ed_mask"][z], fs, args.radius)
                   for z in range(len(d["z"]))]
            pred, jd = np.stack([r[0] for r in res]), np.stack([r[1] for r in res])
            rows.append(dict(method="syn", flow_sigma=fs, pid=d["pid"], group=d["group"],
                             **dice(pred, d["es_mask"]), negJ_pct=100 * (jd <= 0).mean(), absJ1=np.abs(jd - 1).mean(),
                             **lv_areas(d, pred)))
        print(f"flow_sigma {fs}: {sum(len(d['z']) for d in data)} pairs in {time.time() - t0:.0f}s")

    with open(out / "syn.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\n{'flow_sigma':>10s} " + " ".join(f"{s:>5s}" for s in STRUCTURES.values()) + "  mean  negJ%  |J|-1  LV contraction captured")
    for fs in args.flow_sigmas:
        sel = [r for r in rows if r["flow_sigma"] == fs]
        ds = [np.nanmean([r[s] for r in sel]) for s in STRUCTURES.values()]
        print(f"{fs:>10} " + " ".join(f"{x:5.3f}" for x in ds)
              + f" {np.mean(ds):5.3f} {np.mean([r['negJ_pct'] for r in sel]):6.3f} {np.mean([r['absJ1'] for r in sel]):6.3f}"
              + f"  {lv_captured(sel):.1%}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/raw/ACDC")
    p.add_argument("--flow_sigmas", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    p.add_argument("--radius", type=int, default=10, help="CC radius; 10 = the teacher's 21-px LNCC window")
    p.add_argument("--out", default="outputs/teacher")
    main(p.parse_args())
