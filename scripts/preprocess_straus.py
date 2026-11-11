"""
Build the STRAUS cache (simulated truth; evaluation only, never training): ACDC-style slices,
true labels and true motion (src/data/straus.py) plus teacher SVFs (ED -> every frame).
One file per sequence, out_dir/testing/patientNN_condition.npz, with the keys of the ACDC cache
(cine, svf, ed_mask, es_mask, z, tau, ed, es, group = condition) and masks, truth, through, aha, anatomy.
out_dir/qc.csv holds the sanity gates: blood-to-muscle contrast under the true labels at every frame,
and teacher against no motion on true ES motion and labels. Existing files are skipped.
Usage: python scripts/preprocess_straus.py --data_root data/raw/STRAUS --out_dir data/processed/straus
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.straus import STRAUSPatient, list_patients  # noqa: E402
from src.evaluation import endpoint_error, score  # noqa: E402
from src.teacher import register_batched  # noqa: E402


def process(pat: STRAUSPatient, args) -> tuple[dict, dict]:
    d = pat.processed(size=args.size)
    (T, Z), s, es = d["cine"].shape[:2], args.size, d["es"]
    cine = torch.from_numpy(d["cine"]).to(args.device)
    moving = cine[0].repeat(T - 1, 1, 1).reshape(-1, 1, s, s)
    v = register_batched(moving, cine[1:].reshape(-1, 1, s, s), chunk=args.chunk, lam=args.lam)
    svf = torch.cat([torch.zeros(1, Z, 2, s, s, device=args.device), v.reshape(T - 1, Z, 2, s, s)])

    truth, myo = torch.from_numpy(d["truth"][es]).to(args.device), torch.from_numpy(d["ed_mask"] == 2).to(args.device)
    contrast = [float(d["cine"][t][d["masks"][t] == 3].mean() - d["cine"][t][d["masks"][t] == 2].mean()) for t in range(T)]
    found, still = score(d, svf[es]), score(d, torch.zeros_like(svf[es]))
    qc = dict(pid=d["pid"], anatomy=d["anatomy"], condition=d["group"], slices=Z, es=es, turns=d["turns"],
              contrast_min=min(contrast), through_mm=float(np.abs(d["through"][es][d["ed_mask"] == 2]).mean()),
              epe_teacher=endpoint_error(svf[es], truth, myo), epe_identity=endpoint_error(torch.zeros_like(svf[es]), truth, myo),
              dice_teacher=np.nanmean([found[f"dice_{k}"] for k in ("RV", "MYO", "LV")]),
              dice_identity=np.nanmean([still[f"dice_{k}"] for k in ("RV", "MYO", "LV")]), folded_pct=found["folded_pct"])
    d["cine"], d["svf"] = d["cine"].astype(np.float16), svf.cpu().numpy().astype(np.float16)
    d["truth"], d["through"] = d["truth"].astype(np.float16), d["through"].astype(np.float16)
    return d, qc


def main(args):
    out = Path(args.out_dir)
    (out / "testing").mkdir(parents=True, exist_ok=True)
    qc_path = out / "qc.csv"
    rows = pd.read_csv(qc_path).to_dict("records") if qc_path.exists() else []
    for p in list_patients(Path(args.data_root)):
        if any(r["pid"] == p.name for r in rows):
            continue
        d, qc = process(STRAUSPatient(p), args)
        np.savez_compressed(out / "testing" / f"{p.name}.npz", **{k: np.asarray(x) for k, x in d.items()})
        rows.append(qc)
        pd.DataFrame(rows).to_csv(qc_path, index=False)
        print(f"{p.name}: {qc['slices']} slices, ES frame {qc['es']}, contrast min {qc['contrast_min']:.2f}, ES error "
              f"teacher {qc['epe_teacher']:.2f} mm vs no motion {qc['epe_identity']:.2f} mm, Dice {qc['dice_teacher']:.3f} vs "
              f"{qc['dice_identity']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    print(df.groupby("condition", sort=False)[["epe_teacher", "epe_identity", "dice_teacher", "dice_identity", "through_mm"]].mean().round(3).to_string())
    gates = {"cavity brighter than myocardium at every frame": df.contrast_min > 0,
             "teacher beats no motion on true ES motion and labels": (df.epe_teacher < df.epe_identity) & (df.dice_teacher > df.dice_identity)}
    for name, ok in gates.items():
        print(f"GATE {name}: {'PASS' if ok.all() else 'FAIL'} ({int(ok.sum())}/{len(ok)})")
    if not all(ok.all() for ok in gates.values()):
        sys.exit("a sanity gate failed: truth and images may be misaligned")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/raw/STRAUS")
    parser.add_argument("--out_dir", default="data/processed/straus")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available() else "cpu")
    main(parser.parse_args())
