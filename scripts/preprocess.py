"""
Build the cache: heart-centred 128x128 slices + teacher SVFs (ED -> every frame).
One file per patient, out_dir/{training,testing}/patientXXX.npz:
    cine (T, Z, S, S) float16, svf (T, Z, 2, S, S) float16 (ED frame = 0),
    ed_mask, es_mask (Z, S, S) uint8, z, tau, center, ed, es, group
and out_dir/qc.csv: ES Dice of the warped ED mask (vs identity), folded pixels
over all frames, mean |det J - 1|, runtime.  Existing files are skipped (resumable).
Usage: python scripts/preprocess.py --data_root data/raw/ACDC --out_dir data/processed/acdc
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
from src.data.acdc import ACDCPatient, list_patients  # noqa: E402
from src.teacher import register_batched  # noqa: E402
from src.utils.metrics import STRUCTURES, dice  # noqa: E402
from src.utils.svf import integrate, jacobian_det, warp  # noqa: E402


def process(pat: ACDCPatient, args) -> tuple[dict, dict]:
    d = pat.processed(size=args.size)
    (T, Z), s = d["cine"].shape[:2], args.size
    frames = [t for t in range(T) if t != d["ed"]]
    cine = torch.from_numpy(d["cine"]).to(args.device)
    moving = cine[d["ed"]].repeat(len(frames), 1, 1).reshape(-1, 1, s, s)
    fixed = cine[frames].reshape(-1, 1, s, s)

    t0 = time.time()
    v = register_batched(moving, fixed, chunk=args.chunk, lam=args.lam)
    dt = time.time() - t0

    svf = torch.zeros(T, Z, 2, s, s, device=args.device)
    svf[frames] = v.reshape(len(frames), Z, 2, s, s)
    u = integrate(v)
    jd = jacobian_det(u)
    es = frames.index(d["es"])
    onehot = F.one_hot(torch.from_numpy(d["ed_mask"]).long(), 4).permute(0, 3, 1, 2).float().to(args.device)
    pred = warp(onehot, u.reshape(len(frames), Z, 2, s, s)[es]).argmax(1).cpu().numpy()

    d["cine"] = d["cine"].astype(np.float16)
    d["svf"] = svf.cpu().numpy().astype(np.float16)
    qc = dict(pid=d["pid"], split=pat.info.split, group=d["group"], pairs=len(frames) * Z, seconds=round(dt, 1),
              folded_px=int((jd <= 0).sum()), absJ1=(jd - 1).abs().mean().item(),
              **{f"dice_{k}": x for k, x in dice(pred, d["es_mask"]).items()},
              **{f"id_{k}": x for k, x in dice(d["ed_mask"], d["es_mask"]).items()})
    return d, qc


def main(args):
    out = Path(args.out_dir)
    qc_path = out / "qc.csv"
    done = set()
    if qc_path.exists():
        with open(qc_path) as f:
            done = {r["pid"] for r in csv.DictReader(f)}
    patients = list_patients(Path(args.data_root))
    print(f"{len(patients)} patients, {len(done)} already done, device {args.device}", flush=True)

    for p in patients:
        pat = ACDCPatient(p)
        if pat.pid in done:
            continue
        d, qc = process(pat, args)
        (out / pat.info.split).mkdir(parents=True, exist_ok=True)
        np.savez(out / pat.info.split / f"{pat.pid}.npz", **{k: np.asarray(x) for k, x in d.items()})
        new = not qc_path.exists()
        with open(qc_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=qc.keys())
            if new:
                w.writeheader()
            w.writerow(qc)
        mean_dice = np.nanmean([qc[f"dice_{k}"] for k in STRUCTURES.values()])
        print(f"{pat.pid} {qc['split']:8s} {qc['group']:4s} {qc['pairs']:4d} pairs {qc['seconds']:6.1f}s "
              f"ES Dice {mean_dice:.3f} folded {qc['folded_px']}", flush=True)

    with open(qc_path) as f:
        rows = list(csv.DictReader(f))
    for split in ("training", "testing"):
        sel = [r for r in rows if r["split"] == split]
        if sel:
            ds = {k: np.nanmean([float(r[f"dice_{k}"]) for r in sel]) for k in STRUCTURES.values()}
            ids = {k: np.nanmean([float(r[f"id_{k}"]) for r in sel]) for k in STRUCTURES.values()}
            print(f"{split}: {len(sel)} patients | ES Dice " + " ".join(f"{k} {x:.3f}" for k, x in ds.items())
                  + " | identity " + " ".join(f"{k} {x:.3f}" for k, x in ids.items())
                  + f" | folded px total {sum(int(r['folded_px']) for r in sel)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/raw/ACDC")
    parser.add_argument("--out_dir", default="data/processed/acdc")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available() else "cpu")
    main(parser.parse_args())
