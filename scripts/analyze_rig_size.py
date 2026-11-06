"""
How much ES Dice does the rig keep when its numbers are shared across slices and time?
Fits the teacher motion, so every row is a ceiling, not a prediction. Train + val patients only.
Usage: python scripts/analyze_rig_size.py --cache data/processed/acdc
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.cache import VAL_PATIENTS, load  # noqa: E402
from src.evaluation import score  # noqa: E402
from src.rig import fit_shared, patient_fields, time_basis  # noqa: E402


def smooth(tau: np.ndarray, harmonics: int) -> np.ndarray:
    """Projector onto the time curves, applied to coefficients (motions, frames) from the right."""
    phi = time_basis(tau, harmonics)
    return phi @ np.linalg.pinv(phi)


def fit(d: dict) -> list[dict]:
    """Rows of one patient: every variant's ES metrics, size and all-frame residual energy."""
    frames = [t for t in range(len(d["tau"])) if t != int(d["ed"])]
    es = frames.index(int(d["es"]))
    motion, tau = d["svf"][frames].astype(np.float32), d["tau"][frames]  # (F, Z, 2, H, W)
    fields, slots, regions = patient_fields(d["ed_mask"])
    per_slice = []
    for z, (b, region) in enumerate(zip(fields, regions)):
        y = motion[:, z][:, :, region].reshape(len(frames), -1)
        per_slice.append(np.linalg.lstsq(b[:, :, region].reshape(len(b), -1).T, y.T, rcond=1e-6)[0])  # (motions, frames)
    shared = fit_shared(fields, slots, regions, motion)
    used = len(set(np.concatenate(slots).tolist()))
    two, three = smooth(tau, 2), smooth(tau, 3)
    variants = {
        "per slice, per frame": ([c for c in per_slice], sum(map(len, per_slice)) * len(frames)),
        "per slice, smooth curves": ([c @ two for c in per_slice], sum(map(len, per_slice)) * 4),
        "shared slices, per frame": ([shared[i] for i in slots], used * len(frames)),
        "compact": ([(shared @ two)[i] for i in slots], used * 4),
        "compact, 6 curve numbers": ([(shared @ three)[i] for i in slots], used * 6),
    }
    base = dict(pid=str(d["pid"]), group=str(d["group"]))
    teacher, zero = motion[es], np.zeros_like(motion[es])
    device = args.device
    rows = [dict(base, variant=name, size=0, energy=e, **score(d, torch.from_numpy(f).float().to(device)))
            for name, f, e in (("identity", zero, 1.0), ("teacher", teacher, 0.0))]
    total = sum((motion[:, z, :, r] ** 2).sum() for z, r in enumerate(regions))
    for name, (coeffs, size) in variants.items():
        at_es = np.zeros_like(teacher)
        error = 0.0
        for z, (b, c, r) in enumerate(zip(fields, coeffs, regions)):
            recon = np.tensordot(c.T, b, 1)  # (F, 2, H, W)
            error += ((motion[:, z] - recon)[:, :, r] ** 2).sum()
            at_es[z] = recon[es]
        rows.append(dict(base, variant=name, size=size, energy=error / total,
                         **score(d, torch.from_numpy(at_es).float().to(device))))
    return rows


def report(rows: pd.DataFrame) -> None:
    rows["dice"] = rows[["dice_RV", "dice_MYO", "dice_LV"]].mean(axis=1)
    mean = rows.groupby("variant", sort=False)[["size", "dice", "dice_RV", "dice_MYO", "dice_LV", "energy", "folded_pct"]].mean()
    gap = mean.dice - mean.dice["identity"]
    mean["G"] = gap / gap["teacher"]
    mean["dice_val"] = rows[rows.pid.isin(VAL_PATIENTS)].groupby("variant").dice.mean()
    print(mean.round(4).to_string())


def main():
    patients = load(args.cache, "train") + load(args.cache, "val")
    rows = []
    for i, d in enumerate(patients):
        d["cine"] = None
        rows += fit(d)
        patients[i] = None
        print(f"{i + 1}/{len(patients)} {d['pid']}", flush=True)
    rows = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / "size.csv", index=False)
    report(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data/processed/acdc")
    p.add_argument("--out", default="outputs/rig")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    main()
