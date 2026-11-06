"""
Are the rig model's per-segment thickening ranges calibrated against real ES labels, and do they
beat a population range? Uses saved held-out predictions (scripts/evaluate_cv.py); train + val only.
Usage: python scripts/analyze_uncertainty.py --predictions outputs/cv/rig_predictions.npz --truth outputs/rig/segments.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import norm, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.cache import fold_patients, load  # noqa: E402
from src.rig import (BIN_CENTRES, OFFSET, SECTORS, patient_fields, positions, rays, sector_of_bins,  # noqa: E402
                     septal_angle, time_basis)
from src.utils.svf import integrate  # noqa: E402

MM = 1.5
LEVELS = (0.5, 0.8, 0.9)


def radial(w: torch.Tensor, k: dict, radius: np.ndarray) -> torch.Tensor:
    """Radial displacement of a batch w (K, 2, H, W) along a curve, (K, bins)."""
    c, s = (torch.tensor(f(BIN_CENTRES), dtype=w.dtype, device=w.device) for f in (np.cos, np.sin))
    r = torch.tensor(radius, dtype=w.dtype, device=w.device)
    h, n = w.shape[-2:]
    x, y = 2 * (k["cx"] + r * c) / (n - 1) - 1, 2 * (k["cy"] + r * s) / (h - 1) - 1
    grid = torch.stack([x, y], -1)[None, None].expand(len(w), -1, -1, -1)
    at = F.grid_sample(w, grid, mode="bilinear", align_corners=True, padding_mode="border")[:, :, 0]
    return at[:, 0] * c + at[:, 1] * s


def segment_thickening(motion: dict, mask: np.ndarray, device) -> tuple[torch.Tensor, np.ndarray]:
    """Thickening per segment (K, 16) in mm, from SVFs motion[z] (K, 2, H, W), and which segments exist."""
    septum, total, count = septal_angle(mask), None, np.zeros(16)
    for z, pos in positions(mask).items():
        k = rays(mask[z])
        w = integrate(-motion[z])
        dt = (radial(w, k, k["epi"]) - radial(w, k, k["endo"])) * MM
        wall = torch.from_numpy(k["has_wall"]).to(device)
        total = torch.zeros(len(w), 16, device=device) if total is None else total
        sector = sector_of_bins(septum, SECTORS[pos])
        for s in range(SECTORS[pos]):
            bins = torch.from_numpy(sector == s).to(device) & wall
            if bins.any():
                seg = OFFSET[pos] + s
                total[:, seg] += dt[:, bins].mean(1)
                count[seg] += 1
    return total / torch.from_numpy(np.maximum(count, 1)).float().to(device), count > 0


@torch.no_grad()
def predict(d: dict, numbers: np.ndarray, spread: np.ndarray, draws: int, gen: torch.Generator, device) -> pd.DataFrame:
    """Mean and sd of segment thickening under the predicted distribution of the rig numbers."""
    mask = d["ed_mask"]
    fields, slots, _ = patient_fields(mask)
    mean, sd = torch.from_numpy(numbers).float().to(device), torch.from_numpy(spread).float().to(device)
    sample = mean + sd * torch.randn(draws, *mean.shape, generator=gen).to(device)
    phi = torch.from_numpy(time_basis(np.array([float(d["tau"][int(d["es"])])]), 2)[0]).float().to(device)
    c = sample @ phi  # (draws, motions)
    motion = {z: torch.einsum("kp,pchw->kchw", c[:, slots[z]], torch.from_numpy(fields[z]).float().to(device))
              for z in positions(mask)}
    out, seen = segment_thickening(motion, mask, device)
    keep = np.nonzero(seen)[0]
    return pd.DataFrame(dict(seg=keep, mu=out[:, keep].mean(0).cpu().numpy(), sd=out[:, keep].std(0).cpu().numpy()))


def quantile(x: np.ndarray, level: float) -> float:
    """Finite-sample conformal quantile."""
    return float(np.quantile(x, min(1.0, np.ceil((len(x) + 1) * level) / len(x)), method="higher"))


def cross_fit(df: pd.DataFrame, level: float) -> pd.DataFrame:
    """Calibrated intervals for every row, each fitted on the other folds."""
    parts = []
    for fold, test in df.groupby("fold"):
        cal, test = df[df.fold != fold], test.copy()
        bias = (cal.y - cal.mu).groupby(cal.seg).mean()
        centre = cal.groupby("seg").y.mean()
        scale = quantile(((cal.y - cal.mu - cal.seg.map(bias)).abs() / cal.sd).to_numpy(), level)
        fixed = (cal.y - cal.mu - cal.seg.map(bias)).abs().groupby(cal.seg).apply(lambda e: quantile(e.to_numpy(), level))
        wide = (cal.y - cal.seg.map(centre)).abs().groupby(cal.seg).apply(lambda e: quantile(e.to_numpy(), level))
        mid = test.mu + test.seg.map(bias)
        test["rig_lo"], test["rig_hi"] = mid - scale * test.sd, mid + scale * test.sd
        test["fixed_lo"], test["fixed_hi"] = mid - test.seg.map(fixed), mid + test.seg.map(fixed)
        pop = test.seg.map(centre)
        test["population_lo"], test["population_hi"] = pop - test.seg.map(wide), pop + test.seg.map(wide)
        test["error"] = (test.y - mid).abs()
        parts.append(test)
    return pd.concat(parts, ignore_index=True)


def scored(df: pd.DataFrame, name: str, alpha: float = 0.1) -> pd.DataFrame:
    lo, hi = df[f"{name}_lo"], df[f"{name}_hi"]
    miss = (lo - df.y).clip(lower=0) + (df.y - hi).clip(lower=0)
    return pd.DataFrame(dict(pid=df.pid, group=df.group, inside=(miss == 0).astype(float), width=hi - lo, score=hi - lo + 2 / alpha * miss))


def ratio_ci(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 5000) -> tuple[float, float, float]:
    """Mean interval score ratio a / b, patient bootstrap."""
    pa, pb = a.groupby("pid").score.sum(), b.groupby("pid").score.sum()
    rng, m = np.random.default_rng(0), len(pa)
    idx = rng.integers(m, size=(n_boot, m))
    boots = pa.to_numpy()[idx].sum(1) / pb.to_numpy()[idx].sum(1)
    return pa.sum() / pb.sum(), *np.percentile(boots, [2.5, 97.5])


def report(df: pd.DataFrame) -> None:
    print(f"{df.pid.nunique()} patients, {len(df)} segments, truth sd {df.y.std():.2f} mm, rig mean r {np.corrcoef(df.y, df.mu)[0, 1]:.3f}")
    for level in LEVELS:
        z = norm.ppf(0.5 + level / 2)
        print(f"raw coverage at {level:.0%}: {((df.y - df.mu).abs() <= z * df.sd).mean():.3f}")
    fit = cross_fit(df, 0.9)
    results = {name: scored(fit, name) for name in ("rig", "fixed", "population")}
    print(pd.DataFrame({k: dict(coverage=v.inside.mean(), width=v.width.mean(), score=v.score.mean())
                        for k, v in results.items()}).T.round(3).to_string())
    print("coverage by group:\n" + pd.DataFrame({k: v.groupby("group").inside.mean() for k, v in results.items()}).round(3).to_string())
    print("interval score by group:\n" + pd.DataFrame({k: v.groupby("group").score.mean() for k, v in results.items()}).round(3).to_string())
    for name in ("population", "fixed"):
        print("score ratio rig / {}: {:.3f} [{:.3f}, {:.3f}]".format(name, *ratio_ci(results["rig"], results[name])))
    print(f"spread vs error Spearman r: {spearmanr(fit.sd, fit.error).statistic:.3f}")


def main(args):
    saved = np.load(args.predictions)
    index = {str(p): i for i, p in enumerate(saved["pids"])}
    truth = pd.read_csv(args.truth).dropna(subset=["dt_true"])[["pid", "seg", "group", "dt_true"]]
    fold = {p: k for k in range(5) for p in fold_patients(k)}
    gen, rows = torch.Generator().manual_seed(0), []
    for d in load(args.cache, "train", svf=False) + load(args.cache, "val", svf=False):
        pid = str(d["pid"])
        i = index[pid]
        rows.append(predict(d, saved["numbers"][i], saved["spread"][i], args.draws, gen, args.device).assign(pid=pid, fold=fold[pid]))
        print(pid, flush=True)
    df = pd.concat(rows, ignore_index=True).merge(truth.rename(columns=dict(dt_true="y")), on=["pid", "seg"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "segments.csv", index=False)
    report(df)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", default="outputs/cv/rig_predictions.npz")
    p.add_argument("--truth", default="outputs/rig/segments.csv")
    p.add_argument("--cache", default="data/processed/acdc")
    p.add_argument("--out", default="outputs/uncertainty")
    p.add_argument("--draws", type=int, default=200)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(p.parse_args())
