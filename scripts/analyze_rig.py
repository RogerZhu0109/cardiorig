"""
Can the rig represent the dense registration, and does the ED frame predict regional
thickening? Training patients only; ES labels only score.
Usage: python scripts/analyze_rig.py --cache data/processed/acdc
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.cache import VAL_PATIENTS, load  # noqa: E402
from src.evaluation import score  # noqa: E402
from src.rig import (MIN_PX, OFFSET, SECTORS, basis, positions, project, rays, roi,  # noqa: E402
                     sector_of_bins, septal_angle, spline_basis, thickening)
from src.utils.svf import integrate  # noqa: E402

MM = 1.5  # px size
FEATURES = ["seg", "t_ed", "t_rel", "t_mean", "lv_area"]
TARGETS = {"teacher deviation (primary)": "dt_teacher_dev", "teacher absolute": "dt_teacher", "ES-label deviation": "dt_true_dev"}


def reconstruct(d: dict, v: np.ndarray, where: dict[int, str], septum: float) -> dict[str, np.ndarray]:
    """Project each slice's teacher SVF; returns the reconstructions (Z, 2, H, W)."""
    out = {k: np.zeros_like(v) for k in ("rig", "rig_residual", "spline")}
    for z, mask in enumerate(d["ed_mask"]):
        region, control = roi(mask), spline_basis(mask)
        out["rig"][z] = project(basis(mask, septum, SECTORS[where.get(z, "mid")]), v[z], region)
        out["rig_residual"][z] = out["rig"][z] + project(control, v[z] - out["rig"][z], region)
        out["spline"][z] = project(control, v[z], region)
    return out


def expressivity(d: dict, v: np.ndarray, recon: dict[str, np.ndarray], device: str) -> list[dict]:
    pid, group, heart = str(d["pid"]), str(d["group"]), np.stack([roi(m) for m in d["ed_mask"]])
    fields = dict(identity=np.zeros_like(v), teacher=v, **recon)
    rows = []
    for name, field in fields.items():
        rows.append(dict(pid=pid, group=group, method=name,
                         residual_energy=((v - field) ** 2).sum(1)[heart].sum() / (v ** 2).sum(1)[heart].sum(),
                         **score(d, torch.from_numpy(field).float().to(device))))
    return rows


def segments(d: dict, v: np.ndarray, where: dict[int, str], septum: float, device: str) -> pd.DataFrame:
    """Per patient-segment: ED thickness, teacher thickening, ES-label thickening (mm, mean over slices)."""
    forward = integrate(torch.from_numpy(-v).float().to(device)).cpu().numpy()
    rows = []
    for z, pos in where.items():
        ed, es = d["ed_mask"][z], d["es_mask"][z]
        k = rays(ed)
        t_ed = np.where(k["has_wall"], k["epi"] - k["endo"], np.nan) * MM
        dt_teacher = np.where(k["has_wall"], thickening(forward[z], k), np.nan) * MM
        dt_true = np.full_like(t_ed, np.nan)
        if (es == 3).sum() >= MIN_PX:
            e = rays(es)
            dt_true = np.where(e["has_wall"], (e["epi"] - e["endo"]) * MM, np.nan) - t_ed
        sector = sector_of_bins(septum, SECTORS[pos])
        for s in range(SECTORS[pos]):
            if np.isfinite(t_ed[sector == s]).any():
                rows.append(dict(seg=OFFSET[pos] + s, t_ed=np.nanmean(t_ed[sector == s]),
                                 dt_teacher=np.nanmean(dt_teacher[sector == s]),
                                 dt_true=np.nanmean(dt_true[sector == s]) if np.isfinite(dt_true[sector == s]).any() else np.nan))
    df = pd.DataFrame(rows).groupby("seg", as_index=False).mean()
    df["pid"], df["group"] = str(d["pid"]), str(d["group"])
    df["lv_area"] = (d["ed_mask"] == 3).sum() * MM ** 2
    return df


def add_derived(seg: pd.DataFrame) -> pd.DataFrame:
    by = seg.groupby("pid")
    seg = seg.assign(t_mean=by.t_ed.transform("mean"))
    return seg.assign(t_rel=seg.t_ed - seg.t_mean, dt_teacher_dev=seg.dt_teacher - by.dt_teacher.transform("mean"),
                      dt_true_dev=seg.dt_true - by.dt_true.transform("mean"))


def demeaned(seg: pd.DataFrame, col: str) -> pd.Series:
    """Remove patient means, then segment means."""
    x = seg[col] - seg.groupby("pid")[col].transform("mean")
    return x - x.groupby(seg.seg).transform("mean")


def boot_pids(frame: pd.DataFrame, n: int, seed: int = 0) -> list[np.ndarray]:
    """Row-index sets of n patient-level bootstrap resamples."""
    groups = list(frame.groupby("pid").indices.values())
    rng = np.random.default_rng(seed)
    return [np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))]) for _ in range(n)]


def skill(seg: pd.DataFrame, target: str, n_boot: int, group: str | None = None) -> dict:
    """Held-out skill of the ED model over the population segment-mean baseline."""
    data = seg.dropna(subset=[target]).reset_index(drop=True)
    model_pred, base_pred = np.zeros(len(data)), np.zeros(len(data))
    folds = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    for tr, te in folds.split(data, data.group, data.pid):
        model = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0,
                                              categorical_features=[0])
        model.fit(data.loc[tr, FEATURES], data.loc[tr, target])
        model_pred[te] = model.predict(data.loc[te, FEATURES])
        base_pred[te] = data.loc[te, "seg"].map(data.loc[tr].groupby("seg")[target].mean())
    data["e_model"], data["e_base"] = (data[target] - model_pred) ** 2, (data[target] - base_pred) ** 2
    data = data[data.group == group].reset_index(drop=True) if group else data
    resamples = boot_pids(data, n_boot)
    s = 1 - data.e_model.sum() / data.e_base.sum()
    boots = [1 - data.e_model.values[i].sum() / data.e_base.values[i].sum() for i in resamples]
    return dict(s=s, lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5))


def partial_r(seg: pd.DataFrame, a: str, b: str, n_boot: int) -> dict:
    data = seg.dropna(subset=[a, b]).reset_index(drop=True)
    x, y = demeaned(data, a).to_numpy(), demeaned(data, b).to_numpy()
    boots = [np.corrcoef(x[i], y[i])[0, 1] for i in boot_pids(data, n_boot)]
    return dict(r=np.corrcoef(x, y)[0, 1], lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5))


def report(rows: pd.DataFrame, seg: pd.DataFrame, n_boot: int) -> None:
    rows["dice"] = rows[["dice_RV", "dice_MYO", "dice_LV"]].mean(axis=1)
    mean = rows.groupby("method")[["dice", "dice_RV", "dice_MYO", "dice_LV", "residual_energy", "folded_pct"]].mean()
    gap = mean.dice - mean.dice["identity"]
    mean["G"] = gap / gap["teacher"]
    mean["dice_val"] = rows[rows.pid.isin(VAL_PATIENTS)].groupby("method").dice.mean()
    print(f"Test 1: expressivity, {rows.pid.nunique()} patients")
    print(mean.round(4).to_string())
    print(f"\nTest 2: predictability, {seg.pid.nunique()} patients, {len(seg)} segments")
    validity = partial_r(seg, "dt_teacher", "dt_true", n_boot)
    print(f"teacher validity r {validity['r']:.3f} [{validity['lo']:.3f}, {validity['hi']:.3f}]")
    for label, col in TARGETS.items():
        for group in (None, "MINF"):
            k = skill(seg, col, n_boot, group)
            print(f"S {label:29s} {group or 'all':4s} {k['s']:+.3f} [{k['lo']:+.3f}, {k['hi']:+.3f}]")
    for name, part in (("all", seg), ("MINF", seg[seg.group == "MINF"])):
        k = partial_r(part, "t_ed", "dt_teacher", n_boot)
        print(f"partial r T_ED vs dT {name:4s} {k['r']:+.3f} [{k['lo']:+.3f}, {k['hi']:+.3f}]")


def main(args):
    patients = load(args.cache, "train") + load(args.cache, "val")
    rows, seg = [], []
    for i, d in enumerate(patients):
        d["svf"] = d["svf"][int(d["es"])].astype(np.float32)  # only ES is needed
        d["cine"] = None
        v = d["svf"]
        septum = septal_angle(d["ed_mask"])
        where = positions(d["ed_mask"])
        rows += expressivity(d, v, reconstruct(d, v, where, septum), args.device)
        seg.append(segments(d, v, where, septum, args.device))
        print(f"{i + 1}/{len(patients)} {d['pid']} septum {np.degrees(septum):.0f} deg", flush=True)
    rows, seg = pd.DataFrame(rows), add_derived(pd.concat(seg, ignore_index=True))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / "expressivity.csv", index=False)
    seg.to_csv(out / "segments.csv", index=False)
    report(rows, seg, args.boot)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data/processed/acdc")
    p.add_argument("--out", default="outputs/rig")
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(p.parse_args())
