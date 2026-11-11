"""
Read-out test (pre-registered in EXPERIMENTS.md, "Read-out test"): (A) segment wall motion read straight from
the rig numbers against the same measurement from the ED and ES labels; (B) a linear classifier of the disease
group on the rig numbers against a spline code fitted the same way. 100 training patients, nothing is trained
beyond the logistic regression.
Usage: python scripts/analyze_readout.py --cache data/processed/acdc [--fits outputs/rig_registration/fits]
       python scripts/analyze_readout.py                               (re-print from the files in --out)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.register_rig import SPLINE_GRID, code_fields  # noqa: E402
from src.data.cache import fold_patients, load  # noqa: E402
from src.rig import MIN_PX, OFFSET, SECTORS, fit_compact, fit_shared, positions, radial_at, rays, roi, sector_of_bins, septal_angle, thickening, time_basis  # noqa: E402
from src.utils.svf import integrate  # noqa: E402

MM = 1.5
PARTS = {"translation": range(0, 2), "LV endo radial": range(2, 50, 3), "LV epi radial": range(3, 50, 3),
         "LV tangential": range(4, 50, 3), "RV": range(50, 58)}  # shared motions of the rig


def spline_compact(d: dict) -> np.ndarray:
    """The spline code fitted to the teacher exactly like src.rig.fit_compact."""
    frames = [t for t in range(len(d["tau"])) if t != int(d["ed"])]
    fields, slots, size = code_fields(d["ed_mask"], "spline")
    shared = fit_shared(fields, slots, [roi(m) for m in d["ed_mask"]], d["svf"][frames].astype(np.float32), size)
    return shared @ np.linalg.pinv(time_basis(d["tau"][frames])).T


def segments(d: dict, codes: dict[str, np.ndarray], device) -> pd.DataFrame:
    """Per LV segment at ES (mm): endo inward motion and thickening from the labels, the teacher field and the numbers."""
    mask, es = d["ed_mask"], int(d["es"])
    septum, phi = septal_angle(mask), time_basis(d["tau"][[es]])[0]
    forward = integrate(torch.from_numpy(-d["svf"][es].astype(np.float32)).to(device)).cpu().numpy()
    rows = []
    for z, pos in positions(mask).items():
        if (d["es_mask"][z] == 3).sum() < MIN_PX:
            continue
        k, e = rays(mask[z]), rays(d["es_mask"][z])
        wall = np.where(k["has_wall"] & e["has_wall"], 1.0, np.nan)
        measures = dict(endo_label=(k["endo"] - e["endo"]) * wall, thick_label=((e["epi"] - e["endo"]) - (k["epi"] - k["endo"])) * wall,
                        endo_teacher=-radial_at(forward[z], k, k["endo"]) * wall, thick_teacher=thickening(forward[z], k) * wall)
        sector = sector_of_bins(septum, SECTORS[pos])
        for s in range(SECTORS[pos]):
            if np.isfinite(wall[sector == s]).any():
                rows.append(dict(seg=OFFSET[pos] + s, **{name: np.nanmean(x[sector == s]) * MM for name, x in measures.items()}))
    df = pd.DataFrame(rows).groupby("seg", as_index=False).mean()
    for name, numbers in codes.items():
        at_es = numbers @ phi  # (58,)
        df[f"endo_{name}"] = MM * at_es[2 + 3 * df.seg.to_numpy()]
        df[f"thick_{name}"] = MM * (at_es[2 + 3 * df.seg.to_numpy()] - at_es[3 + 3 * df.seg.to_numpy()])
    return df.assign(pid=d["pid"], group=str(d["group"]))


def collect(args) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    seg, codes, groups = [], {}, []
    for d in load(args.cache, "train") + load(args.cache, "val"):
        d["pid"] = str(d["pid"])
        own = dict(rig=fit_compact(d), spline=spline_compact(d))
        if args.fits:
            with np.load(Path(args.fits) / f"{d['pid']}.npz") as f:
                own.update(rig_images=f["rig_registration"], spline_images=f["spline_registration"])
        seg.append(segments(d, {k: v for k, v in own.items() if k.startswith("rig")}, args.device))
        for k, v in own.items():
            codes.setdefault(k, []).append(v.ravel())
        groups.append((d["pid"], str(d["group"])))
        print(f"{len(groups)} {d['pid']}", flush=True)
    codes = {k: np.stack(v) for k, v in codes.items()}
    return pd.concat(seg, ignore_index=True), dict(codes, pid=np.array([p for p, _ in groups]), group=np.array([g for _, g in groups]))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def report_segments(seg: pd.DataFrame, n_boot: int = 2000) -> None:
    seg = seg.dropna(subset=["endo_label", "thick_label"]).reset_index(drop=True)
    groups = list(seg.groupby("pid").indices.values())
    rng = np.random.default_rng(0)
    boots = [np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))]) for _ in range(n_boot)]
    within = seg.drop(columns=["seg", "group"]).groupby("pid").transform(lambda x: x - x.mean())
    print(f"== A. Numbers against label measurements, {seg.pid.nunique()} patients, {len(seg)} patient-segments, ES ==")
    for what in ("endo", "thick"):
        y, r = f"{what}_label", {}
        for src in [c[len(what) + 1:] for c in seg if c.startswith(what + "_") and c != y]:
            x = f"{what}_{src}"
            r[src] = np.array([pearson(seg[x].to_numpy()[b], seg[y].to_numpy()[b]) for b in boots])
            lo, hi = np.percentile(r[src], [2.5, 97.5])
            slope = np.polyfit(seg[x], seg[y], 1)[0]
            print(f"{what:5s} {src:11s} r = {pearson(seg[x], seg[y]):.3f} [{lo:.3f}, {hi:.3f}]  within-patient r = {pearson(within[x], within[y]):.3f}  "
                  f"label = {slope:.2f} x read  mean read - label = {(seg[x] - seg[y]).mean():+.2f} mm")
        for src in r:
            if src != "teacher":
                lo, hi = np.percentile(r[src] - r["teacher"], [2.5, 97.5])
                print(f"{what:5s} r({src}) - r(teacher field): {pearson(seg[f'{what}_{src}'], seg[y]) - pearson(seg[f'{what}_teacher'], seg[y]):+.3f} [{lo:+.3f}, {hi:+.3f}]")
    r, (lo, _) = pearson(seg.endo_rig, seg.endo_label), np.percentile(np.array(
        [pearson(seg.endo_rig.to_numpy()[b], seg.endo_label.to_numpy()[b]) - pearson(seg.endo_teacher.to_numpy()[b], seg.endo_label.to_numpy()[b]) for b in boots]), [2.5, 97.5])
    print(f"PRIMARY endo motion read from the rig numbers: {'PASS' if r >= 0.7 and lo > -0.1 else 'WEAK' if r >= 0.5 else 'FAIL'}")
    print("-- per group, endo r(rig numbers, label): " + ", ".join(f"{g} {pearson(x.endo_rig, x.endo_label):.2f}" for g, x in seg.groupby("group")))


def classify(x: np.ndarray, pid: np.ndarray, group: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Held-out predicted group per patient over the 5 CV folds, and the mean squared standardised weights (classes, features)."""
    pred, weight = np.empty(len(pid), dtype=object), 0
    for fold in range(5):
        test = np.isin(pid, fold_patients(fold))
        model = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1], cv=5, max_iter=5000))
        model.fit(x[~test], group[~test])
        pred[test] = model.predict(x[test])
        weight = weight + model[-1].coef_ ** 2 / 5
    return pred, weight


def report_classifier(codes: dict[str, np.ndarray], n_boot: int = 10000) -> None:
    pid, group = codes["pid"], codes["group"]
    names = [k for k in codes if k not in ("pid", "group")]
    idx = np.random.default_rng(0).integers(len(pid), size=(n_boot, len(pid)))
    hit, weights = {}, {}
    print(f"\n== B. Linear read-out of the disease group, {len(pid)} held-out patients, chance 0.20 ==")
    for name in names:
        pred, weights[name] = classify(codes[name], pid, group)
        hit[name] = (pred == group).astype(float)
        lo, hi = np.percentile(hit[name][idx].mean(1), [2.5, 97.5])
        print(f"{name:14s} ({codes[name].shape[1]} numbers) accuracy {hit[name].mean():.2f} [{lo:.2f}, {hi:.2f}]")
        if name == "rig":
            print(f"  carries the diagnosis: {'YES' if lo > 0.40 else 'NO'}")
            print(pd.crosstab(pd.Series(group, name="true"), pd.Series(pred, name="predicted")).to_string())
    for a, b in (("rig", "spline"), ("rig_images", "spline_images")):
        if a in hit:
            diff = hit[a] - hit[b]
            lo, hi = np.percentile(diff[idx].mean(1), [2.5, 97.5])
            verdict = "BETTER" if lo > 0 else "AT LEAST AS GOOD" if lo > -0.10 else "WORSE"
            print(f"accuracy {a} - {b}: {diff.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] -> {verdict}")
    w = weights["rig"].reshape(len(weights["rig"]), 58, -1).sum(2)  # (classes, shared motions)
    share = pd.DataFrame({part: w[:, list(rows)].sum(1) for part, rows in PARTS.items()}, index=sorted(set(group)))
    print("-- share of each class's squared weight by kind of rig number (share of numbers: "
          + ", ".join(f"{part} {len(rows) / 58:.2f}" for part, rows in PARTS.items()) + ")")
    print(share.div(share.sum(1), axis=0).round(2).to_string())


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.cache:
        seg, codes = collect(args)
        seg.to_csv(out / "segments.csv", index=False)
        np.savez(out / "codes.npz", **codes)
    pd.set_option("display.width", 250)
    report_segments(pd.read_csv(out / "segments.csv"))
    with np.load(out / "codes.npz") as f:
        report_classifier({k: f[k] for k in f.files})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache")
    p.add_argument("--fits", help="per-patient image fits of scripts/register_rig.py")
    p.add_argument("--out", default="outputs/readout")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(p.parse_args())
