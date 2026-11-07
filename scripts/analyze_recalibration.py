"""
Contraction recalibration (EXPERIMENTS.md): scale the predicted ES motion by s, with s cross-fitted by fold to match
pooled LV contraction, for the rig and the current model on the 100 CV held-out patients. Then re-run the
report edits on the recalibrated rig. ES labels of the other folds set s; train + val only.
Usage: python scripts/analyze_recalibration.py --out outputs/recalibration
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_edits import Patient, outcome, report_edits, reports  # noqa: E402
from scripts.evaluate_cv import load_current, paired  # noqa: E402
from src.data.cache import load  # noqa: E402
from src.evaluation import es_inputs  # noqa: E402
from src.models.flow import sample  # noqa: E402
from src.rig import SHARED  # noqa: E402

GRID = np.round(np.arange(0.80, 2.0001, 0.05), 2)
RADIAL = np.array([2 + 3 * k + j for k in range(16) for j in (0, 1)] + list(range(50, SHARED, 2)))
FLAGS = ("akinetic", "hypokinetic", "normal")


def scaled_radial(numbers: np.ndarray, s: float) -> np.ndarray:
    out = numbers.copy()
    out[RADIAL] *= s
    return out


def rows_for(p: Patient, method: str, v: torch.Tensor, flags: dict[int, str], fold: int) -> list[dict]:
    """Metrics of one method's ES motion v (GRID, Z, 2, H, W), one row per s."""
    rows = []
    for s, r in zip(GRID, outcome(p, v)):
        endo = r.pop("endo_err")
        row = dict(pid=str(p.d["pid"]), group=str(p.d["group"]), fold=fold, method=method, s=s,
                   ed_lv=int((p.d["ed_mask"] == 3).sum()), gt_lv=int((p.d["es_mask"] == 3).sum()), **r)
        for f in FLAGS:
            segs = [k for k in range(16) if flags.get(k, "normal") == f and np.isfinite(endo["excursion_true"][k])]
            for key in ("err", "excursion_true", "excursion_pred"):
                row[f"{key}_{f}"] = np.nanmean(endo[key][segs]) if segs else np.nan
        rows.append(row)
    return rows


def contraction(df: pd.DataFrame) -> float:
    return 1 - df.pred_lv.sum() / df.ed_lv.sum()


def fit_s(df: pd.DataFrame) -> float:
    """Grid s whose pooled predicted contraction is closest to the true pooled contraction."""
    true = 1 - df.gt_lv.sum() / df.ed_lv.sum()
    gap = df.groupby("s").apply(lambda g: abs(contraction(g) - true), include_groups=False)
    return float(gap.idxmin())


def ci(x: np.ndarray, n_boot: int = 10000) -> str:
    boots = np.random.default_rng(0).choice(x, (n_boot, len(x))).mean(1)
    return f"{x.mean():+.4f} [{np.percentile(boots, 2.5):+.4f}, {np.percentile(boots, 97.5):+.4f}]"


def summarise(grid: pd.DataFrame, chosen: dict, edits: pd.DataFrame) -> None:
    grid = grid.assign(dice=grid[["dice_RV", "dice_MYO", "dice_LV"]].mean(axis=1))
    base = grid[grid.s == 1.0].assign(variant="unscaled")
    recal = pd.concat([g[g.s == chosen[(m, f)]] for (m, f), g in grid.groupby(["method", "fold"])]).assign(variant="recalibrated")
    both = pd.concat([base, recal])
    print("s per fold:", {f"{k[0]} f{k[1]}": v for k, v in chosen.items() if k[1] != "all"})
    print("s on all 100 (test recipe):", {k[0]: v for k, v in chosen.items() if k[1] == "all"})
    cols = ["dice", "dice_RV", "dice_MYO", "dice_LV", "folded_px"]
    print(both.groupby(["method", "variant"])[cols].agg({**{c: "mean" for c in cols[:-1]}, "folded_px": "sum"}).round(4).to_string())
    print("pooled LV contraction:", both.groupby(["method", "variant"]).apply(contraction, include_groups=False).round(3).to_dict(),
          "true", round(1 - base[base.method == "rig"].gt_lv.sum() / base[base.method == "rig"].ed_lv.sum(), 3))
    for m in ("rig", "rig_radial", "current"):
        wide = both[both.method == m].pivot(index="pid", columns="variant", values="dice")
        print(f"[{m}]", paired(wide, "recalibrated", "unscaled"))
    wide = recal.pivot(index="pid", columns="method", values="dice")
    print("[after recalibration]", paired(wide, "rig", "current"))
    print("Dice by group:")
    print(both.pivot_table(index="group", columns=["method", "variant"], values="dice").round(3).to_string())
    drop = both.pivot_table(index="group", columns=["method", "variant"], values="dice")
    print("largest group drop, rig:", round(float((drop[("rig", "recalibrated")] - drop[("rig", "unscaled")]).min()), 4))
    print("Endo error / excursion by report flag (mm, mean over patients):")
    flag_cols = [f"{k}_{f}" for f in FLAGS for k in ("err", "excursion_true", "excursion_pred")]
    print(both.groupby(["method", "variant"])[flag_cols].mean().round(2).T.to_string())
    print(f"(folded_px is a total over {base.pid.nunique()} patients)")

    print("\nReport edits on the recalibrated rig (same flags and alpha):")
    flagged = edits[(edits.n_flagged > 0) & edits.method.isin(["rig", "rig_report", "current", "current_mask"])]
    flagged = flagged.assign(dice=flagged[["dice_RV", "dice_MYO", "dice_LV"]].mean(axis=1))
    print(flagged.groupby("method")[["endo_flagged", "endo_unflagged", "dice"]].mean().round(4).to_string())
    for metric in ("endo_flagged", "dice"):
        wide = flagged.pivot(index="pid", columns="method", values=metric)
        print(f"[{metric}]", paired(wide, "rig_report", "rig"))


def main(args):
    pred = np.load(args.predictions)
    numbers = dict(zip(pred["pids"].tolist(), pred["numbers"]))
    truth = pd.read_csv(args.truth)
    flags = reports(truth)
    truth_by = {pid: dict(zip(g.seg.astype(int), g.dt_true)) for pid, g in truth.groupby("pid")}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start, rows = time.time(), []
    for fold in range(5):
        current, sigma = load_current(Path(args.runs) / f"current_f{fold}" / "last.pt", args.device)
        for d in load(args.cache, "val", svf=False, features=args.features, fold=fold)[:args.limit or None]:
            pid = str(d["pid"])
            with torch.no_grad():
                p = Patient(d, numbers[pid], args.device)
                v_rig = p.decode([numbers[pid]])[0]
                v_rad = p.decode([scaled_radial(numbers[pid], s) for s in GRID])
                ed, tau, cond = es_inputs(d, args.device)
                v_cur = sample(current, "regress", ed, tau, sigma, 1, **cond).float()
                s = torch.tensor(GRID, dtype=torch.float32, device=args.device)[:, None, None, None, None]
                rows += rows_for(p, "rig", v_rig[None] * s, flags[pid], fold)
                rows += rows_for(p, "rig_radial", v_rad, flags[pid], fold)
                rows += rows_for(p, "current", v_cur[None] * s, flags[pid], fold)
        print(f"grid fold {fold} done, {time.time() - start:.0f} s", flush=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(out / "grid.csv", index=False)

    chosen = {}
    for m, g in grid.groupby("method"):
        for fold in range(5):
            chosen[(m, fold)] = fit_s(g[g.fold != fold])
        chosen[(m, "all")] = fit_s(g)
    pd.DataFrame([dict(method=m, fold=f, s=s) for (m, f), s in chosen.items()]).to_csv(out / "scales.csv", index=False)

    rng, edit_rows = np.random.default_rng(0), []
    for fold in range(5):
        current, sigma = load_current(Path(args.runs) / f"current_f{fold}" / "last.pt", args.device)
        for d in load(args.cache, "val", svf=False, features=args.features, fold=fold)[:args.limit or None]:
            pid = str(d["pid"])
            with torch.no_grad():
                p = Patient(d, numbers[pid] * chosen[("rig", fold)], args.device)
                ed, tau, cond = es_inputs(d, args.device)
                v_cur = sample(current, "regress", ed, tau, sigma, 1, **cond).float() * chosen[("current", fold)]
                edit_rows += report_edits(p, flags[pid], truth_by[pid], v_cur, rng)[0]
        print(f"edits fold {fold} done, {time.time() - start:.0f} s", flush=True)
    edits = pd.DataFrame(edit_rows)
    edits.to_csv(out / "report_edits.csv", index=False)
    summarise(grid, chosen, edits)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", default="outputs/cv/rig_predictions.npz")
    p.add_argument("--truth", default="outputs/rig/segments.csv")
    p.add_argument("--runs", default="checkpoints/cv")
    p.add_argument("--cache", default="data/processed/acdc")
    p.add_argument("--features", default="data/processed/acdc_cinema")
    p.add_argument("--out", default="outputs/recalibration")
    p.add_argument("--limit", type=int, default=0, help="patients per fold, for a quick check")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(p.parse_args())
