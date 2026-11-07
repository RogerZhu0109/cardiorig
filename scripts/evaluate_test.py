"""
Final, one-time evaluation on the 50 ACDC test patients (pre-registered in EXPERIMENTS.md,
"Final test-set run"). Models: {rig,current,phaseflow,krebs}_s{seed}, trained on all 100 training
patients. Inference variants are the ones fixed by cross-validation; ES labels only score.
Writes test.csv (one row per patient, method and seed) and prints the pre-registered comparisons
with a two-level bootstrap (patients and seeds).
Usage: python scripts/evaluate_test.py --runs /content/runs/final --features /content/cache/cinema
       python scripts/evaluate_test.py --report outputs/test/test.csv   (re-print from the CSV)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_cv import load_current, load_rig, predict_rig  # noqa: E402
from scripts.evaluate_published import (curve_metrics, es_krebs, frames_of, krebs_fields, load_phaseflow,  # noqa: E402
                                        phaseflow_fields, predicted_area)
from scripts.train_phaseflow import lv_probability  # noqa: E402
from src.data.cache import load, ordered, resampled  # noqa: E402
from src.models.flow import sample  # noqa: E402
from src.models.krebs import Krebs  # noqa: E402
from src.models.phaseflow import T  # noqa: E402
from src.evaluation import score  # noqa: E402
from src.rig import decode, fit_compact, patient_fields  # noqa: E402
from src.utils.metrics import STRUCTURES, dice  # noqa: E402

REFERENCES = ("identity", "teacher", "ceiling")  # no training: one value per patient


@torch.no_grad()
def evaluate(args) -> pd.DataFrame:
    device, rows, runs = args.device, [], Path(args.runs)
    patients = load(args.cache, "testing", features=args.features)
    for seed in args.seeds:
        pf = load_phaseflow(runs / f"phaseflow_s{seed}", device)
        kr = Krebs().to(device).eval()
        kr.load_state_dict(torch.load(runs / f"krebs_s{seed}" / "last.pt", map_location=device, weights_only=False)["model"])
        current, sigma = load_current(runs / f"current_s{seed}" / "last.pt", device)
        rig, mean, std = load_rig(runs / f"rig_s{seed}" / "last.pt", device)
        gen = torch.Generator().manual_seed(seed)
        for d in patients:
            info = dict(pid=str(d["pid"]), group=str(d["group"]))
            es, pos = frames_of(d)
            cine = resampled(ordered(d["cine"], d["tau"]).astype(np.float32), T)
            truth = np.array([(lv_probability(pf["seg"], cine[t], device) >= 0.5).sum() for t in range(T)], float)
            taus = pos / len(d["tau"])
            tau_es = float(d["tau"][int(d["es"])])
            fields, slots, _ = patient_fields(d["ed_mask"])
            numbers, _ = predict_rig(rig, mean, std, d, device, fields, slots)
            ed = torch.from_numpy(d["cine"][int(d["ed"])][:, None].astype(np.float32)).to(device)
            feat = torch.from_numpy(d["feat"].astype(np.float32)).to(device)
            cls = torch.from_numpy(d["cls"].astype(np.float32)).to(device)[None].expand(len(ed), -1)
            at = lambda t: torch.full((len(ed),), float(t), device=device)  # noqa: E731
            curves = dict(current=torch.stack([sample(current, "regress", ed, at(t), sigma, 1, feat=feat, cls=cls) for t in taus]),
                          rig=torch.from_numpy(np.stack([decode(numbers, float(t), fields, slots) for t in taus])).float().to(device),
                          phaseflow_tau0=phaseflow_fields(pf, d, 0.0, device, gen),
                          phaseflow_tau1=phaseflow_fields(pf, d, 1.0, device, gen),
                          krebs_mean=krebs_fields(kr, d, None, device))
            es_fields = {k: v[es] for k, v in curves.items()}
            es_fields["current"] = sample(current, "regress", ed, at(tau_es), sigma, 1, feat=feat, cls=cls)
            es_fields["rig"] = torch.from_numpy(decode(numbers, tau_es, fields, slots)).float().to(device)
            es_fields["krebs_mean"] = es_krebs(torch.cat([kr.simulate(c) for c in ed.split(8)]), d)
            for name, v in es_fields.items():
                rows.append(dict(**info, seed=seed, method=name, **score(d, v.float()),
                                 **curve_metrics(predicted_area(curves[name], d), truth)))
            es_cine = torch.from_numpy(d["cine"][int(d["es"])][:, None].astype(np.float32)).to(device)
            found = dice(pf["seg"](es_cine).argmax(1).cpu().numpy(), d["es_mask"])
            rows.append(dict(**info, seed=seed, method="segmentation", **{f"dice_{k}": x for k, x in found.items()}))
            if seed == args.seeds[0]:
                teacher = torch.from_numpy(d["svf"][int(d["es"])].astype(np.float32)).to(device)
                refs = dict(identity=torch.zeros_like(teacher), teacher=teacher,
                            ceiling=torch.from_numpy(decode(fit_compact(d), tau_es, fields, slots)).float().to(device))
                for name, v in refs.items():
                    rows.append(dict(**info, seed=-1, method=name, **score(d, v)))
        print(f"seed {seed} done", flush=True)
    return pd.DataFrame(rows)


def matrix(df: pd.DataFrame, method: str) -> np.ndarray:
    """Dice per patient (rows, sorted by pid) and seed (columns) of one method."""
    return df[df.method == method].pivot(index="pid", columns="seed", values="dice").sort_index().to_numpy()


def two_level(df: pd.DataFrame, a: str, b: str, primary: bool = False, n_boot: int = 10000) -> str:
    """Δ = a − b with a 95% CI from resampling patients (shared) and each method's seeds (independently)."""
    A, B = matrix(df, a), matrix(df, b)
    rng, n = np.random.default_rng(0), len(A)
    boots = np.empty(n_boot)
    for r in range(n_boot):
        p = rng.integers(n, size=n)
        boots[r] = A[p][:, rng.integers(A.shape[1], size=A.shape[1])].mean() - B[p][:, rng.integers(B.shape[1], size=B.shape[1])].mean()
    diff = A.mean(1) - B.mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    if primary:
        verdict = "MATCHES" if lo > -0.01 else "SMALL COST" if diff.mean() > -0.02 else "WORSE"
    else:
        verdict = "BEATS" if lo > 0 else "MATCHES" if lo > -0.01 else "BEHIND"
    return (f"{a} - {b}: {diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], Wilcoxon p={wilcoxon(diff).pvalue:.3g}, "
            f"n={n} patients x {A.shape[1]}/{B.shape[1]} seeds -> {verdict}")


def report(df: pd.DataFrame) -> None:
    cols = [f"dice_{k}" for k in STRUCTURES.values()]
    df = df.assign(dice=df[cols].mean(axis=1))
    print("segmentation network on test ES (mean over seeds):", df[df.method == "segmentation"][cols].mean().round(4).to_dict())
    df = df[df.method != "segmentation"]
    metrics = ["dice", *cols, "folded_pct", "ef_err", "vol_corr", "vol_r2", "vol_mae"]
    print(df.groupby("method", sort=False)[metrics].mean().round(4).to_string())
    per_seed = df[df.seed >= 0].groupby(["method", "seed"]).dice.mean().unstack()
    print("mean Dice per seed:\n" + per_seed.round(4).to_string())
    print("sd across seeds:", per_seed.std(axis=1).round(4).to_dict())
    print(df.pivot_table(index="group", columns="method", values="dice", sort=False).round(3).to_string())
    pooled = df.groupby("method", sort=False)[["ed_lv", "gt_lv", "pred_lv"]].sum()
    print("LV contraction:", (1 - pooled.pred_lv / pooled.ed_lv).round(3).to_dict(),
          "true", round(1 - pooled.gt_lv.iloc[0] / pooled.ed_lv.iloc[0], 3))
    print("PRIMARY   " + two_level(df, "rig", "current", primary=True))
    for method in ("phaseflow_tau0", "krebs_mean"):
        print("SECONDARY " + two_level(df, "rig", method))
    print("reported  " + two_level(df, "ceiling", "rig") + "  (ED-information gap)")
    print("reported  " + two_level(df, "rig", "phaseflow_tau1"))


def main(args):
    if args.report:
        return report(pd.read_csv(args.report))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = evaluate(args)
    df.to_csv(out / "test.csv", index=False)
    report(df)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", help="folder with {rig,current,phaseflow,krebs}_s{seed}")
    p.add_argument("--cache", default="data/processed/acdc")
    p.add_argument("--features")
    p.add_argument("--out", default="outputs/test")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--report", help="re-print the comparisons from an existing test.csv")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(p.parse_args())
