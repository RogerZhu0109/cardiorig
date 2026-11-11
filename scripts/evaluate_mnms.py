"""
One-time zero-shot evaluation of the final ACDC-trained models on the 345 M&Ms subjects
(pre-registered in EXPERIMENTS.md, "M&Ms external test"). Scoring is scripts/evaluate_test.py
unchanged; this adds the ACDC pathology label PhaseFlow needs, the rig-anatomy check before any
model runs, and the per-vendor report. Writes mnms.csv (one row per subject, method and seed).
Usage: python scripts/evaluate_mnms.py --runs /content/runs/final --cache /content/cache/mnms --features /content/cache/mnms/cinema
       python scripts/evaluate_mnms.py --report outputs/mnms/mnms.csv   (re-print from the CSV)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import evaluate_test  # noqa: E402
from scripts.evaluate_test import two_level  # noqa: E402
from src.data.cache import load  # noqa: E402
from src.rig import fit_compact  # noqa: E402
from src.utils.metrics import STRUCTURES  # noqa: E402

# PhaseFlow is conditioned on an ACDC pathology class; M&Ms classes outside ACDC take the nearest one
ACDC_GROUP = dict(DCM="DCM", HCM="HCM", NOR="NOR", HHD="HCM", ARV="RV", IHD="MINF", AHS="NOR", LVNC="DCM", Other="NOR")
MODELS = ("current", "rig", "phaseflow_tau0", "krebs_mean")
COMPARISONS = (("rig", "current"), ("rig", "phaseflow_tau0"), ("rig", "krebs_mean"))


def evaluate(args) -> pd.DataFrame:
    patients, excluded = [], []
    for d in load(args.cache, "testing", features=args.features):
        try:
            fit_compact(d)  # builds the rig anatomy from the ED mask; no model involved
        except Exception as e:  # noqa: BLE001  any failure excludes the subject for every method
            excluded.append(f"{d['pid']} ({type(e).__name__}: {e})")
            continue
        d["group"] = ACDC_GROUP[str(d["group"])]
        patients.append(d)
    print(f"{len(patients)} subjects, excluded {len(excluded)}: {'; '.join(excluded)}", flush=True)
    if len(excluded) > 5:
        sys.exit("more than 5 subjects without a rig anatomy")
    df = evaluate_test.evaluate(args, patients).rename(columns={"group": "phaseflow_group"})
    info = pd.read_csv(Path(args.cache) / "qc.csv")[["pid", "vendor", "centre", "group", "split"]]
    return df.merge(info.rename(columns={"group": "pathology"}), on="pid")


def report(df: pd.DataFrame) -> None:
    df = df.assign(dice=df[[f"dice_{k}" for k in STRUCTURES.values()]].mean(axis=1), group=df.pathology)
    print(f"== all {df.pid.nunique()} subjects (the report of the ACDC test-set run, by M&Ms pathology)")
    evaluate_test.report(df)
    df = df[df.method != "segmentation"]
    for method in ("rig", "current"):
        print("TRANSFER  " + two_level(df, method, "identity"))
    exact = df[df.pathology.isin(["DCM", "HCM", "NOR"])]  # PhaseFlow gets its true class only here: its comparison is judged on these
    print(f"SECONDARY pathology is an ACDC class ({exact.pid.nunique()} subjects): " + two_level(exact, "rig", "phaseflow_tau0"))

    mean = df.groupby(["vendor", "method"], sort=False).dice.mean().unstack()
    mean.loc["all"] = df.groupby("method", sort=False).dice.mean()
    mean["n"] = df[df.method == "identity"].groupby("vendor").size().reindex(mean.index).fillna(df.pid.nunique()).astype(int)
    order = ["identity", *MODELS, "phaseflow_tau1", "ceiling", "teacher"]
    print("== ES Dice by vendor (A Siemens, B Philips, C GE, D Canon)\n" + mean[order + ["n"]].round(3).to_string())
    closed = mean[[*MODELS, "ceiling"]].sub(mean.identity, axis=0).div(mean.teacher - mean.identity, axis=0)
    print("share of the no-motion -> teacher gap closed\n" + closed.round(3).to_string())
    for vendor, g in df.groupby("vendor"):
        for a, b in COMPARISONS:
            g_ab = g[g.pathology.isin(exact.pathology.unique())] if b == "phaseflow_tau0" else g
            print(f"vendor {vendor}  " + two_level(g_ab, a, b, primary=b == "current"))
    mean_exact = exact.groupby(["vendor", "method"], sort=False).dice.mean().unstack()
    diff = pd.DataFrame({f"{a} - {b}": (mean_exact if b == "phaseflow_tau0" else mean.drop(index="all")).pipe(lambda m: m[a] - m[b])
                         for a, b in COMPARISONS})
    print("consistent across vendors: " + ", ".join(
        f"{c}: {'YES' if ((diff[c] > -0.02).all() if c.endswith('current') else (diff[c] > 0).all() or (diff[c] < 0).all()) else 'NO'}"
        f" ({' '.join(f'{x:+.3f}' for x in diff[c])})" for c in diff))


def main(args):
    if args.report:
        return report(pd.read_csv(args.report))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = evaluate(args)
    df.to_csv(out / "mnms.csv", index=False)
    report(df)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", help="folder with {rig,current,phaseflow,krebs}_s{seed}")
    p.add_argument("--cache", default="data/processed/mnms")
    p.add_argument("--features")
    p.add_argument("--out", default="outputs/mnms")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--report", help="re-print the comparisons from an existing mnms.csv")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(p.parse_args())
