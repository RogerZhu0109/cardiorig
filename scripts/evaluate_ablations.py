"""
Ablations of the rig model on the 5 cross-validation folds (pre-registered in EXPERIMENTS.md,
"Ablations of the rig model"). Every held-out patient is scored by the 3 members of each variant
trained on the fold that held it out. Writes ablations.csv (one row per patient, variant and
member) and prints each variant against the full model with a two-level bootstrap.
Usage: python scripts/evaluate_ablations.py --runs /content/runs/ablations --full /content/runs/cv --features /content/cache/cinema
       python scripts/evaluate_ablations.py --report outputs/ablations/ablations.csv   (re-print from the CSV)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_rig import VARIANTS, ablate, build, collate, stack_inputs  # noqa: E402
from src.data.cache import load  # noqa: E402
from src.evaluation import score  # noqa: E402
from src.rig import decode, patient_fields  # noqa: E402
from src.utils.metrics import STRUCTURES  # noqa: E402


def checkpoint(args, variant: str, member: int, fold: int) -> Path:
    """The full model's member 0 is the existing cross-validation model."""
    if variant == "full" and member == 0:
        return Path(args.full) / f"rig_f{fold}" / "last.pt"
    return Path(args.runs) / f"rig_{variant}_m{member}_f{fold}" / "last.pt"


def load_model(path: Path, variant: str, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    assert ck["iter"] == ck["args"]["iters"], f"{path} stopped at iteration {ck['iter']}"
    assert ck["args"].get("variant", "full") == variant, path
    model = build(variant).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, ck["mean"], ck["std"]


@torch.no_grad()
def evaluate(args) -> pd.DataFrame:
    rows = []
    for fold in args.folds:
        patients = load(args.cache, "val", features=args.features, fold=fold)
        prepared = []
        for d in patients:
            fields, slots, _ = patient_fields(d["ed_mask"])
            item = {k: torch.from_numpy(v) for k, v in stack_inputs(d, fields, slots).items()}
            prepared.append((d, fields, slots, collate([item], args.device)))
        for variant in args.variants:
            for member in args.members:
                model, mean, std = load_model(checkpoint(args, variant, member, fold), variant, args.device)
                for d, fields, slots, (x, maps, owner, feat, cls) in prepared:
                    v_maps, v_feat, v_cls = ablate(variant, maps, feat, cls)
                    m, _ = model(x, v_maps, owner, v_feat, v_cls)
                    numbers = (m[0] * std + mean).cpu().numpy()
                    v = torch.from_numpy(decode(numbers, float(d["tau"][int(d["es"])]), fields, slots)).float()
                    rows.append(dict(pid=str(d["pid"]), group=str(d["group"]), fold=fold, method=variant, seed=member,
                                     **score(d, v.to(args.device))))
            print(f"fold {fold} {variant} done", flush=True)
    return pd.DataFrame(rows)


def matrix(df: pd.DataFrame, method: str) -> np.ndarray:
    """Dice per patient (rows, sorted by pid) and member (columns) of one variant."""
    return df[df.method == method].pivot(index="pid", columns="seed", values="dice").sort_index().to_numpy()


def against_full(df: pd.DataFrame, variant: str, n_boot: int = 10000) -> str:
    """Δ = variant − full, 95% CI from resampling patients (shared) and each method's members (independently)."""
    A, B = matrix(df, variant), matrix(df, "full")
    rng, n = np.random.default_rng(0), len(A)
    boots = np.empty(n_boot)
    for r in range(n_boot):
        p = rng.integers(n, size=n)
        boots[r] = A[p][:, rng.integers(A.shape[1], size=A.shape[1])].mean() - B[p][:, rng.integers(B.shape[1], size=B.shape[1])].mean()
    delta = A.mean() - B.mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    verdict = "CONTRIBUTES" if hi < 0 and delta <= -0.005 else "BETTER WITHOUT" if lo > 0 and delta >= 0.005 \
        else "NO MEASURABLE EFFECT"
    return f"{variant} - full: {delta:+.4f} [{lo:+.4f}, {hi:+.4f}], n={n} patients x {A.shape[1]}/{B.shape[1]} members -> {verdict}"


def report(df: pd.DataFrame) -> None:
    cols = [f"dice_{k}" for k in STRUCTURES.values()]
    df = df.assign(dice=df[cols].mean(axis=1))
    print(df.groupby("method", sort=False)[["dice", *cols, "folded_pct"]].mean().round(4).to_string())
    per_member = df.groupby(["method", "seed"], sort=False).dice.mean().unstack()
    print("mean Dice per member:\n" + per_member.round(4).to_string())
    print("sd across members:", per_member.std(axis=1).round(4).to_dict())
    print(df.pivot_table(index="group", columns="method", values="dice", sort=False).round(3).to_string())
    pooled = df.groupby("method", sort=False)[["ed_lv", "gt_lv", "pred_lv"]].sum()
    print("LV contraction:", (1 - pooled.pred_lv / pooled.ed_lv).round(3).to_dict(),
          "true", round(1 - pooled.gt_lv.iloc[0] / pooled.ed_lv.iloc[0], 3))
    for variant in df.method.unique():
        if variant != "full":
            print(against_full(df, variant))


def main(args):
    if args.report:
        return report(pd.read_csv(args.report))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = evaluate(args)
    df.to_csv(out / "ablations.csv", index=False)
    report(df)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", help="folder with rig_{variant}_m{member}_f{fold}")
    p.add_argument("--full", help="folder with the cross-validation models rig_f{fold}")
    p.add_argument("--cache", default="data/processed/acdc")
    p.add_argument("--features")
    p.add_argument("--out", default="outputs/ablations")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--members", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--report", help="re-print the comparisons from an existing ablations.csv")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(p.parse_args())
