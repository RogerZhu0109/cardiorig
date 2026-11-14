"""
Cross-validation comparison of the rig model and the dense-field model ("current") on held-out
patients. Every patient is scored once, by the models of the fold that held it out. ES labels only
score. Writes cv.csv (one row per patient and method) and rig_predictions.npz (held-out rig parameters).
Usage: python scripts/evaluate_cv.py --runs checkpoints/cv --features data/processed/acdc_cinema
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_rig import collate, stack_inputs  # noqa: E402
from src.data.cache import load  # noqa: E402
from src.evaluation import es_inputs, score  # noqa: E402
from src.models.rig_net import RigNet  # noqa: E402
from src.models.unet import UNet, predict  # noqa: E402
from src.rig import decode, fit_compact, patient_fields  # noqa: E402
from src.utils.metrics import STRUCTURES  # noqa: E402


def load_current(path: Path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = UNet(cond=True).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, ck["sigma"]


def load_rig(path: Path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = RigNet().to(device).eval()
    model.load_state_dict(ck["model"])
    return model, ck["mean"], ck["std"]


@torch.no_grad()
def predict_rig(model, mean, std, d, device, fields, slots):
    item = {k: torch.from_numpy(v) for k, v in stack_inputs(d, fields, slots).items()}
    m, log_var = model(*collate([item], device))
    numbers = (m[0] * std + mean).cpu().numpy()
    return numbers, (log_var[0].exp().sqrt() * std).cpu().numpy()


def evaluate(args) -> tuple[pd.DataFrame, dict]:
    rows, saved = [], {}
    for fold in range(5):
        current, sigma = load_current(Path(args.runs) / f"current_f{fold}" / "last.pt", args.device)
        rig, mean, std = load_rig(Path(args.runs) / f"rig_f{fold}" / "last.pt", args.device)
        for d in load(args.cache, "val", features=args.features, fold=fold):
            fields, slots, _ = patient_fields(d["ed_mask"])
            tau_es = float(d["tau"][int(d["es"])])
            target = fit_compact(d)
            numbers, spread = predict_rig(rig, mean, std, d, args.device, fields, slots)
            with torch.no_grad():
                ed, tau, cond = es_inputs(d, args.device)
                v_current = predict(current, ed, tau, sigma, **cond)
            teacher = d["svf"][int(d["es"])].astype(np.float32)
            fields_by_method = dict(identity=torch.zeros_like(v_current), teacher=torch.from_numpy(teacher),
                                    ceiling=torch.from_numpy(decode(target, tau_es, fields, slots)), current=v_current,
                                    rig=torch.from_numpy(decode(numbers, tau_es, fields, slots)))
            for name, v in fields_by_method.items():
                rows.append(dict(pid=str(d["pid"]), group=str(d["group"]), fold=fold, method=name,
                                 **score(d, v.float().to(args.device))))
            saved[str(d["pid"])] = dict(numbers=numbers, spread=spread, target=target)
        print(f"fold {fold} done", flush=True)
    return pd.DataFrame(rows), saved


def paired(wide: pd.DataFrame, a: str, b: str, n_boot: int = 10000) -> str:
    diff = (wide[a] - wide[b]).to_numpy()
    rng = np.random.default_rng(0)
    boots = rng.choice(diff, (n_boot, len(diff))).mean(1)
    p = wilcoxon(diff).pvalue
    return f"{a} - {b}: {diff.mean():+.4f} [{np.percentile(boots, 2.5):+.4f}, {np.percentile(boots, 97.5):+.4f}], Wilcoxon p={p:.3g}, n={len(diff)}"


def report(df: pd.DataFrame) -> None:
    cols = [f"dice_{k}" for k in STRUCTURES.values()]
    df["dice"] = df[cols].mean(axis=1)
    print(df.groupby("method", sort=False)[["dice", *cols, "folded_pct"]].mean().round(4).to_string())
    print(df.pivot_table(index="group", columns="method", values="dice", sort=False).round(3).to_string())
    pooled = df.groupby("method", sort=False)[["ed_lv", "gt_lv", "pred_lv"]].sum()
    print("LV contraction:", (1 - pooled.pred_lv / pooled.ed_lv).round(3).to_dict(),
          "true", round(1 - pooled.gt_lv.iloc[0] / pooled.ed_lv.iloc[0], 3))
    print(paired(df.pivot(index="pid", columns="method", values="dice"), "rig", "current"))


def main(args):
    df, saved = evaluate(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "cv.csv", index=False)
    np.savez(out / "rig_predictions.npz", pids=np.array(sorted(saved)),
             **{k: np.stack([saved[p][k] for p in sorted(saved)]) for k in ("numbers", "spread", "target")})
    report(df)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True, help="folder with current_f{0-4} and rig_f{0-4}")
    p.add_argument("--cache", default="data/processed/acdc")
    p.add_argument("--features", required=True)
    p.add_argument("--out", default="outputs/cv")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(p.parse_args())
