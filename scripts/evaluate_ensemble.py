"""
Deep-ensemble uncertainty of the rig model on held-out patients: patient-level LV contraction
(primary) and segment thickening (secondary), both against real ES labels. ES labels only score.
Usage: python scripts/evaluate_ensemble.py --runs checkpoints/cv --members checkpoints/cv_ensemble --features data/processed/acdc_cinema
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_uncertainty import cross_fit, ratio_ci, scored, segment_thickening  # noqa: E402
from scripts.evaluate_cv import load_rig, predict_rig  # noqa: E402
from src.data.cache import load  # noqa: E402
from src.evaluation import score  # noqa: E402
from src.rig import decode, patient_fields, positions, time_basis  # noqa: E402
from src.utils.svf import integrate, warp  # noqa: E402

METHODS = ("rig", "fixed", "population")


@torch.no_grad()
def contraction(d: dict, motion: dict, device) -> torch.Tensor:
    """LV contraction per draw, from motion[z] (K, 2, H, W) for every slice."""
    mask = torch.from_numpy(d["ed_mask"]).long().to(device)
    onehot = F.one_hot(mask, 4).permute(0, 3, 1, 2).float()
    area = sum(((warp(onehot[z].expand(len(v), -1, -1, -1), integrate(v)).argmax(1) == 3).sum((1, 2))) for z, v in motion.items())
    return 1 - area.float() / (mask == 3).sum()


@torch.no_grad()
def patient_draws(d: dict, means: np.ndarray, spreads: np.ndarray, draws: int, gen: torch.Generator, device) -> dict:
    """Contraction and segment thickening under the ensemble's predictive distribution."""
    fields, slots, _ = patient_fields(d["ed_mask"])
    pick = torch.randint(len(means), (draws,), generator=gen)
    mean = torch.from_numpy(means).float()[pick].to(device)
    spread = torch.from_numpy(spreads).float()[pick].to(device)
    sample = mean + spread * torch.randn(mean.shape, generator=gen).to(device)
    phi = torch.from_numpy(time_basis(np.array([float(d["tau"][int(d["es"])])]), 2)[0]).float().to(device)
    c = sample @ phi
    motion = {z: torch.einsum("kp,pchw->kchw", c[:, slots[z]], torch.from_numpy(fields[z]).float().to(device))
              for z in range(len(fields))}
    lv = contraction(d, motion, device)
    seg, seen = segment_thickening({z: motion[z] for z in positions(d["ed_mask"])}, d["ed_mask"], device)
    keep = np.nonzero(seen)[0]
    segments = pd.DataFrame(dict(seg=keep, mu=seg[:, keep].mean(0).cpu().numpy(), sd=seg[:, keep].std(0).cpu().numpy()))
    return dict(mu=lv.mean().item(), sd=lv.std().item(), segments=segments)


def collect(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    gen, patients, segments = torch.Generator().manual_seed(0), [], []
    truth = pd.read_csv(args.truth).dropna(subset=["dt_true"])[["pid", "seg", "dt_true"]].rename(columns=dict(dt_true="y"))
    for fold in range(5):
        paths = [Path(args.runs) / f"rig_f{fold}" / "last.pt", *sorted(Path(args.members).glob(f"rig_m*_f{fold}/last.pt"))]
        models = [load_rig(p, args.device) for p in paths]
        for d in load(args.cache, "val", svf=False, features=args.features, fold=fold):
            fields, slots, _ = patient_fields(d["ed_mask"])
            out = [predict_rig(*m, d, args.device, fields, slots) for m in models]
            means, spreads = np.stack([o[0] for o in out]), np.stack([o[1] for o in out])
            result = patient_draws(d, means, spreads, args.draws, gen, args.device)
            tau = float(d["tau"][int(d["es"])])
            metrics = score(d, torch.from_numpy(decode(means.mean(0), tau, fields, slots)).float().to(args.device))
            pid, group = str(d["pid"]), str(d["group"])
            patients.append(dict(pid=pid, group=group, fold=fold, seg=0, members=len(models), mu=result["mu"], sd=result["sd"],
                                 y=1 - (d["es_mask"] == 3).sum() / (d["ed_mask"] == 3).sum(), **metrics))
            segments.append(result["segments"].assign(pid=pid, group=group, fold=fold))
        print(f"fold {fold} done, {len(models)} members", flush=True)
    return pd.DataFrame(patients), pd.concat(segments, ignore_index=True).merge(truth, on=["pid", "seg"])


def report(name: str, df: pd.DataFrame) -> None:
    fit = cross_fit(df, 0.9)
    results = {m: scored(fit, m) for m in METHODS}
    print(f"\n{name}: {df.pid.nunique()} patients, {len(df)} rows, median sd {df.sd.median():.4f}, median |error| {(df.y - df.mu).abs().median():.4f}")
    print(pd.DataFrame({k: dict(coverage=v.inside.mean(), width=v.width.mean(), score=v.score.mean())
                        for k, v in results.items()}).T.round(4).to_string())
    print(pd.DataFrame({k: v.groupby("group").inside.mean() for k, v in results.items()}).round(3).to_string())
    for other in METHODS[1:][::-1]:
        print("score ratio rig / {}: {:.3f} [{:.3f}, {:.3f}]".format(other, *ratio_ci(results["rig"], results[other])))
    print(f"spread vs error Spearman r: {fit.sd.corr(fit.error, method='spearman'):.3f}")


def main(args):
    patients, segments = collect(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    patients.to_csv(out / "patients.csv", index=False)
    segments.to_csv(out / "segments.csv", index=False)
    dice = patients[["dice_RV", "dice_MYO", "dice_LV"]].mean(axis=1)
    print(f"ensemble ES Dice {dice.mean():.4f}, members per fold {sorted(patients.members.unique())}")
    report("Primary, LV contraction", patients)
    report("Secondary, segment thickening", segments)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True, help="folder with rig_f{0-4}")
    p.add_argument("--members", required=True, help="folder with rig_m*_f{0-4}")
    p.add_argument("--truth", default="outputs/rig/segments.csv")
    p.add_argument("--cache", default="data/processed/acdc")
    p.add_argument("--features", required=True)
    p.add_argument("--out", default="outputs/ensemble")
    p.add_argument("--draws", type=int, default=200)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(p.parse_args())
