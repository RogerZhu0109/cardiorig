"""
ED -> ES evaluation. Inputs per patient: ED slices, ED labels, tau_ES; ES labels only score.
Default split is val (train = generalisation-gap diagnostic). Run --split testing once, for the final numbers, after every choice is frozen.
Per-patient rows go to CSV: method, nfe, sample (k = single sample k; "mean" = log-Euclidean
mean, i.e. the average of the K SVFs). References: identity (no motion) and teacher (registered
to the true ES frame, so it sees the target: an upper bound, not a competitor).
Usage: python scripts/evaluate.py --checkpoints checkpoints/fm_s0/last.pt checkpoints/reg_s0/last.pt
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.cache import load  # noqa: E402
from src.evaluation import es_inputs, score  # noqa: E402
from src.models.flow import sample  # noqa: E402
from src.models.unet import UNet  # noqa: E402
from src.utils.metrics import STRUCTURES  # noqa: E402
from src.utils.svf import integrate, warp  # noqa: E402


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +- std over patients; single samples are first averaged over k within each patient."""
    df = df.assign(kind=np.where(df["sample"] == "mean", "mean-of-K", "single"))
    cols = [f"dice_{k}" for k in STRUCTURES.values()]
    df["dice_mean"] = df[cols].mean(axis=1)
    per_pt = df.groupby(["method", "nfe", "kind", "pid"])[cols + ["dice_mean", "folded_pct", "ms_per_slice",
                                                                   "ed_lv", "gt_lv", "pred_lv"]].mean()
    out = []
    for key, g in per_pt.groupby(level=[0, 1, 2]):
        r = dict(zip(["method", "nfe", "kind"], key), n=len(g))
        for c in cols + ["dice_mean"]:
            r[c] = f"{g[c].mean():.3f}±{g[c].std():.3f}"
        r["lv_captured"] = round((1 - g.pred_lv.sum() / g.ed_lv.sum()) / (1 - g.gt_lv.sum() / g.ed_lv.sum()), 3)
        r["folded_pct"] = f"{g.folded_pct.mean():.4f}"
        r["ms_per_slice"] = round(g.ms_per_slice.mean(), 2)
        out.append(r)
    return pd.DataFrame(out)


def main(args):
    patients = load(args.cache, args.split)
    rows = []
    for d in patients:
        base = dict(pid=str(d["pid"]), group=str(d["group"]), ms_per_slice=np.nan)
        z = torch.zeros(len(d["z"]), 2, *d["cine"].shape[-2:], device=args.device)
        rows.append(dict(base, method="identity", nfe=0, sample="-", **score(d, z)))
        v = torch.from_numpy(d["svf"][d["es"]].astype(np.float32)).to(args.device)
        rows.append(dict(base, method="teacher", nfe=0, sample="-", **score(d, v)))

    for path in args.checkpoints:
        ck = torch.load(path, map_location=args.device)
        model = UNet().to(args.device).eval()
        model.load_state_dict(ck["model"])
        obj, sigma, name = ck["args"]["objective"], ck["sigma"], Path(path).parent.name
        n_samples = 1 if obj == "regress" else args.samples
        for nfe in ([1] if obj == "regress" else args.nfe):
            for d in patients:
                ed, tau = es_inputs(d, args.device)
                sample(model, obj, ed, tau, sigma, nfe)  # warm-up for timing
                vs = []
                for k in range(n_samples):
                    g = torch.Generator().manual_seed(1000 * k + int(str(d["pid"])[-3:]))
                    sync(args.device)
                    t0 = time.perf_counter()
                    v = sample(model, obj, ed, tau, sigma, nfe, g)
                    warp(ed, integrate(v))
                    sync(args.device)
                    ms = 1000 * (time.perf_counter() - t0) / len(ed)
                    rows.append(dict(pid=str(d["pid"]), group=str(d["group"]), method=name, nfe=nfe, sample=str(k),
                                     ms_per_slice=ms, **score(d, v)))
                    vs.append(v)
                if n_samples > 1:
                    rows.append(dict(pid=str(d["pid"]), group=str(d["group"]), method=name, nfe=nfe, sample="mean",
                                     ms_per_slice=np.nan, **score(d, torch.stack(vs).mean(0))))
            print(f"{name} nfe {nfe} done", flush=True)

    df = pd.DataFrame(rows)
    out = Path(args.out or f"outputs/eval/{args.split}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summarise(df).to_string(index=False))
    print(f"per-patient rows -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="*", default=[])
    p.add_argument("--cache", default="data/processed/acdc")
    p.add_argument("--split", choices=["train", "val", "testing"], default="val")
    p.add_argument("--nfe", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available() else "cpu")
    main(p.parse_args())
