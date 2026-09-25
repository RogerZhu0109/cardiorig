"""
Train the SVF generator on teacher targets: flow matching (fm) or the regression baseline.
Only ACDC training patients are read: 90 train, 10 val (src/data/cache.py VAL_PATIENTS) for
monitoring. The test split is never read. The saved model is the EMA after a fixed iteration
budget; there is no checkpoint selection.
Usage: python scripts/train.py --objective fm --out checkpoints/fm_s0
"""
import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.cache import Pairs, load  # noqa: E402
from src.evaluation import es_inputs, score  # noqa: E402
from src.models.flow import loss, sample  # noqa: E402
from src.models.unet import UNet  # noqa: E402
from src.utils.metrics import STRUCTURES  # noqa: E402


def validate(model, objective, sigma, patients, device, nfe):
    g = torch.Generator().manual_seed(0)
    rows = []
    for d in patients:
        ed, tau = es_inputs(d, device)
        rows.append(score(d, sample(model, objective, ed, tau, sigma, nfe, g)))
    return dict(val_dice=float(np.nanmean([[r[f"dice_{k}"] for k in STRUCTURES.values()] for r in rows])),
                val_folded_pct=float(np.mean([r["folded_pct"] for r in rows])))


def main(args):
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_pts = load(args.cache, "train")
    val = load(args.cache, "val", svf=False)
    train = Pairs(train_pts, args.device)
    del train_pts
    sigma = train.rms()
    print(f"{len(train)} pairs from train, {len(val)} val patients, sigma {sigma:.3f} px, device {args.device}", flush=True)

    model = UNet().to(args.device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * i / args.iters)))
    gen = torch.Generator().manual_seed(args.seed)
    amp = args.device == "cuda"

    t0, run = time.time(), 0.0
    for it in range(1, args.iters + 1):
        ed, tau, v = train.sample(args.batch, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            l = loss(model, args.objective, ed, tau, v / sigma)
        opt.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        decay = min(args.ema, (1 + it) / (10 + it))
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), model.parameters()):
                pe.lerp_(p, 1 - decay)
        run += l.item()

        if it % args.log_every == 0:
            print(f"it {it} loss {run / args.log_every:.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{it / (time.time() - t0):.1f} it/s", flush=True)
            run = 0.0
        if it % args.val_every == 0 or it == args.iters:
            log = dict(iter=it, **validate(ema, args.objective, sigma, val, args.device, args.val_nfe))
            print(json.dumps(log), flush=True)
            with open(out / "log.jsonl", "a") as f:
                f.write(json.dumps(log) + "\n")
            torch.save(dict(model=ema.state_dict(), sigma=sigma, args=vars(args), iter=it), out / "last.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data/processed/acdc")
    p.add_argument("--out", required=True)
    p.add_argument("--objective", choices=["fm", "regress"], default="fm")
    p.add_argument("--iters", type=int, default=60000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_nfe", type=int, default=4)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--val_every", type=int, default=5000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available() else "cpu")
    main(p.parse_args())
