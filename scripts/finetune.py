"""
Image-consistency fine-tuning of a trained regression generator on real frames.

Starts from the EMA weights of a train.py regression checkpoint (--init) and minimises, over the same
training pairs,

    -LNCC(ED o exp(v/2), I_tau o exp(-v/2)) + lam * ||grad v||^2  +  alpha * ||w - v_teacher / sigma||^2

(src/models/consistency.py; the last term keeps the teacher as a prior, alpha = 0 drops it). I_tau is
the real frame at tau, used only as a training target: the generator still sees only the ED slice,
tau and the ED-stack CineMA features. Geometric augmentation moves ED, target and teacher SVF together;
intensity augmentation acts on the network input only, the loss compares un-augmented intensities.
The init checkpoint is only read; the output is an ordinary regression checkpoint for evaluate.py.
Resumable like train.py. Usage:
    python scripts/finetune.py --init checkpoints/phase1/reg_enc_aug_s0/last.pt --features <cinema root> \
        --augment --out checkpoints/ncc/reg_enc_aug_ncc_s0
"""
import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import validate  # noqa: E402
from src.data.augment import apply_similarity, intensity, random_similarity  # noqa: E402
from src.data.cache import Pairs, load  # noqa: E402
from src.models.consistency import consistency  # noqa: E402
from src.models.unet import UNet  # noqa: E402


def main(args):
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    init = torch.load(args.init, map_location=args.device, weights_only=False)
    assert init["args"]["objective"] == "regress", "fine-tuning starts from a regression checkpoint"
    assert bool(init["args"].get("features")) == bool(args.features), "--features must match the init checkpoint"
    sigma = init["sigma"]  # keep the init's normalisation, so w means the same thing

    train_pts = load(args.cache, "train", features=args.features)
    val = load(args.cache, "val", svf=False, features=args.features)
    train = Pairs(train_pts, args.device, targets=True)
    del train_pts
    print(f"{len(train)} pairs from train, {len(val)} val patients, sigma {sigma:.3f} px (init), "
          f"device {args.device}", flush=True)

    model = UNet(cond=bool(args.features)).to(args.device)
    model.load_state_dict(init["model"])
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * i / args.iters)))
    gen = torch.Generator().manual_seed(args.seed)
    amp = args.device == "cuda"

    start = 1
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=args.device, weights_only=False)
        model.load_state_dict(ck["train_model"]), ema.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"]), sched.load_state_dict(ck["sched"])
        gen.set_state(ck["gen"].cpu()), torch.set_rng_state(ck["rng"].cpu())
        if amp:
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu())
        start = ck["iter"] + 1
        print(f"resumed from iter {ck['iter']}", flush=True)
    else:
        log = dict(iter=0, **validate(ema, "regress", sigma, val, args.device, 1))
        print(json.dumps(log), flush=True)
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(log) + "\n")

    t0, run = time.time(), torch.zeros(4, device=args.device)
    for it in range(start, args.iters + 1):
        ed, tau, v, cond, target = train.sample(args.batch, gen, target=True)
        if args.augment:  # ED and target share one geometric transform; channels are sampled alike
            A, b = random_similarity(len(ed), ed.shape[-1], 30.0, 0.1, 8.0, gen)
            imgs, v, *feat = apply_similarity(torch.cat([ed, target], 1), v, A, b, cond.get("feat"))
            ed, target = imgs[:, :1], imgs[:, 1:]
            if feat:
                cond["feat"] = feat[0]
            ed_in = intensity(ed, gen)
        else:
            ed_in = ed
        zero = torch.zeros(len(ed), device=ed.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            w = model(torch.zeros_like(v), zero, ed_in, tau, **cond)
        w = w.float()  # LNCC variances need fp32
        sim, reg = consistency(sigma * w, ed, target, args.win)
        anchor = F.mse_loss(w, v / sigma)
        l = (-sim + args.lam * reg).mean() + args.alpha * anchor
        opt.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        decay = min(args.ema, (1 + it) / (10 + it))
        with torch.no_grad():
            torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - decay)
        run += torch.stack([l, sim.mean(), reg.mean(), anchor]).detach()

        if it % args.log_every == 0:
            m = (run / args.log_every).tolist()
            print(f"it {it} loss {m[0]:.4f} sim {m[1]:.4f} reg {m[2]:.4f} anchor {m[3]:.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} {(it - start + 1) / (time.time() - t0):.1f} it/s", flush=True)
            run.zero_()
        if it % args.val_every == 0 or it == args.iters:
            log = dict(iter=it, **validate(ema, "regress", sigma, val, args.device, 1))
            print(json.dumps(log), flush=True)
            with open(out / "log.jsonl", "a") as f:
                f.write(json.dumps(log) + "\n")
            # objective/features/sigma in the train.py format, so evaluate.py loads it as a regression model
            saved = dict(vars(args), objective="regress")
            torch.save(dict(model=ema.state_dict(), sigma=sigma, args=saved, iter=it,
                            train_model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            gen=gen.get_state(), rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state() if amp else None), out / "tmp.pt")
            (out / "tmp.pt").replace(out / "last.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--init", required=True, help="train.py regression checkpoint (read only)")
    p.add_argument("--cache", default="data/processed/acdc")
    p.add_argument("--out", required=True)
    p.add_argument("--lam", type=float, default=0.5, help="diffusion weight; the teacher uses 0.5")
    p.add_argument("--alpha", type=float, default=0.0, help="teacher-SVF anchor weight (normalised MSE)")
    p.add_argument("--win", type=int, default=21, help="LNCC window; the teacher uses 21")
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--augment", action="store_true", help="same similarity + intensity augmentation as train.py")
    p.add_argument("--features", default=None, help="CineMA feature root; required if the init used features")
    p.add_argument("--log_every", type=int, default=250)
    p.add_argument("--val_every", type=int, default=1000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available() else "cpu")
    main(p.parse_args())
