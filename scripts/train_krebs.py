"""
Train the Krebs 2021 reimplementation (src/models/krebs.py) on one fold. Fixed budget, final EMA,
no checkpoint selection, resumes from out/last.pt. Image-only: neither teacher nor CineMA.
Usage: python scripts/train_krebs.py --fold 0 --out checkpoints/published/krebs_m0_f0
"""
import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.augment import random_similarity  # noqa: E402
from src.data.cache import load, ordered  # noqa: E402
from src.models.krebs import SIGMA_L, T_MAX, Krebs, kl_divergence  # noqa: E402
from src.utils.svf import integrate, warp  # noqa: E402


def slots_of(n: int) -> np.ndarray:
    """Positions of n frames spread evenly over the T_MAX latent columns, first at 0."""
    return np.round(np.linspace(0, T_MAX - 1, n)).astype(np.int64)


def augmented(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Same random similarity and mirror on every frame of one slice (N, 1, H, W)."""
    A, b = random_similarity(1, x.shape[-1], 30.0, 0.1, 8.0, gen)
    theta = torch.cat([A, b[:, :, None]], 2).to(x.device).expand(len(x), -1, -1)
    x = F.grid_sample(x, F.affine_grid(theta, list(x.shape), align_corners=True), padding_mode="zeros", align_corners=True)
    return x.flip(-1) if torch.rand(1, generator=gen) < 0.5 else x


def main(args):
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sequences = [s for d in load(args.cache, "train", svf=False, fold=args.fold)
                 for s in ordered(d["cine"], d["tau"]).transpose(1, 0, 2, 3)]
    print(f"{len(sequences)} slice sequences, device {args.device}", flush=True)
    model = Krebs().to(args.device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(args.seed)
    inv_k = torch.linalg.inv(model.kernel)
    start = 1
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=args.device, weights_only=False)
        model.load_state_dict(ck["train_model"]), ema.load_state_dict(ck["model"]), opt.load_state_dict(ck["opt"])
        gen.set_state(ck["gen"].cpu())
        start = ck["iter"] + 1
        print(f"resumed from iter {ck['iter']}", flush=True)

    t0, run = time.time(), 0.0
    for it in range(start, args.iters + 1):
        s = sequences[torch.randint(len(sequences), (1,), generator=gen).item()]
        x = augmented(torch.from_numpy(s.astype(np.float32))[:, None].to(args.device), gen)
        i0, frames, slots = x[:1], x[1:], torch.from_numpy(slots_of(len(x))[1:]).to(args.device)
        mu, log_sigma = model.posterior(model.encode(i0, frames, slots, drop=0.5))
        v = model.decode(model.sample(mu, log_sigma), i0)[0, slots]
        warped = warp(i0.expand(len(frames), -1, -1, -1), integrate(v))
        recon = (warped - frames).pow(2).sum() / (2 * SIGMA_L)
        l = recon + kl_divergence(mu, log_sigma, inv_k).sum()
        opt.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - min(args.ema, (1 + it) / (10 + it)))
        run += l.detach()
        if it % args.log_every == 0:
            print(f"it {it} loss {run.item() / args.log_every:.1f} {(it - start + 1) / (time.time() - t0):.1f} it/s", flush=True)
            run = 0.0
        if it % args.save_every == 0 or it == args.iters:
            torch.save(dict(model=ema.state_dict(), train_model=model.state_dict(), opt=opt.state_dict(),
                            gen=gen.get_state(), iter=it, args=vars(args)), out / "tmp.pt")
            (out / "tmp.pt").replace(out / "last.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data/processed/acdc")
    p.add_argument("--out", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=150000)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--features", default=None, help="ignored; run_cv.py passes it to every model")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available() else "cpu")
    main(p.parse_args())
