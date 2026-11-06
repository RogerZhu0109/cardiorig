"""
Train the rig network on compact rig numbers fitted to the teacher motion. Same protocol as
train.py: fixed budget, final EMA, no checkpoint selection, resumes from out/last.pt.
Usage: python scripts/train_rig.py --fold 0 --features data/processed/acdc_cinema --out checkpoints/cv/rig_f0
"""
import argparse
import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.augment import intensity, random_similarity  # noqa: E402
from src.data.cache import load  # noqa: E402
from src.models.rig_net import RigNet, beta_nll  # noqa: E402
from src.rig import SHARED, fit_compact, patient_fields, token_maps  # noqa: E402


def stack_inputs(d: dict, fields=None, slots=None) -> dict:
    """Network inputs of one patient, numpy float32, from the ED frame and ED mask only."""
    if fields is None:
        fields, slots, _ = patient_fields(d["ed_mask"])
    onehot = np.stack([d["ed_mask"] == k for k in (1, 2, 3)], 1).astype(np.float32)
    return dict(x=np.concatenate([d["cine"][int(d["ed"])][:, None].astype(np.float32), onehot], 1),
                maps=token_maps(fields, slots), feat=d["feat"].astype(np.float32), cls=d["cls"].astype(np.float32))


def collate(items: list[dict], device) -> tuple:
    x = torch.cat([i["x"] for i in items]).to(device)
    maps = torch.cat([i["maps"] for i in items]).to(device)
    feat = torch.cat([i["feat"] for i in items]).to(device)
    cls = torch.stack([i["cls"] for i in items]).to(device)
    owner = torch.cat([torch.full((len(i["x"]),), k) for k, i in enumerate(items)]).to(device)
    return x, maps, owner, feat, cls


def similarity(x, maps, feat, owner, generator):
    """One similarity per patient, on the whole stack. Returns transformed inputs and the 2x2 A per patient."""
    n = int(owner.max()) + 1
    A, b = random_similarity(n, x.shape[-1], 30.0, 0.1, 8.0, generator)
    A, b = A.to(x.device), b.to(x.device)
    theta = torch.cat([torch.linalg.inv(A), b[:, :, None]], 2)[owner]
    grid = F.affine_grid(theta, list(x.shape), align_corners=True)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    maps = F.grid_sample(maps, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    g = F.affine_grid(theta, list(feat.shape), align_corners=False)
    return x, maps, F.grid_sample(feat, g, mode="bilinear", padding_mode="border", align_corners=False), A


def transform_targets(a: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Rig numbers under a similarity: radial/tangential ones scale, translation turns with the heart."""
    scale = torch.linalg.det(A).abs().sqrt()[:, None, None]
    out = a * scale
    out[:, :2] = torch.einsum("bij,bjk->bik", A, a[:, :2])
    return out


def main(args):
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    patients = load(args.cache, "train", features=args.features, fold=args.fold)
    items, targets = [], []
    for d in patients:
        fields, slots, _ = patient_fields(d["ed_mask"])
        targets.append(fit_compact(d).astype(np.float32))
        item = stack_inputs(d, fields, slots)
        items.append({k: torch.from_numpy(v) for k, v in item.items()})
    del patients
    targets = torch.from_numpy(np.stack(targets)).to(args.device)
    used = (targets.abs().sum(2) > 0)[:, :, None].float()  # a slot unused by a patient stays exactly 0
    mean, std = targets.mean(0), targets.std(0).clamp_min(1e-4)
    print(f"{len(items)} train patients, {SHARED} motions, device {args.device}", flush=True)

    model = RigNet().to(args.device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
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

    t0, run = time.time(), 0.0
    for it in range(start, args.iters + 1):
        idx = torch.randint(len(items), (args.batch,), generator=gen)
        x, maps, owner, feat, cls = collate([items[i] for i in idx.tolist()], args.device)
        x, maps, feat, A = similarity(x, maps, feat, owner, gen)
        x = torch.cat([intensity(x[:, :1], gen), x[:, 1:]], 1)
        idx = idx.to(args.device)
        y = (transform_targets(targets[idx], A) - mean) / std
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            m, log_var = model(x, maps, owner, feat, cls)
        l = beta_nll(m.float(), log_var.float(), y, used[idx])
        opt.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        decay = min(args.ema, (1 + it) / (10 + it))
        with torch.no_grad():
            torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - decay)
        run += l.detach()
        if it % args.log_every == 0:
            print(f"it {it} loss {run.item() / args.log_every:.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{(it - start + 1) / (time.time() - t0):.1f} it/s", flush=True)
            run = 0.0
        if it % args.save_every == 0 or it == args.iters:
            torch.save(dict(model=ema.state_dict(), mean=mean, std=std, args=vars(args), iter=it,
                            train_model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            gen=gen.get_state(), rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state() if amp else None), out / "tmp.pt")
            (out / "tmp.pt").replace(out / "last.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data/processed/acdc")
    p.add_argument("--features", required=True, help="CineMA feature root")
    p.add_argument("--out", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--batch", type=int, default=8, help="patients per step")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available() else "cpu")
    main(p.parse_args())
