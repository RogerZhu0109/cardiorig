"""
Train the PhaseFlow reimplementation (src/models/phaseflow.py) on one fold, in three resumable
stages: segmentation network, VAE, then flow and deformation decoder. Each stage's result is a
file in --out, and the fold's held-out patients are never read. Fixed budget, final EMA.
Usage: python scripts/train_phaseflow.py --fold 0 --out checkpoints/published/phaseflow_m0_f0
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
from src.data.cache import load, ordered, resampled  # noqa: E402
from src.models.phaseflow import ALPHA, T, DeformDecoder, SegUNet, VAE3D, VelocityUNet, bending, nonlinear_phase  # noqa: E402
from src.teacher import lncc  # noqa: E402
from src.utils.svf import integrate, warp  # noqa: E402

GROUPS = ("DCM", "HCM", "MINF", "NOR", "RV")
CLIP = 8  # VAE training frames: the 3D convolutions do not depend on T
LAMBDA_LNCC, LAMBDA_BEND, LAMBDA_VOL = 1.0, 0.001, 2.0


def train_seg(patients: list[dict], args, path: Path) -> SegUNet:
    """Segmentation on the fold's ED and ES labels, with similarity augmentation."""
    model = SegUNet().to(args.device)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=args.device, weights_only=False)["model"])
        return model.eval()
    x = np.concatenate([d["cine"][int(d[k])] for d in patients for k in ("ed", "es")]).astype(np.float32)
    y = np.concatenate([d[f"{k}_mask"] for d in patients for k in ("ed", "es")]).astype(np.int64)
    x, y = torch.from_numpy(x)[:, None].to(args.device), torch.from_numpy(y).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.seg_iters)
    gen, t0 = torch.Generator().manual_seed(args.seed), time.time()
    for it in range(1, args.seg_iters + 1):
        i = torch.randint(len(x), (16,), generator=gen).to(args.device)
        A, b = random_similarity(16, x.shape[-1], 30.0, 0.1, 8.0, gen)
        grid = F.affine_grid(torch.cat([A, b[:, :, None]], 2).to(args.device), [16, 1, *x.shape[-2:]], align_corners=True)
        xb = F.grid_sample(x[i], grid, padding_mode="zeros", align_corners=True)
        yb = F.grid_sample(y[i][:, None].float(), grid, mode="nearest", align_corners=True)[:, 0].long()
        logits = model(xb)
        p = logits.softmax(1)
        one = F.one_hot(yb, 4).permute(0, 3, 1, 2).float()
        soft_dice = 1 - (2 * (p * one).sum((0, 2, 3)) / (p + one).sum((0, 2, 3)).clamp_min(1)).mean()
        l = F.cross_entropy(logits, yb) + soft_dice
        opt.zero_grad(set_to_none=True)
        l.backward()
        opt.step()
        sched.step()
        if it % 500 == 0:
            print(f"seg it {it} loss {l.item():.4f} {it / (time.time() - t0):.1f} it/s", flush=True)
    torch.save(dict(model=model.state_dict()), path)
    return model.eval()


@torch.no_grad()
def lv_probability(seg: SegUNet, frames: np.ndarray, device) -> np.ndarray:
    """Soft LV map (N, H, W) of frames (N, H, W), chunked."""
    x = torch.from_numpy(frames.astype(np.float32))[:, None]
    return np.concatenate([seg(c.to(device)).softmax(1)[:, 3].cpu().numpy() for c in x.split(64)])


def slices_of(d: dict, seg: SegUNet, device) -> list[dict]:
    """Per-slice samples of a patient: 30-frame cine, LV areas, nonlinear phase, soft ED LV mask."""
    cine = resampled(ordered(d["cine"], d["tau"]).astype(np.float32), T)
    n, out = cine.shape[1], []
    for z in range(n):
        p = lv_probability(seg, cine[:, z], device)
        area = p.sum((1, 2)).astype(np.float32)
        out.append(dict(cine=cine[:, z].astype(np.float16), area=area, phase=nonlinear_phase(area).astype(np.float32),
                        lv=p[0].astype(np.float16), pos=z / max(n - 1, 1)))
    return out


def templates(patients: list[list[dict]], groups: list[str]) -> dict:
    """Mean phase curve per pathology, Sec. 3.2."""
    return {g: np.mean([s["phase"] for p, h in zip(patients, groups) if h == g for s in p], 0) for g in GROUPS}


def batch_of(samples: list[dict], device) -> dict:
    stack = lambda k: torch.from_numpy(np.stack([s[k] for s in samples]).astype(np.float32)).to(device)  # noqa: E731
    out = dict(x=stack("cine")[:, None] * 2 - 1, area=stack("area"), phase=stack("phase"), lv=stack("lv")[:, None],
               pos=torch.tensor([s["pos"] for s in samples], dtype=torch.float32, device=device))
    return out | ({"z1": stack("z1"), "z0": stack("z0")} if "z1" in samples[0] else {})


def train_vae(samples: list[dict], args, path: Path) -> VAE3D:
    model = VAE3D().to(args.device)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=args.device, weights_only=False)["model"])
        return model.eval().requires_grad_(False)
    iters = args.vae_epochs * len(samples) // args.batch
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    gen, state, begin = torch.Generator().manual_seed(args.seed), path.with_name("vae_last.pt"), 1
    if state.exists():
        ck = torch.load(state, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model"]), opt.load_state_dict(ck["opt"]), gen.set_state(ck["gen"].cpu())
        begin = ck["iter"] + 1
        print(f"vae resumed from iter {ck['iter']}", flush=True)
    t0, amp = time.time(), args.device == "cuda"
    for it in range(begin, iters + 1):
        idx = torch.randint(len(samples), (args.batch,), generator=gen).tolist()
        x = batch_of([samples[i] for i in idx], args.device)["x"]
        first = torch.randint(T - CLIP + 1, (1,), generator=gen).item()
        x = x[:, :, first:first + CLIP]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            recon, kl = model(x)
            l = F.l1_loss(recon, x) + 1e-6 * kl
        opt.zero_grad(set_to_none=True)
        l.backward()
        opt.step()
        if it % 500 == 0:
            print(f"vae it {it} of {iters} loss {l.item():.4f} {(it - begin + 1) / (time.time() - t0):.1f} it/s", flush=True)
        if it % args.save_every == 0:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), gen=gen.get_state(), iter=it), path.with_name("tmp.pt"))
            path.with_name("tmp.pt").replace(state)
    torch.save(dict(model=model.state_dict()), path)
    return model.eval().requires_grad_(False)


@torch.no_grad()
def anchor(vae: VAE3D, x: torch.Tensor) -> torch.Tensor:
    """Latent of the ED frame tiled over T, before anchor noise (Eq. 2)."""
    return vae.encode(x[:, :, :1].expand_as(x))[0]


@torch.no_grad()
def add_latents(samples: list[dict], vae: VAE3D, device) -> None:
    """z1 and the noise-free anchor per slice; the frozen encoder need not run during flow training."""
    for s in samples:
        x = batch_of([s], device)["x"]
        s["z1"] = vae.encode(x)[0][0].half().cpu().numpy()
        s["z0"] = anchor(vae, x)[0].half().cpu().numpy()


def main(args):
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    patients = load(args.cache, "train", svf=False, fold=args.fold)
    seg = train_seg(patients, args, out / "seg.pt")
    per_patient = [slices_of(d, seg, args.device) for d in patients]
    groups = [str(d["group"]) for d in patients]
    del patients
    phase_templates = templates(per_patient, groups)
    samples = [s for p in per_patient for s in p]
    del per_patient
    vae = train_vae(samples, args, out / "vae.pt")
    add_latents(samples, vae, args.device)
    del vae
    iters = args.epochs * len(samples) // args.batch
    print(f"{len(samples)} slices, {iters} iterations, device {args.device}", flush=True)

    net, decoder = VelocityUNet().to(args.device), DeformDecoder().to(args.device)
    params = list(net.parameters()) + list(decoder.parameters())
    ema_net = copy.deepcopy(net).eval().requires_grad_(False)
    ema_dec = copy.deepcopy(decoder).eval().requires_grad_(False)
    ema_params = list(ema_net.parameters()) + list(ema_dec.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    gen = torch.Generator().manual_seed(args.seed)
    amp = args.device == "cuda"
    start = 1
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=args.device, weights_only=False)
        net.load_state_dict(ck["train_net"]), decoder.load_state_dict(ck["train_dec"])
        ema_net.load_state_dict(ck["net"]), ema_dec.load_state_dict(ck["dec"])
        opt.load_state_dict(ck["opt"]), sched.load_state_dict(ck["sched"])
        gen.set_state(ck["gen"].cpu()), torch.set_rng_state(ck["rng"].cpu())
        start = ck["iter"] + 1
        print(f"resumed from iter {ck['iter']}", flush=True)

    t0, run = time.time(), torch.zeros(4)
    for it in range(start, iters + 1):
        idx = torch.randint(len(samples), (args.batch,), generator=gen).tolist()
        b = batch_of([samples[i] for i in idx], args.device)
        x = b["x"]
        z1, z0 = b["z1"], b["z0"] + ALPHA * torch.randn(b["z0"].shape, generator=gen).to(args.device)
        tau = torch.rand(args.batch, generator=gen).to(args.device)
        s = tau[:, None, None, None, None]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            v = net((1 - s) * z0 + s * z1, tau, b["phase"], b["pos"])
            flow = F.mse_loss(v, z1 - z0)
            svf = decoder(v)
        with torch.autocast("cuda", enabled=False):  # LNCC needs float32 (Appendix E)
            n = x.shape[0] * T
            phi = integrate(svf.float().reshape(n, *svf.shape[2:]))
            ed = (x[:, :, 0] + 1) / 2
            warped = warp(ed.repeat_interleave(T, 0), phi)
            real = (x[:, 0].reshape(n, 1, *x.shape[-2:]) + 1) / 2
            sim = 1 - lncc(warped, real).mean()
            bend = bending(phi)
            lv = warp(b["lv"].repeat_interleave(T, 0), phi).reshape(args.batch, T, *x.shape[-2:])
            present = b["area"][:, 0] >= 20  # a slice without an LV has no meaningful volume ratio
            vol = (((lv.sum((2, 3)) - b["area"]).abs().mean(1) / b["area"][:, 0].clamp_min(1)) * present).sum() / present.sum().clamp_min(1)
            l = flow + LAMBDA_LNCC * sim + LAMBDA_BEND * bend + LAMBDA_VOL * vol
        opt.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            torch._foreach_lerp_(ema_params, params, 1 - min(args.ema, (1 + it) / (10 + it)))
        run += torch.stack([flow.detach().cpu().float(), sim.detach().cpu(), bend.detach().cpu(), vol.detach().cpu()])
        if it % args.log_every == 0:
            r = (run / args.log_every).tolist()
            print(f"it {it} flow {r[0]:.4f} lncc {r[1]:.4f} bend {r[2]:.5f} vol {r[3]:.4f} "
                  f"{(it - start + 1) / (time.time() - t0):.1f} it/s", flush=True)
            run = torch.zeros(4)
        if it % args.save_every == 0 or it == iters:
            torch.save(dict(net=ema_net.state_dict(), dec=ema_dec.state_dict(), train_net=net.state_dict(), train_dec=decoder.state_dict(),
                            templates=phase_templates, opt=opt.state_dict(), sched=sched.state_dict(), gen=gen.get_state(),
                            rng=torch.get_rng_state(), iter=it, args=vars(args)), out / "tmp.pt")
            (out / "tmp.pt").replace(out / "last.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data/processed/acdc")
    p.add_argument("--out", required=True)
    p.add_argument("--fold", type=int, required=True, help="cross-validation fold 0-4, or -1 for all 100 training patients")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--vae_epochs", type=int, default=100)
    p.add_argument("--seg_iters", type=int, default=12000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--features", default=None, help="ignored; run_cv.py passes it to every model")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available() else "cpu")
    main(p.parse_args())
