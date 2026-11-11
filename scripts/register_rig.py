"""
Rig as a registration model (pre-registered in EXPERIMENTS.md, "Rig registration"): the compact rig numbers
of a patient fitted directly to the images with the teacher's symmetric LNCC and no smoothness term, against
the teacher and a spline code with the same sharing. Nothing is trained; fits are cached per patient.
Usage: python scripts/register_rig.py --cache data/processed/acdc        (100 training patients, ES labels)
       python scripts/register_rig.py --straus data/processed/straus     (18 sequences, true motion)
       python scripts/register_rig.py                                    (re-print from the CSVs in --out)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_straus import consistent, decoded, geometry, load_straus, score_frames  # noqa: E402
from src.data.cache import load  # noqa: E402
from src.evaluation import score  # noqa: E402
from src.rig import SHARED, decode, fit_compact, patient_fields, spline_basis, time_basis  # noqa: E402
from src.teacher import lncc  # noqa: E402
from src.utils.svf import integrate, warp  # noqa: E402

SPLINE_GRID = 6  # 2 * 6^2 = 72 shared motions, 288 numbers (rig: 58 motions, 232 numbers)
CODE = dict(ceiling="rig", rig_registration="rig", spline_registration="spline")  # method -> decoder
PAIRS = (("rig_registration", "teacher"), ("rig_registration", "spline_registration"), ("rig_registration", "ceiling"))


def code_fields(mask: np.ndarray, code: str) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    """Decoder fields per slice of an ED stack, their shared slots, and the number of shared motions."""
    if code == "rig":
        fields, slots, _ = patient_fields(mask)
        return fields, slots, SHARED
    n = 2 * SPLINE_GRID ** 2
    return [spline_basis(m, SPLINE_GRID) for m in mask], [np.arange(n)] * len(mask), n


def fit_images(ed: np.ndarray, frames: np.ndarray, tau: np.ndarray, fields, slots, size: int, device,
               levels=(4, 2, 1), iters: int = 150, lr: float = 0.1, win: int = 21, harmonics: int = 2) -> np.ndarray:
    """Numbers (size, 2 * harmonics) whose SVFs register the ED stack (Z, H, W) to frames (F, Z, H, W) at phases tau."""
    basis = [torch.from_numpy(b).float().to(device) for b in fields]
    index = [torch.from_numpy(np.asarray(i)).to(device) for i in slots]
    phi = torch.from_numpy(time_basis(tau, harmonics)).float().to(device)  # (F, curves)
    fixed = torch.from_numpy(frames.astype(np.float32)).to(device).flatten(0, 1)[:, None]  # frame-major pairs
    moving = torch.from_numpy(ed.astype(np.float32)).to(device).repeat(len(frames), 1, 1)[:, None]
    numbers = torch.zeros(size, phi.shape[1], device=device, requires_grad=True)
    for s in levels:
        m, f = (F.avg_pool2d(x, s) if s > 1 else x for x in (moving, fixed))
        opt = torch.optim.Adam([numbers], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        trace = []
        for _ in range(iters):
            c = numbers @ phi.T  # (size, F)
            v = torch.stack([torch.einsum("pf,pchw->fchw", c[i], b) for b, i in zip(basis, index)], 1).flatten(0, 1)
            if s > 1:
                v = F.avg_pool2d(v, s) / s
            loss = -lncc(warp(m, integrate(v / 2)), warp(f, integrate(-v / 2)), win).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            trace.append(-loss.item())
        print(f"    level {s}: LNCC {trace[0]:.4f} -> {trace[2 * iters // 3]:.4f} -> {trace[-1]:.4f}", flush=True)
    return numbers.detach().cpu().numpy()


def fits(d: dict, out: Path, device) -> dict[str, np.ndarray]:
    """Compact numbers of every method for one patient; the image fits are cached in out/fits/{pid}.npz."""
    path = out / "fits" / f"{d['pid']}.npz"
    if not path.exists():
        frames = [t for t in range(len(d["tau"])) if t != int(d["ed"])]
        cine, res = d["cine"].astype(np.float32), {}
        for method in ("rig_registration", "spline_registration"):
            print(f"  {d['pid']} {method}", flush=True)
            res[method] = fit_images(cine[int(d["ed"])], cine[frames], d["tau"][frames], *code_fields(d["ed_mask"], CODE[method]), device)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **res)
    with np.load(path) as f:
        return dict({k: f[k] for k in f.files}, ceiling=fit_compact(d))


def acdc(args) -> pd.DataFrame:
    rows, out = [], Path(args.out)
    patients = load(args.cache, "train") + load(args.cache, "val")
    for i, d in enumerate(patients[:args.limit]):
        d["pid"], es = str(d["pid"]), int(d["es"])
        stacks = dict(identity=np.zeros_like(d["svf"][es], dtype=np.float32), teacher=d["svf"][es].astype(np.float32))
        for method, numbers in fits(d, out, args.device).items():
            fields, slots, _ = code_fields(d["ed_mask"], CODE[method])
            stacks[method] = decode(numbers, float(d["tau"][es]), fields, slots)
        rows += [dict(pid=d["pid"], group=str(d["group"]), method=m, **score(d, torch.from_numpy(v).float().to(args.device)))
                 for m, v in stacks.items()]
        patients[i] = None
        print(f"{i + 1}/{len(patients)} {d['pid']}", flush=True)
    return pd.DataFrame(rows)


def straus(args) -> pd.DataFrame:
    rows, out = [], Path(args.out) / "straus"
    for d in load_straus(args.straus, None)[:args.limit]:
        er, seg = geometry(d["ed_mask"])
        fields = dict(identity=np.zeros_like(d["svf"]), teacher=d["svf"])
        for method, numbers in fits(d, out, args.device).items():
            f, slots, _ = code_fields(d["ed_mask"], CODE[method])
            fields[method] = decoded(numbers, d, f, slots)
        rows += [dict(pid=d["pid"], anatomy=d["anatomy"], condition=d["group"], method=m, **score_frames(d, v, er, seg, args.device)[0])
                 for m, v in fields.items()]
        print(f"{d['pid']} done", flush=True)
    return pd.DataFrame(rows)


def captured(rows: pd.DataFrame) -> float:
    """Fraction of the true ED -> ES LV area reduction reproduced, pooled over rows."""
    return (1 - rows.pred_lv.sum() / rows.ed_lv.sum()) / (1 - rows.gt_lv.sum() / rows.ed_lv.sum())


def report_acdc(rows: pd.DataFrame, n_boot: int = 10000) -> None:
    rows["dice"] = rows[["dice_RV", "dice_MYO", "dice_LV"]].mean(axis=1)
    by = rows.groupby("method", sort=False)
    mean = by[["dice", "dice_RV", "dice_MYO", "dice_LV", "folded_pct"]].mean()
    mean["lv_captured"] = by.apply(captured)
    groups = rows.groupby(["method", "group"], sort=False).apply(captured).unstack()
    print(f"== ACDC, {rows.pid.nunique()} patients, ES ==\n" + mean.join(groups.add_prefix("captured_")).round(4).to_string())
    dice = rows.pivot(index="pid", columns="method", values="dice")
    idx = np.random.default_rng(0).integers(len(dice), size=(n_boot, len(dice)))
    for a, b in PAIRS:
        diff = (dice[a] - dice[b]).to_numpy()
        lo, hi = np.percentile(diff[idx].mean(1), [2.5, 97.5])
        verdict = "BEATS" if lo > 0 else "MATCHES" if lo > -0.01 else "SMALL COST" if diff.mean() > -0.02 else "WORSE"
        print(f"Dice {a} - {b}: {diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] -> {verdict}")
    area = {m: rows[rows.method == m].set_index("pid").loc[dice.index, ["ed_lv", "gt_lv", "pred_lv"]] for m in ("rig_registration", "teacher")}
    gain = np.array([captured(area["rig_registration"].iloc[i]) - captured(area["teacher"].iloc[i]) for i in idx[:2000]])
    lo, hi = np.percentile(gain, [2.5, 97.5])
    print(f"LV contraction captured, rig_registration - teacher: {captured(area['rig_registration']) - captured(area['teacher']):+.3f} "
          f"[{lo:+.3f}, {hi:+.3f}] -> {'recovers more contraction' if lo > 0 else 'no more contraction'}")


def report_straus(rows: pd.DataFrame) -> None:
    rows["kind"] = rows.condition.map(lambda c: c if c in ("healthy", "lbbb") else "infarct")
    cols = ["epe_es", "epe_all", "dice_es", "energy", "radial", "radial_true", "folded_pct", "epe_shape_es"]
    for kind, part in rows.groupby("kind"):
        print(f"== STRAUS {kind} ({part.pid.nunique()} sequences); epe_shape_es is exploratory ==\n"
              + part.groupby("method", sort=False)[cols].mean().round(3).to_string())
    healthy = rows[rows.condition == "healthy"].pivot(index="anatomy", columns="method", values="epe_es")
    print("-- healthy, ES EPE (mm) per anatomy\n" + healthy.round(2).to_string())
    for a, b in PAIRS:
        print(consistent(healthy, a, b))


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.cache:
        acdc(args).to_csv(out / "acdc.csv", index=False)
    if args.straus:
        straus(args).to_csv(out / "straus.csv", index=False)
    pd.set_option("display.width", 250)
    if args.limit:  # error check: no results are read
        return
    if (out / "acdc.csv").exists():
        report_acdc(pd.read_csv(out / "acdc.csv"))
    if (out / "straus.csv").exists():
        report_straus(pd.read_csv(out / "straus.csv"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache")
    p.add_argument("--straus")
    p.add_argument("--out", default="outputs/rig_registration")
    p.add_argument("--limit", type=int, help="first patients only (error check)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(p.parse_args())
