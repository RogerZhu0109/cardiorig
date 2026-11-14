"""
Published methods against the rig and the dense-field model on the same held-out patients. ES Dice
comes from warped ED labels; volume curves compare the warped ED LV mask with the fold's
segmentation network on real frames, over T = 30 resampled frames. ES labels only score.
Writes published.csv (one row per patient and method) and prints the comparison.
Usage: python scripts/evaluate_published.py --runs checkpoints/published --cv_runs checkpoints/cv --features data/processed/acdc_cinema
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_cv import load_current, load_rig, paired, predict_rig  # noqa: E402
from scripts.train_krebs import slots_of  # noqa: E402
from scripts.train_phaseflow import anchor, lv_probability  # noqa: E402
from src.data.cache import load, ordered, resampled  # noqa: E402
from src.evaluation import score  # noqa: E402
from src.models.unet import predict  # noqa: E402
from src.models.krebs import T_MAX, Krebs  # noqa: E402
from src.models.phaseflow import ALPHA, T, DeformDecoder, SegUNet, VAE3D, VelocityUNet  # noqa: E402
from src.rig import decode, patient_fields  # noqa: E402
from src.utils.metrics import STRUCTURES, dice  # noqa: E402
from src.utils.svf import integrate, warp  # noqa: E402


def frames_of(d: dict) -> tuple[int, np.ndarray]:
    """ES index in the 30-frame sequence, and the original-frame position of each of the 30 frames."""
    order = np.argsort(d["tau"], kind="stable")
    es = int(np.flatnonzero(order == int(d["es"]))[0])
    pos = np.linspace(0, len(order) - 1, T)
    return int(np.abs(pos - es).argmin()), pos


def curve_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Paper definitions (Appendix G) on one patient's LV area curves."""
    ef = lambda v: 100 * (v.max() - v.min()) / v.max()  # noqa: E731
    ss = ((truth - truth.mean()) ** 2).sum()
    return dict(ef_err=abs(ef(pred) - ef(truth)), vol_corr=np.corrcoef(pred, truth)[0, 1],
                vol_r2=1 - ((pred - truth) ** 2).sum() / ss, vol_mae=np.abs(pred - truth).mean() / truth.max())


@torch.no_grad()
def predicted_area(v: torch.Tensor, d: dict) -> np.ndarray:
    """LV area over the stack per frame, for SVFs (N, Z, 2, H, W) warping the ED LV mask."""
    lv = torch.from_numpy((d["ed_mask"] == 3).astype(np.float32))[:, None].to(v.device)
    out = []
    for frame in v:
        out.append(float((warp(lv, integrate(frame)) >= 0.5).sum()))
    return np.array(out)


def load_phaseflow(path: Path, device) -> dict:
    ck = torch.load(path / "last.pt", map_location=device, weights_only=False)
    net, dec = VelocityUNet().to(device).eval(), DeformDecoder().to(device).eval()
    net.load_state_dict(ck["net"]), dec.load_state_dict(ck["dec"])
    vae = VAE3D().to(device).eval()
    vae.load_state_dict(torch.load(path / "vae.pt", map_location=device, weights_only=False)["model"])
    return dict(net=net, dec=dec, vae=vae, templates=ck["templates"], seg=load_seg(path, device))


def load_seg(path: Path, device) -> SegUNet:
    seg = SegUNet().to(device).eval()
    seg.load_state_dict(torch.load(path / "seg.pt", map_location=device, weights_only=False)["model"])
    return seg


@torch.no_grad()
def phaseflow_fields(m: dict, d: dict, query: float, device, generator) -> torch.Tensor:
    """SVFs (T, Z, 2, H, W) from the ED frame, its pathology's phase template and anchor noise only."""
    ed = torch.from_numpy(d["cine"][int(d["ed"])].astype(np.float32)).to(device) * 2 - 1
    z = len(ed)
    x = ed[:, None, None].expand(-1, 1, T, -1, -1)
    phase = torch.from_numpy(m["templates"][str(d["group"])].astype(np.float32)).to(device).expand(z, -1)
    pos = torch.tensor([i / max(z - 1, 1) for i in range(z)], dtype=torch.float32, device=device)
    out = []
    for i in range(z):
        z0 = anchor(m["vae"], x[i:i + 1])
        z0 = z0 + ALPHA * torch.randn(z0.shape, generator=generator).to(device)
        out.append(m["dec"](m["net"](z0, torch.full((1,), query, device=device), phase[i:i + 1], pos[i:i + 1]))[0])
    return torch.stack(out, 1)


@torch.no_grad()
def krebs_fields(model: Krebs, d: dict, generator, device) -> torch.Tensor:
    """SVFs (T, Z, 2, H, W) at the T_MAX positions nearest the 30 frames."""
    ed = torch.from_numpy(d["cine"][int(d["ed"])][:, None].astype(np.float32)).to(device)
    v = torch.cat([model.simulate(c, generator) for c in ed.split(8)])
    pos = np.round(np.linspace(0, T_MAX - 1, T)).astype(int)
    return v[:, pos].transpose(0, 1)


def es_krebs(v_all: torch.Tensor, d: dict) -> torch.Tensor:
    """SVF stack at the patient's exact ES position, which is what its model saw in training."""
    order = np.argsort(d["tau"], kind="stable")
    return v_all[:, slots_of(len(order))[int(np.flatnonzero(order == int(d["es"]))[0])]]


@torch.no_grad()
def evaluate(args) -> pd.DataFrame:
    device, rows = args.device, []
    for fold in args.folds:
        pf = load_phaseflow(Path(args.runs) / f"phaseflow_m0_f{fold}", device)
        kr = Krebs().to(device).eval()
        kr.load_state_dict(torch.load(Path(args.runs) / f"krebs_m0_f{fold}" / "last.pt", map_location=device, weights_only=False)["model"])
        current, sigma = load_current(Path(args.cv_runs) / f"current_f{fold}" / "last.pt", device)
        rig, mean, std = load_rig(Path(args.cv_runs) / f"rig_f{fold}" / "last.pt", device)
        gen = torch.Generator().manual_seed(fold)
        for d in load(args.cache, "val", svf=False, features=args.features, fold=fold):
            es, pos = frames_of(d)
            cine = resampled(ordered(d["cine"], d["tau"]).astype(np.float32), T)
            truth = np.array([(lv_probability(pf["seg"], cine[t], device) >= 0.5).sum() for t in range(T)], float)
            taus = pos / len(d["tau"])
            fields, slots, _ = patient_fields(d["ed_mask"])
            numbers, _ = predict_rig(rig, mean, std, d, device, fields, slots)
            ed = torch.from_numpy(d["cine"][int(d["ed"])][:, None].astype(np.float32)).to(device)
            feat = torch.from_numpy(d["feat"].astype(np.float32)).to(device)
            cls = torch.from_numpy(d["cls"].astype(np.float32)).to(device)[None].expand(len(ed), -1)
            curves = dict(current=torch.stack([predict(current, ed, torch.full((len(ed),), float(t), device=device), sigma, feat=feat, cls=cls)
                                               for t in taus]),
                          rig=torch.from_numpy(np.stack([decode(numbers, float(t), fields, slots) for t in taus])).float().to(device),
                          phaseflow_tau1=phaseflow_fields(pf, d, 1.0, device, gen),
                          phaseflow_tau0=phaseflow_fields(pf, d, 0.0, device, gen),
                          krebs_mean=krebs_fields(kr, d, None, device))
            es_fields = {k: v[es] for k, v in curves.items()}
            tau_es = float(d["tau"][int(d["es"])])
            es_fields["current"] = predict(current, ed, torch.full((len(ed),), tau_es, device=device), sigma, feat=feat, cls=cls)
            es_fields["rig"] = torch.from_numpy(decode(numbers, tau_es, fields, slots)).float().to(device)
            es_fields["krebs_mean"] = es_krebs(torch.cat([kr.simulate(c) for c in ed.split(8)]), d)
            for name, v in es_fields.items():
                rows.append(dict(pid=str(d["pid"]), group=str(d["group"]), fold=fold, method=name, **score(d, v.float()),
                                 **curve_metrics(predicted_area(curves[name], d), truth)))
            draws = [es_krebs(torch.cat([kr.simulate(c, gen) for c in ed.split(8)]), d) for _ in range(5)]
            scored = pd.DataFrame([score(d, v.float()) for v in draws]).mean().to_dict()
            rows.append(dict(pid=str(d["pid"]), group=str(d["group"]), fold=fold, method="krebs_sample", **scored))
            es_cine = torch.from_numpy(d["cine"][int(d["es"])][:, None].astype(np.float32)).to(device)
            found = dice(pf["seg"](es_cine).argmax(1).cpu().numpy(), d["es_mask"])
            rows.append(dict(pid=str(d["pid"]), group=str(d["group"]), fold=fold, method="segmentation",
                             **{f"dice_{k}": x for k, x in found.items()}))
        print(f"fold {fold} done", flush=True)
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    cols = [f"dice_{k}" for k in STRUCTURES.values()]
    df["dice"] = df[cols].mean(axis=1)
    print("segmentation network on held-out ES:", df[df.method == "segmentation"][cols].mean().round(4).to_dict())
    df = df[df.method != "segmentation"]
    print(df.groupby("method", sort=False)[["dice", *cols, "folded_pct", "ef_err", "vol_corr", "vol_r2", "vol_mae"]].mean().round(4).to_string())
    print(df.pivot_table(index="group", columns="method", values="dice", sort=False).round(3).to_string())
    pooled = df.groupby("method", sort=False)[["ed_lv", "gt_lv", "pred_lv"]].sum()
    print("LV contraction:", (1 - pooled.pred_lv / pooled.ed_lv).round(3).to_dict(), "true", round(1 - pooled.gt_lv.iloc[0] / pooled.ed_lv.iloc[0], 3))
    wide = df.pivot(index="pid", columns="method", values="dice")
    for method in ("phaseflow_tau1", "phaseflow_tau0", "krebs_mean", "krebs_sample", "current"):
        print(paired(wide, "rig", method))


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = evaluate(args)
    df.to_csv(out / "published.csv", index=False)
    report(df)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True, help="folder with phaseflow_m0_f{0-4} and krebs_m0_f{0-4}")
    p.add_argument("--cv_runs", required=True, help="folder with current_f{0-4} and rig_f{0-4}")
    p.add_argument("--cache", default="data/processed/acdc")
    p.add_argument("--features", required=True)
    p.add_argument("--out", default="outputs/published")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(p.parse_args())
