"""
Export the generated cardiac cycle of the final seed-0 models for a few ACDC test subjects, for the cycle figure.
The fields over the T = 30 resampled frames are made as in scripts/evaluate_test.py; this scores nothing new.
Writes {out}/{pid}.npz for the middle mid-ventricular slice: the real frames, the LV area curve of the
segmentation network on the real frames (the reference of the volume metrics), and per method the warped ED
slice, its labels and the LV area curve over the stack.
Usage: python scripts/export_cycles.py --runs checkpoints/final --cache data/processed/acdc --features data/processed/acdc_cinema
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_cv import load_current, load_rig, predict_rig  # noqa: E402
from scripts.evaluate_published import frames_of, krebs_fields, load_phaseflow, phaseflow_fields, predicted_area  # noqa: E402
from scripts.train_phaseflow import lv_probability  # noqa: E402
from src.data.cache import load, ordered, resampled  # noqa: E402
from src.models.unet import predict  # noqa: E402
from src.models.krebs import Krebs  # noqa: E402
from src.models.phaseflow import T  # noqa: E402
from src.rig import decode, patient_fields, positions  # noqa: E402
from src.utils.svf import integrate, warp  # noqa: E402


@torch.no_grad()
def main(args):
    device, runs, out = args.device, Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kr = Krebs().to(device).eval()
    kr.load_state_dict(torch.load(runs / "krebs_s0" / "last.pt", map_location=device, weights_only=False)["model"])
    pf = load_phaseflow(runs / "phaseflow_s0", device)
    current, sigma = load_current(runs / "current_s0" / "last.pt", device)
    rig, mean, std = load_rig(runs / "rig_s0" / "last.pt", device)
    gen = torch.Generator().manual_seed(0)
    for d in load(args.cache, "testing", features=args.features):
        es, pos = frames_of(d)
        order = np.argsort(d["tau"], kind="stable")
        cine = resampled(ordered(d["cine"], d["tau"]).astype(np.float32), T)
        truth = np.array([(lv_probability(pf["seg"], cine[t], device) >= 0.5).sum() for t in range(T)], float)
        taus = pos / len(d["tau"])
        fields, slots, _ = patient_fields(d["ed_mask"])
        numbers, _ = predict_rig(rig, mean, std, d, device, fields, slots)
        ed = torch.from_numpy(d["cine"][int(d["ed"])][:, None].astype(np.float32)).to(device)
        feat = torch.from_numpy(d["feat"].astype(np.float32)).to(device)
        cls = torch.from_numpy(d["cls"].astype(np.float32)).to(device)[None].expand(len(ed), -1)
        at = lambda t: torch.full((len(ed),), float(t), device=device)  # noqa: E731
        curves = dict(krebs_mean=krebs_fields(kr, d, None, device),
                      phaseflow_tau0=phaseflow_fields(pf, d, 0.0, device, gen),
                      current=torch.stack([predict(current, ed, at(t), sigma, feat=feat, cls=cls) for t in taus]),
                      rig=torch.from_numpy(np.stack([decode(numbers, float(t), fields, slots) for t in taus])).float().to(device),
                      teacher=torch.from_numpy(d["svf"][order[np.round(pos).astype(int)]].astype(np.float32)).to(device))
        mids = [z for z, p in sorted(positions(d["ed_mask"]).items()) if p == "mid"]
        z = mids[len(mids) // 2]
        onehot = F.one_hot(torch.from_numpy(d["ed_mask"][z:z + 1]).long(), 4).permute(0, 3, 1, 2).float().to(device)
        save = dict(real=cine[:, z], truth_area=truth, tau=taus, es=es, z=z, ed_mask=d["ed_mask"][z], es_mask=d["es_mask"][z],
                    group=str(d["group"]))
        for name, v in curves.items():
            u = integrate(v[:, z].float())
            save[f"{name}_img"] = warp(ed[z:z + 1].expand(T, -1, -1, -1), u)[:, 0].cpu().numpy().astype(np.float32)
            save[f"{name}_lab"] = warp(onehot.expand(T, -1, -1, -1), u).argmax(1).cpu().numpy().astype(np.uint8)
            save[f"{name}_area"] = predicted_area(v.float(), d)
            print(f"{d['pid']} {name}: LV area ED {save[f'{name}_area'][0]:.0f} min {save[f'{name}_area'].min():.0f} "
                  f"(reference {truth[0]:.0f} / {truth.min():.0f})", flush=True)
        np.savez_compressed(out / f"{d['pid']}.npz", **save)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True, help="folder with {rig,current,phaseflow,krebs}_s0")
    p.add_argument("--cache", required=True, help="cache root holding only the chosen subjects under testing/")
    p.add_argument("--features", required=True)
    p.add_argument("--out", default="outputs/cycles")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(p.parse_args())
