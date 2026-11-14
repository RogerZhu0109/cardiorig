"""
Export the ES predictions of the final seed-0 models for a few subjects, for the qualitative figure.
The ES fields are made as in scripts/evaluate_test.py; this scores nothing new. The printed Dice is a check
that the export reproduces test.csv / mnms.csv (rig, current, Krebs, teacher are deterministic; PhaseFlow
draws a new sample).
Writes {out}/{pid}.npz: ED and ES images and labels, and per method the warped ED image and labels.
Usage: python scripts/export_qualitative.py --runs checkpoints/final --cache data/processed/acdc --features data/processed/acdc_cinema
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_cv import load_current, load_rig, predict_rig  # noqa: E402
from scripts.evaluate_mnms import ACDC_GROUP  # noqa: E402
from scripts.evaluate_published import es_krebs, frames_of, load_phaseflow, phaseflow_fields  # noqa: E402
from src.data.cache import load  # noqa: E402
from src.evaluation import score  # noqa: E402
from src.models.unet import predict  # noqa: E402
from src.models.krebs import Krebs  # noqa: E402
from src.rig import decode, patient_fields  # noqa: E402
from src.utils.metrics import STRUCTURES  # noqa: E402
from src.utils.svf import integrate, warp  # noqa: E402


@torch.no_grad()
def es_fields(d: dict, models: dict, device, gen) -> dict:
    """ES SVF stack (Z, 2, S, S) per method, as scored in evaluate_test.py."""
    pf, kr, (current, sigma), (rig, mean, std) = models["phaseflow"], models["krebs"], models["current"], models["rig"]
    es, _ = frames_of(d)
    tau_es = float(d["tau"][int(d["es"])])
    fields, slots, _ = patient_fields(d["ed_mask"])
    numbers, _ = predict_rig(rig, mean, std, d, device, fields, slots)
    ed = torch.from_numpy(d["cine"][int(d["ed"])][:, None].astype(np.float32)).to(device)
    feat = torch.from_numpy(d["feat"].astype(np.float32)).to(device)
    cls = torch.from_numpy(d["cls"].astype(np.float32)).to(device)[None].expand(len(ed), -1)
    return dict(krebs_mean=es_krebs(torch.cat([kr.simulate(c) for c in ed.split(8)]), d),
                phaseflow_tau0=phaseflow_fields(pf, d, 0.0, device, gen)[es],
                current=predict(current, ed, torch.full((len(ed),), tau_es, device=device), sigma, feat=feat, cls=cls),
                rig=torch.from_numpy(decode(numbers, tau_es, fields, slots)).float().to(device),
                teacher=torch.from_numpy(d["svf"][int(d["es"])].astype(np.float32)).to(device))


def main(args):
    device, runs, out = args.device, Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kr = Krebs().to(device).eval()
    kr.load_state_dict(torch.load(runs / "krebs_s0" / "last.pt", map_location=device, weights_only=False)["model"])
    models = dict(phaseflow=load_phaseflow(runs / "phaseflow_s0", device), krebs=kr,
                  current=load_current(runs / "current_s0" / "last.pt", device),
                  rig=load_rig(runs / "rig_s0" / "last.pt", device))
    gen = torch.Generator().manual_seed(0)
    for d in load(args.cache, "testing", features=args.features):
        d["group"] = ACDC_GROUP.get(str(d["group"]), str(d["group"]))  # PhaseFlow's class, as in evaluate_mnms.py
        ed = torch.from_numpy(d["cine"][int(d["ed"])][:, None].astype(np.float32)).to(device)
        onehot = F.one_hot(torch.from_numpy(d["ed_mask"]).long(), 4).permute(0, 3, 1, 2).float().to(device)
        save = dict(ed=d["cine"][int(d["ed"])].astype(np.float32), es=d["cine"][int(d["es"])].astype(np.float32),
                    ed_mask=d["ed_mask"], es_mask=d["es_mask"], z=d["z"], group=str(d["group"]))
        for name, v in es_fields(d, models, device, gen).items():
            u = integrate(v.float())
            save[f"{name}_img"] = warp(ed, u)[:, 0].cpu().numpy().astype(np.float32)
            save[f"{name}_lab"] = warp(onehot, u).argmax(1).cpu().numpy().astype(np.uint8)
            s = score(d, v.float())
            print(f"{d['pid']} {name}: dice {np.mean([s[f'dice_{k}'] for k in STRUCTURES.values()]):.4f}", flush=True)
        np.savez_compressed(out / f"{d['pid']}.npz", **save)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True, help="folder with {rig,current,phaseflow,krebs}_s0")
    p.add_argument("--cache", required=True, help="cache root holding only the chosen subjects under testing/")
    p.add_argument("--features", required=True)
    p.add_argument("--out", default="outputs/qualitative")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(p.parse_args())
