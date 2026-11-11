"""
Build the M&Ms cache (evaluation only, never training): ACDC-style slices (src/data/mnms.py) plus
teacher SVFs (ED -> every frame). One file per subject, out_dir/testing/CODE.npz, with the keys of
the ACDC cache (cine, svf, ed_mask, es_mask, z, tau, center, ed, es, group = pathology) and vendor,
centre, split (the M&Ms challenge split), turns, mirrored. out_dir/qc.csv holds the teacher against no motion
at ES and the sanity checks on the conventions: the LV cavity lies inside the myocardium, shrinks
from ED to ES, and the LV -> RV direction is near the ACDC one. Existing files are skipped.
Usage: python scripts/preprocess_mnms.py --data_root data/raw/MnMs --out_dir data/processed/mnms
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_fill_holes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.mnms import MnMsPatient, list_patients, read_table, septal_angle  # noqa: E402
from src.data.straus import ACDC_SEPTAL  # noqa: E402
from src.evaluation import score  # noqa: E402
from src.teacher import register_batched  # noqa: E402
from src.utils.svf import integrate, jacobian_det  # noqa: E402


def process(pat: MnMsPatient, args) -> tuple[dict, dict]:
    d = pat.processed(size=args.size)
    (T, Z), s = d["cine"].shape[:2], args.size
    frames = [t for t in range(T) if t != d["ed"]]
    cine = torch.from_numpy(d["cine"]).to(args.device)
    moving = cine[d["ed"]].repeat(len(frames), 1, 1).reshape(-1, 1, s, s)

    t0 = time.time()
    v = register_batched(moving, cine[frames].reshape(-1, 1, s, s), chunk=args.chunk, lam=args.lam)
    dt = time.time() - t0
    svf = torch.zeros(T, Z, 2, s, s, device=args.device)
    svf[frames] = v.reshape(len(frames), Z, 2, s, s)
    jd = jacobian_det(integrate(v))

    found, still = score(d, svf[d["es"]]), score(d, torch.zeros_like(svf[d["es"]]))
    lv, myo = d["ed_mask"] == 3, d["ed_mask"] == 2
    qc = dict(pid=d["pid"], split=d["split"], vendor=d["vendor"], centre=d["centre"], group=d["group"], frames=T, slices=Z,
              turns=d["turns"], reversed=bool(d["z"][0] > d["z"][-1]), mirrored=d["mirrored"], ed=d["ed"], es=d["es"],
              septal_off=float(np.degrees(np.angle(np.exp(1j * (septal_angle(d["ed_mask"]) - ACDC_SEPTAL))))),
              lv_inside_myo=float((lv & np.stack([binary_fill_holes(m) for m in myo])).sum() / max(lv.sum(), 1)),
              lv_ed=int(lv.sum()), lv_es=int((d["es_mask"] == 3).sum()), seconds=round(dt, 1),
              folded_px=int((jd <= 0).sum()), absJ1=(jd - 1).abs().mean().item(),
              **{f"dice_{k}": found[f"dice_{k}"] for k in ("RV", "MYO", "LV")},
              **{f"id_{k}": still[f"dice_{k}"] for k in ("RV", "MYO", "LV")})
    d["cine"], d["svf"] = d["cine"].astype(np.float16), svf.cpu().numpy().astype(np.float16)
    return d, qc


def main(args):
    out, root = Path(args.out_dir), Path(args.data_root)
    (out / "testing").mkdir(parents=True, exist_ok=True)
    qc_path = out / "qc.csv"
    rows = pd.read_csv(qc_path).to_dict("records") if qc_path.exists() else []
    table, patients, skipped = read_table(root), list_patients(root), []
    print(f"{len(patients)} subjects, {len(rows)} already done, device {args.device}", flush=True)
    for p in patients:
        if any(r["pid"] == p.name for r in rows):
            continue
        try:
            d, qc = process(MnMsPatient(p, table[p.name]), args)
        except (AssertionError, ValueError) as e:  # a subject that breaks a convention: reported, not cached
            skipped.append(p.name)
            print(f"{p.name} SKIPPED: {type(e).__name__} {e}", flush=True)
            continue
        np.savez(out / "testing" / f"{p.name}.npz", **{k: np.asarray(x) for k, x in d.items()})
        rows.append(qc)
        pd.DataFrame(rows).to_csv(qc_path, index=False)
        print(f"{p.name} {qc['vendor']} {qc['group']:5s} {qc['frames']}x{qc['slices']} turns {qc['turns']} {qc['seconds']:6.1f}s "
              f"ES Dice {np.nanmean([qc[f'dice_{k}'] for k in ('RV', 'MYO', 'LV')]):.3f} vs "
              f"{np.nanmean([qc[f'id_{k}'] for k in ('RV', 'MYO', 'LV')]):.3f} folded {qc['folded_px']}", flush=True)

    print(f"skipped {len(skipped)}: {' '.join(skipped)}")
    df = pd.DataFrame(rows)
    df["dice"], df["identity"] = df[["dice_RV", "dice_MYO", "dice_LV"]].mean(1), df[["id_RV", "id_MYO", "id_LV"]].mean(1)
    print(df.groupby("vendor")[["dice", "identity", "dice_RV", "dice_MYO", "dice_LV"]].mean().round(3).assign(n=df.groupby("vendor").size()).to_string())
    print(f"all: {len(df)} subjects | teacher ES Dice {df.dice.mean():.3f} | identity {df.identity.mean():.3f} | folded px total {df.folded_px.sum()}")
    checks = {"LV cavity inside the myocardium (> 0.5 of its pixels)": df.lv_inside_myo > 0.5,
              "LV cavity shrinks from ED to ES": df.lv_es < df.lv_ed,
              "LV -> RV direction within 45 degrees of ACDC": df.septal_off.abs() <= 45,
              "teacher beats no motion at ES": df.dice > df.identity}
    for name, ok in checks.items():
        print(f"CHECK {name}: {int(ok.sum())}/{len(ok)}" + ("" if ok.all() else "  failing: " + " ".join(df.pid[~ok][:20])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/raw/MnMs")
    parser.add_argument("--out_dir", default="data/processed/mnms")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available() else "cpu")
    main(parser.parse_args())
