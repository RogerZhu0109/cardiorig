"""
Sanity check for the ACDC loader: dataset statistics + one patient's ED / mid / ES slices.
Usage: python scripts/inspect_acdc.py --root data/raw/ACDC --patient patient001 --out outputs/inspect
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.acdc import ACDCPatient, dataset_summary, list_patients, phase_tau  # noqa: E402


def main(args):
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- dataset statistics ------------------------------------------------
    infos = dataset_summary(root)
    nf = np.array([i.n_frames for i in infos])
    sp = np.array([i.spacing for i in infos])
    ed_not_first = [i.pid for i in infos if i.ed != 0]
    print(f"patients: {len(infos)}  (train {sum(i.split=='training' for i in infos)}, test {sum(i.split=='testing' for i in infos)})")
    print(f"groups: {dict(Counter(i.group for i in infos))}")
    print(f"NbFrame: min {nf.min()} median {int(np.median(nf))} max {nf.max()}  |  N<16: {(nf<16).sum()} patients")
    print(f"in-plane mm: {sp[:,0].min():.2f}-{sp[:,0].max():.2f}   slice mm: {sorted(set(np.round(sp[:,2],1)))}")
    print(f"ED != frame 1: {ed_not_first}")
    print(f"tau_ES: min {min(i.tau_es for i in infos):.2f} median {np.median([i.tau_es for i in infos]):.2f} max {max(i.tau_es for i in infos):.2f}")
    n_pairs = sum(ACDCPatient(p).shape[2] * (ACDCPatient(p).shape[3] - 1) for p in list_patients(root, "training"))
    print(f"2D training pairs (ED-slice -> frame-slice, raw): {n_pairs}")

    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    ax[0].hist(nf, bins=range(10, 42, 2)); ax[0].set_title("NbFrame"); ax[0].set_xlabel("frames / cycle")
    ax[1].hist(sp[:, 0], bins=20); ax[1].set_title("in-plane spacing (mm)")
    fig.tight_layout(); fig.savefig(out / "acdc_stats.png", dpi=120)

    # ---- one patient ------------------------------------------------------
    pat = ACDCPatient(root / "training" / args.patient)
    i = pat.info
    mid = (i.ed + i.n_frames // 2) % i.n_frames
    z = pat.shape[2] // 2
    frames = {"ED": i.ed, "ES": i.es, "mid-cycle": mid}
    print(f"\n{i.pid}: {i.group}, shape {pat.shape}, spacing {tuple(round(s,2) for s in i.spacing)}, ED={i.ed} ES={i.es} N={i.n_frames}")

    fig, ax = plt.subplots(2, 3, figsize=(10, 6.5))
    for k, (name, t) in enumerate(frames.items()):
        img = pat.resample_inplane(pat.frame(t))[z]
        ax[0, k].imshow(img, cmap="gray"); ax[0, k].set_title(f"{name}  t={t}  tau={phase_tau(t, i.ed, i.n_frames):.2f}")
        m = pat.mask(t)
        ax[1, k].imshow(img, cmap="gray")
        if m is not None:
            ax[1, k].imshow(np.ma.masked_equal(pat.resample_inplane(m, order=0)[z], 0), cmap="jet", alpha=0.4, vmin=1, vmax=3)
            ax[1, k].set_title("labels: 1 RV, 2 MYO, 3 LV")
        else:
            ax[1, k].set_title("(no label)")
        for a in ax[:, k]: a.axis("off")
    fig.suptitle(f"{i.pid} ({i.group}) slice z={z}, resampled 1.5 mm")
    fig.tight_layout(); fig.savefig(out / f"{i.pid}_phases.png", dpi=120)

    # ---- pair generator smoke test ---------------------------------------
    pairs = list(pat.pairs())
    ex = pairs[0]
    print(f"pairs(): {len(pairs)} pairs, ed {ex['ed'].shape} {ex['ed'].dtype}, target {ex['target'].shape}, "
          f"tau range {min(p['tau'] for p in pairs):.2f}-{max(p['tau'] for p in pairs):.2f}, "
          f"with ES mask: {sum(p['es_mask'] is not None for p in pairs)}")
    print(f"figures -> {out}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/raw/ACDC")
    p.add_argument("--patient", default="patient001")
    p.add_argument("--out", default="outputs/inspect")
    main(p.parse_args())
