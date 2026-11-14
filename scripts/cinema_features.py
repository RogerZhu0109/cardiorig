"""
Frozen CineMA (Fu et al. 2025, arXiv:2506.00679; MAE pretrained on UK Biobank only) features
of each patient's ED short-axis stack. Input is the ED frame only, exactly as cached by
ACDCPatient.processed(); no other frame or label is used.
Our 128x128 @1.5 mm crop spans the same 192 mm field of view as CineMA's 192x192 @1 mm input,
so it is resampled in-plane to 192x192 and zero-padded to 16 slices (slice spacing unchanged).
Out, one file per patient, out_dir/{training,testing}/patientXXX.npz:
    feat (Z, 768, 12, 12) f16, indexed (slice, channel, y, x) like the cache images;  cls (768,) f16
Runs in a separate environment, .venv-cinema, with CineMA installed (github.com/mathpluscode/CineMA).
Usage: .venv-cinema/bin/python scripts/cinema_features.py --data_root data/raw/ACDC --out_dir data/processed/acdc_cinema
       .venv-cinema/bin/python scripts/cinema_features.py --cache data/processed/straus --out_dir data/processed/straus/cinema
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from cinema import CineMA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.acdc import ACDCPatient, list_patients  # noqa: E402

DEPTH = 16


def encode(model, ed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ed (Z, Y, X) in [0, 1] -> feat (Z, 768, 12, 12), cls (768,). Stacks deeper than 16 use two windows."""
    z = len(ed)
    x = torch.from_numpy(ed).permute(2, 1, 0)[None, None]  # (1, 1, X, Y, Z), CineMA axis order
    x = F.interpolate(x, size=(192, 192, z), mode="trilinear", align_corners=False)
    x = (x - x.min()) / (x.max() - x.min())
    starts = [0] if z <= DEPTH else [0, z - DEPTH]
    feat, cls = np.zeros((z, 768, 12, 12), np.float32), []
    for s in starts:
        w = F.pad(x[..., s:s + DEPTH], (0, max(0, DEPTH - z)))
        with torch.no_grad():
            f = model.feature_forward({"sax": w})
        grid = f["sax"][0].T.reshape(768, 12, 12, DEPTH)  # (C, x, y, z)
        grid = grid.permute(3, 0, 2, 1).numpy()  # (z, C, y, x)
        for k in range(min(z, DEPTH)):
            j = s + k  # a slice in both windows takes the window where it is more central
            if len(starts) == 1 or abs(k - DEPTH / 2) <= abs(j - starts[1 - starts.index(s)] - DEPTH / 2):
                feat[j] = grid[k]
        cls.append(f["cls"][0, 0].numpy())
    return feat.astype(np.float16), np.mean(cls, 0).astype(np.float16)


def main(args):
    model = CineMA.from_pretrained().eval()
    out = Path(args.out_dir)
    if args.cache:  # any cache of processed stacks (STRAUS): the ED stack of every file
        for f in sorted(Path(args.cache).glob("t*ing/*.npz")):
            with np.load(f) as d:
                feat, cls = encode(model, d["cine"][int(d["ed"])].astype(np.float32))
            (out / f.parent.name).mkdir(parents=True, exist_ok=True)
            np.savez(out / f.parent.name / f.name, feat=feat, cls=cls)
            print(f.stem, feat.shape, flush=True)
        return
    patients = list_patients(Path(args.data_root))
    for p in reversed(patients) if args.reverse else patients:  # a second process can work from the other end
        pat = ACDCPatient(p)
        dst = out / pat.info.split / f"{pat.pid}.npz"
        if dst.exists():
            continue
        d = pat.processed()
        feat, cls = encode(model, d["cine"][d["ed"]])
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dst, feat=feat, cls=cls)
        print(pat.pid, pat.info.split, feat.shape, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/raw/ACDC")
    p.add_argument("--cache", help="read ED stacks from a processed cache instead of raw ACDC")
    p.add_argument("--out_dir", default="data/processed/acdc_cinema")
    p.add_argument("--reverse", action="store_true")
    main(p.parse_args())
