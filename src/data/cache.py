"""
Cached ACDC slices + teacher SVFs (scripts/preprocess.py) as generator training pairs.

Splits are by patient, never by slice:
    train    ACDC training minus VAL_PATIENTS (90)
    val      VAL_PATIENTS (10): monitoring and hyper-parameter choices only
    testing  ACDC testing (50): final evaluation only; its SVFs are teacher references, never inputs
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# Last 2 patients of each pathology block (ACDC training is ordered in blocks of 20);
# disjoint from TUNE_PATIENTS, the teacher tuning set.
VAL_PATIENTS = [f"patient{b + i:03d}" for b in (1, 21, 41, 61, 81) for i in (18, 19)]


def fold_patients(fold: int) -> list[str]:
    """Held-out patients of a cross-validation fold: 4 per pathology block, every patient once in folds 0-4."""
    rng, out = np.random.default_rng(0), []
    for b in range(1, 101, 20):
        out += [f"patient{b + i:03d}" for i in rng.permutation(20)[fold * 4:(fold + 1) * 4]]
    return sorted(out)


def split_ids(root: Path, split: str, fold: int | None = None) -> list[str]:
    folder = "testing" if split == "testing" else "training"
    ids = sorted(p.stem for p in (Path(root) / folder).glob("patient*.npz"))
    held_out = VAL_PATIENTS if fold is None else fold_patients(fold)
    if split == "val":
        return [p for p in ids if p in held_out]
    if split == "train":
        return [p for p in ids if p not in held_out]
    assert split == "testing", split
    return ids


def ordered(cine: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Frames (N, ...) in cardiac order, ED first: tau is ED-relative."""
    return cine[np.argsort(tau, kind="stable")]


def resampled(x: np.ndarray, t: int) -> np.ndarray:
    """Linear interpolation of the first axis to t frames; both ends are kept."""
    pos = np.linspace(0, len(x) - 1, t)
    lo = np.minimum(pos.astype(int), len(x) - 2)
    w = (pos - lo).reshape(-1, *[1] * (x.ndim - 1))
    return ((1 - w) * x[lo] + w * x[lo + 1]).astype(x.dtype)


def load(root: Path, split: str, svf: bool = True, features: Path | None = None, fold: int | None = None) -> list[dict]:
    """features: optional root of scripts/cinema_features.py output (ED-stack CineMA features).
    fold: cross-validation fold replacing the fixed val patients."""
    sub = "testing" if split == "testing" else "training"
    keys = ["pid", "group", "ed", "es", "tau", "z", "cine", "ed_mask", "es_mask"] + (["svf"] if svf else [])
    out = []
    for pid in split_ids(root, split, fold):
        with np.load(Path(root) / sub / f"{pid}.npz") as f:
            d = {k: f[k] for k in keys}
        if features:
            with np.load(Path(features) / sub / f"{pid}.npz") as f:
                d["feat"], d["cls"] = f["feat"], f["cls"]
            assert len(d["feat"]) == len(d["z"]), pid
        out.append(d)
    return out


class Pairs:
    """All (ED slice, tau, teacher SVF) pairs, i.e. every kept slice x every non-ED frame, held on device."""

    def __init__(self, patients: list[dict], device: str | torch.device):
        eds, svfs, taus, ref, n = [], [], [], [], 0
        for d in patients:
            frames = [t for t in range(len(d["tau"])) if t != d["ed"]]
            z = len(d["z"])
            eds.append(d["cine"][d["ed"]])
            svfs.append(d["svf"][frames].reshape(-1, *d["svf"].shape[2:]))  # frame-major
            taus.append(np.repeat(d["tau"][frames], z))
            ref.append(n + np.tile(np.arange(z), len(frames)))
            n += z
        self.ed = torch.from_numpy(np.concatenate(eds)[:, None]).to(device)  # (slices, 1, S, S) f16
        self.svf = torch.from_numpy(np.concatenate(svfs)).to(device)  # (pairs, 2, S, S) f16
        self.tau = torch.from_numpy(np.concatenate(taus)).to(device)
        self.ref = torch.from_numpy(np.concatenate(ref)).to(device)
        self.feat = self.cls = None
        if "feat" in patients[0]:  # per slice, like self.ed
            self.feat = torch.from_numpy(np.concatenate([d["feat"] for d in patients])).to(device)
            self.cls = torch.from_numpy(np.concatenate([np.repeat(d["cls"][None], len(d["z"]), 0)
                                                        for d in patients])).to(device)

    def __len__(self):
        return len(self.svf)

    def rms(self, chunk: int = 4096) -> float:
        """Root-mean-square SVF component (pixels): the global scale sigma, w1 = v / sigma."""
        sq = sum(self.svf[i:i + chunk].float().pow(2).sum().item() for i in range(0, len(self), chunk))
        return (sq / self.svf.numel()) ** 0.5

    def sample(self, n: int, generator: torch.Generator | None = None):
        """Uniform over pairs. Returns ed (n,1,S,S), tau (n,), svf (n,2,S,S), cond (dict: feat, cls or empty), float32."""
        idx = torch.randint(len(self), (n,), generator=generator).to(self.svf.device)
        s = self.ref[idx]
        cond = {} if self.feat is None else dict(feat=self.feat[s].float(), cls=self.cls[s].float())
        return self.ed[s].float(), self.tau[idx].float(), self.svf[idx].float(), cond
