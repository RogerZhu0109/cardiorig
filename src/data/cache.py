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


def split_ids(root: Path, split: str) -> list[str]:
    folder = "testing" if split == "testing" else "training"
    ids = sorted(p.stem for p in (Path(root) / folder).glob("patient*.npz"))
    if split == "val":
        return [p for p in ids if p in VAL_PATIENTS]
    if split == "train":
        return [p for p in ids if p not in VAL_PATIENTS]
    assert split == "testing", split
    return ids


def load(root: Path, split: str, svf: bool = True, features: Path | None = None) -> list[dict]:
    """features: optional root of scripts/cinema_features.py output (ED-stack CineMA features)."""
    sub = "testing" if split == "testing" else "training"
    keys = ["pid", "group", "ed", "es", "tau", "z", "cine", "ed_mask", "es_mask"] + (["svf"] if svf else [])
    out = []
    for pid in split_ids(root, split):
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

    def __init__(self, patients: list[dict], device: str | torch.device, targets: bool = False):
        """targets: also hold each pair's real target frame (image-consistency fine-tuning)."""
        eds, svfs, taus, ref, tgts, n = [], [], [], [], [], 0
        for d in patients:
            frames = [t for t in range(len(d["tau"])) if t != d["ed"]]
            z = len(d["z"])
            eds.append(d["cine"][d["ed"]])
            svfs.append(d["svf"][frames].reshape(-1, *d["svf"].shape[2:]))  # frame-major
            if targets:
                tgts.append(d["cine"][frames].reshape(-1, *d["cine"].shape[2:]))  # same order as svfs
            taus.append(np.repeat(d["tau"][frames], z))
            ref.append(n + np.tile(np.arange(z), len(frames)))
            n += z
        self.ed = torch.from_numpy(np.concatenate(eds)[:, None]).to(device)  # (slices, 1, S, S) f16
        self.svf = torch.from_numpy(np.concatenate(svfs)).to(device)  # (pairs, 2, S, S) f16
        self.tau = torch.from_numpy(np.concatenate(taus)).to(device)
        self.ref = torch.from_numpy(np.concatenate(ref)).to(device)
        self.target = torch.from_numpy(np.concatenate(tgts)[:, None]).to(device) if targets else None  # like svf
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

    def sample(self, n: int, generator: torch.Generator | None = None, target: bool = False):
        """Uniform over pairs. Returns ed (n,1,S,S), tau (n,), svf (n,2,S,S), cond (dict: feat, cls or empty), float32;
        with target=True also the real target frame (n,1,S,S) last."""
        idx = torch.randint(len(self), (n,), generator=generator).to(self.svf.device)
        s = self.ref[idx]
        cond = {} if self.feat is None else dict(feat=self.feat[s].float(), cls=self.cls[s].float())
        out = self.ed[s].float(), self.tau[idx].float(), self.svf[idx].float(), cond
        return (*out, self.target[idx].float()) if target else out
