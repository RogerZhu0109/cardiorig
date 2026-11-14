"""
ACDC cine-CMR loader.

Produces heart-centred 2D short-axis stacks. tau is the ED-relative, cyclic, normalized cardiac phase:
    tau = ((t - ED) mod N) / N,  ED -> 0.0, ES -> ~0.3-0.45.
Per-patient files (data/raw/ACDC/{training,testing}/patientXXX/):
    Info.cfg                      ED / ES (1-based frame index), Group, Height, Weight, NbFrame
    patientXXX_4d.nii.gz          (X, Y, Z, T) int16 cine, no temporal spacing (pixdim[4]=1)
    patientXXX_frameNN.nii.gz     ED / ES frame; *_gt.nii.gz = labels 0 bg,1 RV,2 MYO,3 LV
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

LABELS = {0: "background", 1: "RV", 2: "MYO", 3: "LV"}
GROUPS = ("NOR", "MINF", "DCM", "HCM", "RV")
# Registration hyper-parameter tuning set: first 4 training patients of each group
# (ACDC training is ordered in blocks of 20: DCM, HCM, MINF, NOR, RV).
TUNE_PATIENTS = [f"patient{b + i:03d}" for b in (1, 21, 41, 61, 81) for i in range(4)]


@dataclass
class PatientInfo:
    pid: str
    split: str
    ed: int  # 0-based
    es: int  # 0-based
    n_frames: int
    group: str
    height: float
    weight: float
    spacing: tuple[float, float, float]  # x, y, z in mm


def phase_tau(t: int, ed: int, n_frames: int) -> float:
    """ED-relative cyclic normalized phase in [0, 1)."""
    return ((t - ed) % n_frames) / n_frames


def read_info(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def list_patients(root: Path, split: str | None = None) -> list[Path]:
    splits = [split] if split else ["training", "testing"]
    return sorted(p for s in splits for p in (root / s).glob("patient*") if p.is_dir())


class ACDCPatient:
    def __init__(self, pdir: Path):
        self.dir = Path(pdir)
        self.pid = self.dir.name
        cfg = read_info(self.dir / "Info.cfg")
        self._img = nib.load(self.dir / f"{self.pid}_4d.nii.gz")
        sx, sy, sz = map(float, self._img.header.get_zooms()[:3])
        self.info = PatientInfo(
            pid=self.pid,
            split=self.dir.parent.name,
            ed=int(cfg["ED"]) - 1,
            es=int(cfg["ES"]) - 1,
            n_frames=int(cfg["NbFrame"]),
            group=cfg["Group"],
            height=float(cfg["Height"]),
            weight=float(cfg["Weight"]),
            spacing=(sx, sy, sz),
        )
        assert self._img.shape[3] == self.info.n_frames, (self.pid, self._img.shape, cfg)

    @property
    def shape(self):
        return self._img.shape  # (X, Y, Z, T)

    def cine(self) -> np.ndarray:
        """(T, Z, Y, X) float32."""
        return np.asarray(self._img.dataobj, dtype=np.float32).transpose(3, 2, 1, 0)

    def frame(self, t: int) -> np.ndarray:
        """(Z, Y, X) float32 for frame t (0-based)."""
        return np.asarray(self._img.dataobj[..., t], dtype=np.float32).transpose(2, 1, 0)

    def mask(self, t: int) -> np.ndarray | None:
        """(Z, Y, X) uint8 labels; only ED / ES have masks."""
        p = self.dir / f"{self.pid}_frame{t + 1:02d}_gt.nii.gz"
        if not p.exists():
            return None
        return np.asarray(nib.load(p).dataobj, dtype=np.uint8).transpose(2, 1, 0)

    def resample_inplane(self, vol: np.ndarray, target_mm: float = 1.5, order: int = 1) -> np.ndarray:
        """Resample (Z, Y, X) or (..., Y, X) to isotropic in-plane spacing."""
        sx, sy, _ = self.info.spacing
        factors = [1.0] * (vol.ndim - 2) + [sy / target_mm, sx / target_mm]
        return zoom(vol, factors, order=order)

    def processed(self, size: int = 128, target_mm: float = 1.5) -> dict:
        """
        Whole sequence resampled to target_mm, cropped to size x size around the
        ED-mask centroid (one centre per patient), keeping only slices whose ED
        mask has foreground.
            cine:    (T, Z', size, size) float32 in [0, 1]
            ed_mask, es_mask: (Z', size, size) uint8
            z:       (Z',) kept slice indices;  tau: (T,) phase per frame
        """
        cine = self.cine()
        # Percentiles of the ED frame only: the model input must not depend on target frames.
        lo, hi = np.percentile(cine[self.info.ed], [1, 99])
        cine = np.clip((cine - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
        cine = self.resample_inplane(cine, target_mm)
        ed_mask = self.resample_inplane(self.mask(self.info.ed), target_mm, order=0)
        es_mask = self.resample_inplane(self.mask(self.info.es), target_mm, order=0)
        z = np.nonzero((ed_mask > 0).any(axis=(1, 2)))[0]
        _, yy, xx = np.nonzero(ed_mask > 0)
        center = (int(round(yy.mean())), int(round(xx.mean())))
        n = self.info.n_frames
        return dict(
            pid=self.pid,
            group=self.info.group,
            ed=self.info.ed,
            es=self.info.es,
            tau=np.array([phase_tau(t, self.info.ed, n) for t in range(n)], np.float32),
            z=z,
            center=np.array(center),
            cine=center_crop(cine[:, z], center, size),
            ed_mask=center_crop(ed_mask[z], center, size),
            es_mask=center_crop(es_mask[z], center, size),
        )


def center_crop(vol: np.ndarray, center: tuple[int, int], size: int) -> np.ndarray:
    """Crop the last two axes to size x size around center, zero-padding outside."""
    h, w = vol.shape[-2:]
    y0, x0 = center[0] - size // 2, center[1] - size // 2
    out = np.zeros(vol.shape[:-2] + (size, size), vol.dtype)
    sy, sx = max(y0, 0), max(x0, 0)
    ey, ex = min(y0 + size, h), min(x0 + size, w)
    out[..., sy - y0:ey - y0, sx - x0:ex - x0] = vol[..., sy:ey, sx:ex]
    return out
