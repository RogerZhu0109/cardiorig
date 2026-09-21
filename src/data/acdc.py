"""
ACDC cine-CMR loader.

Produces 2D short-axis (ED_slice, tau, target_slice) pairs for phase-conditioned
deformation generation.  tau is the ED-relative, cyclic, normalized cardiac phase:
    tau = ((t - ED) mod N) / N,  ED -> 0.0, ES -> ~0.3-0.45.
Note ACDC prospective gating truncates the last 5-10% of the cycle in some cases,
so tau never reaches 1.0 exactly.

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

    @property
    def tau_es(self) -> float:
        return phase_tau(self.es, self.ed, self.n_frames)


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

    # ---- raw access -------------------------------------------------------
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

    # ---- preprocessing ----------------------------------------------------
    def resample_inplane(self, vol: np.ndarray, target_mm: float = 1.5, order: int = 1) -> np.ndarray:
        """Resample (Z, Y, X) or (..., Y, X) to isotropic in-plane spacing."""
        sx, sy, _ = self.info.spacing
        factors = [1.0] * (vol.ndim - 2) + [sy / target_mm, sx / target_mm]
        return zoom(vol, factors, order=order)

    def pairs(self, target_mm: float | None = 1.5, normalize: bool = True):
        """
        Yield dicts with 2D slices for every (slice z, frame t != ED):
            ed:     (Y, X) float32 reference slice
            target: (Y, X) float32 slice at frame t
            tau:    float in [0, 1)
            z, t, pid, group, es_mask (only when t == ES, else None)
        """
        cine = self.cine()
        if normalize:
            lo, hi = np.percentile(cine, [1, 99])
            cine = np.clip((cine - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
        if target_mm:
            cine = self.resample_inplane(cine, target_mm)
        es_mask = self.mask(self.info.es)
        if es_mask is not None and target_mm:
            es_mask = self.resample_inplane(es_mask, target_mm, order=0)
        ed = self.info.ed
        for z in range(cine.shape[1]):
            for t in range(cine.shape[0]):
                if t == ed:
                    continue
                yield dict(
                    pid=self.pid,
                    group=self.info.group,
                    z=z,
                    t=t,
                    tau=phase_tau(t, ed, self.info.n_frames),
                    ed=cine[ed, z],
                    target=cine[t, z],
                    es_mask=es_mask[z] if (es_mask is not None and t == self.info.es) else None,
                )


def dataset_summary(root: Path) -> list[PatientInfo]:
    return [ACDCPatient(p).info for p in list_patients(root)]
