"""
M&Ms cine loader (Campello et al., IEEE TMI 2021): 345 subjects, 4 vendors, 5 centres.
Evaluation only, never training.

data/raw/MnMs/{Training/Labeled,Training/Unlabeled,Validation,Testing}/CODE/
    CODE_sa.nii.gz      (X, Y, Z, T) float32 cine
    CODE_sa_gt.nii.gz   same grid, labels 1 LV cavity, 2 myocardium, 3 RV cavity at the ED and ES frames only
data/raw/MnMs/*.csv     per subject: vendor, centre, pathology, ED and ES (0-based frames), age, sex, height, weight
The 25 Training/Unlabeled subjects (all vendor C) are labelled too, but the table gives ED = ES = 0 for
them: their frames are the two labelled ones, ED the one with the larger LV cavity.

The stack is brought to the ACDC convention the models were trained on, as for STRAUS: labels
1 RV, 2 MYO, 3 LV; slice 0 basal; in-plane quarter turns so that the LV -> RV direction is nearest
the ACDC training mean; 1.5 mm; 128 x 128 crop on the ED-mask centroid; slices with ED foreground
only; intensities scaled by ED-frame percentiles. The slice order comes from the header (the slice
axis points from base to apex when it points left, anterior and inferior). With slices ordered
base to apex, the M&Ms arrays have the handedness of the ACDC arrays, except one subject (R8V0Y4)
whose header says it is a mirror image: it is flipped in-plane.
"""
from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

from src.data.acdc import center_crop, phase_tau
from src.data.straus import ACDC_SEPTAL

TO_ACDC = np.array([0, 3, 2, 1], np.uint8)  # M&Ms label -> ACDC label
APEX = np.array([-1.0, 1.0, -1.0]) / 3 ** 0.5  # base -> apex in RAS: left, anterior, inferior


def read_table(root: Path) -> dict[str, dict]:
    with open(next(Path(root).glob("*.csv"))) as f:
        return {r["External code"]: r for r in csv.DictReader(f)}


def list_patients(root: Path) -> list[Path]:
    return sorted((p.parent for p in Path(root).rglob("*_sa.nii.gz")), key=lambda p: p.name)


def septal_angle(mask: np.ndarray) -> float:
    """LV -> RV direction (radians, x right, y down) between the cavity centroids of an ACDC-labelled stack."""
    (_, yr, xr), (_, yl, xl) = np.nonzero(mask == 1), np.nonzero(mask == 3)
    return float(np.arctan2(yr.mean() - yl.mean(), xr.mean() - xl.mean()))


def turns_to_acdc(angle: float) -> int:
    """Quarter turns (np.rot90) that bring an LV -> RV direction nearest the ACDC training mean."""
    return int(np.argmin([np.abs(np.angle(np.exp(1j * (angle - r * np.pi / 2 - ACDC_SEPTAL)))) for r in range(4)]))


class MnMsPatient:
    def __init__(self, pdir: Path, row: dict):
        self.dir = Path(pdir)
        self.pid = self.dir.name
        self.split = self.dir.parent.name  # Labeled / Unlabeled / Validation / Testing
        self.row = row

    def processed(self, size: int = 128, target_mm: float = 1.5) -> dict:
        """
        The keys of ACDCPatient.processed() plus vendor, centre, split, turns, mirrored:
            cine (T, Z', size, size) float32 in [0, 1];  ed_mask, es_mask (Z', size, size) uint8
            z (Z',) kept slice indices of the file, basal first;  tau (T,)
        """
        img = nib.load(self.dir / f"{self.pid}_sa.nii.gz")
        gt = nib.load(self.dir / f"{self.pid}_sa_gt.nii.gz")
        a, (sx, sy) = img.affine[:3, :3], map(float, img.header.get_zooms()[:2])
        to_apex = bool(a[:, 2] @ APEX > 0)
        mirrored = (np.linalg.det(a) < 0) != to_apex
        order = (slice(None) if to_apex else slice(None, None, -1), slice(None), slice(None, None, -1) if mirrored else slice(None))

        ed, es = int(self.row["ED"]), int(self.row["ES"])
        if ed == es:  # not in the table: the two labelled frames
            lv = (np.rint(np.asarray(gt.dataobj)) == 1).sum((0, 1, 2))
            es, ed = (int(t) for t in sorted(np.nonzero(lv)[0], key=lambda t: lv[t]))
        cine = np.asarray(img.dataobj, dtype=np.float32).transpose(3, 2, 1, 0)[(slice(None), *order)]
        lo, hi = np.percentile(cine[ed], [1, 99])  # ED frame only, as for ACDC
        cine = np.clip((cine - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
        ed_mask, es_mask = (TO_ACDC[np.rint(np.asarray(gt.dataobj[..., t])).astype(int)].transpose(2, 1, 0)[order]
                            for t in (ed, es))

        turns = turns_to_acdc(septal_angle(ed_mask))
        if turns % 2:
            sx, sy = sy, sx
        cine = zoom(np.rot90(cine, turns, axes=(-2, -1)), (1, 1, sy / target_mm, sx / target_mm), order=1)
        ed_mask, es_mask = (zoom(np.rot90(m, turns, axes=(-2, -1)), (1, sy / target_mm, sx / target_mm), order=0)
                            for m in (ed_mask, es_mask))

        z = np.nonzero((ed_mask > 0).any(axis=(1, 2)))[0]
        _, yy, xx = np.nonzero(ed_mask > 0)
        center = (int(round(yy.mean())), int(round(xx.mean())))
        n = len(cine)
        return dict(
            pid=self.pid, group=self.row["Pathology"], vendor=self.row["Vendor"], centre=int(self.row["Centre"]),
            split=self.split, ed=ed, es=es, turns=turns, mirrored=mirrored,
            tau=np.array([phase_tau(t, ed, n) for t in range(n)], np.float32),
            z=np.arange(img.shape[2])[order[0]][z], center=np.array(center),
            cine=center_crop(cine[:, z], center, size),
            ed_mask=center_crop(ed_mask[z], center, size),
            es_mask=center_crop(es_mask[z], center, size),
        )
