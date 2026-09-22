"""Numpy-only metrics (importable from the separate ANTs environment)."""
from __future__ import annotations

import numpy as np

STRUCTURES = {1: "RV", 2: "MYO", 3: "LV"}


def dice(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Per-structure Dice over the whole array (e.g. a slice stack)."""
    out = {}
    for k, name in STRUCTURES.items():
        p, g = pred == k, gt == k
        denom = p.sum() + g.sum()
        out[name] = 2 * (p & g).sum() / denom if denom else np.nan
    return out


def lv_areas(d: dict, pred: np.ndarray) -> dict[str, int]:
    """LV pixel counts over the stack: ED labels, ES labels, prediction at ES."""
    return dict(ed_lv=int((d["ed_mask"] == 3).sum()), gt_lv=int((d["es_mask"] == 3).sum()), pred_lv=int((pred == 3).sum()))


def lv_captured(rows: list[dict]) -> float:
    """Fraction of the true ED -> ES LV area reduction reproduced by the prediction (pooled over rows)."""
    ed, gt, pred = (sum(r[k] for r in rows) for k in ("ed_lv", "gt_lv", "pred_lv"))
    return (1 - pred / ed) / (1 - gt / ed)
