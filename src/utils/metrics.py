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
