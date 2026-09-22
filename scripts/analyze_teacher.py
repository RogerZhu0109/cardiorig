"""
Where does the teacher under-contract?  From the cache, warp each ED mask to ES with
the teacher SVF and compare to the ES labels per slice position (basal / mid /
apical thirds of the kept stack; base = end with the larger ED LV area).
Through-plane motion (base descending at ES) would concentrate the LV-area excess
in basal slices; a regularisation bias would spread it evenly.
Usage: python scripts/analyze_teacher.py --cache data/processed/acdc
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.acdc import GROUPS  # noqa: E402
from src.utils.metrics import dice  # noqa: E402
from src.utils.svf import integrate, warp  # noqa: E402

POSITIONS = ("basal", "mid", "apical")


def main(args):
    rows = []
    for f in sorted(Path(args.cache).glob("*/patient*.npz")):
        d = np.load(f)
        es, ed_mask, es_mask = int(d["es"]), d["ed_mask"], d["es_mask"]
        u = integrate(torch.from_numpy(d["svf"][es].astype(np.float32)).to(args.device))
        onehot = F.one_hot(torch.from_numpy(ed_mask).long(), 4).permute(0, 3, 1, 2).float().to(args.device)
        pred = warp(onehot, u).argmax(1).cpu().numpy()
        thirds = np.array_split(np.arange(len(ed_mask)), 3)
        if (ed_mask[thirds[0]] == 3).sum() < (ed_mask[thirds[-1]] == 3).sum():
            thirds = thirds[::-1]
        for pos, idx in zip(POSITIONS, thirds):
            gt_lv, pred_lv, ed_lv = ((m[idx] == 3).sum() for m in (es_mask, pred, ed_mask))
            rows.append(dict(group=str(d["group"]), pos=pos, ed_lv=ed_lv, gt_lv=gt_lv, pred_lv=pred_lv,
                             **dice(pred[idx], es_mask[idx])))

    print(f"{len(rows) // 3} patients.  LV contraction = 1 - ES area / ED area (summed over slices)")
    print(f"{'':14s} {'true':>6s} {'teacher':>8s} {'captured':>9s}  {'LV Dice':>7s} {'MYO Dice':>8s}")
    for label, sel in [(p, [r for r in rows if r["pos"] == p]) for p in POSITIONS] + \
                      [(g, [r for r in rows if r["group"] == g]) for g in GROUPS]:
        ed, gt, pr = (sum(r[k] for r in sel) for k in ("ed_lv", "gt_lv", "pred_lv"))
        true_c, teach_c = 1 - gt / ed, 1 - pr / ed
        print(f"{label:14s} {true_c:6.1%} {teach_c:8.1%} {teach_c / true_c:9.1%}  "
              f"{np.nanmean([r['LV'] for r in sel]):7.3f} {np.nanmean([r['MYO'] for r in sel]):8.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data/processed/acdc")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(p.parse_args())
