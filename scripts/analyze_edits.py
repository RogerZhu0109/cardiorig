"""
Segment edits of the rig model on held-out patients: locality, regional-report edits with controls,
dose-response. Uses the held-out rig parameters saved by evaluate_cv.py and the dense-field model of
the fold that held each patient out. ES labels only score and build the report. Training patients only.
Usage: python scripts/analyze_edits.py --predictions outputs/cv/rig_predictions.npz --truth outputs/rig/segments.csv \
    --runs checkpoints/cv --features data/processed/acdc_cinema --out outputs/edits
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_cv import load_current, paired  # noqa: E402
from src.data.cache import fold_patients, load  # noqa: E402
from src.evaluation import es_inputs  # noqa: E402
from src.models.unet import predict  # noqa: E402
from src.rig import (BIN_CENTRES, BINS, DECAY, MIN_PX, OFFSET, SECTORS, TAU, _filled, _outer_radius,  # noqa: E402
                     _periodic_hats, edit, patient_fields, positions, rays, roi, sector_centres, sector_of_bins,
                     septal_angle, time_basis)
from src.utils.metrics import dice  # noqa: E402
from src.utils.svf import integrate, jacobian_det, warp  # noqa: E402

MM = 1.5
AKINETIC, HYPO = 0.10, 5  # relative thickening below 0.10; below the NOR 5th percentile
ALPHA = {"akinetic": 0.0, "hypokinetic": 0.5}
DOSE = (-0.5, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
RANDOM_DRAWS = 20
RING = {s: pos for pos in SECTORS for s in range(OFFSET[pos], OFFSET[pos] + SECTORS[pos])}


class Patient:
    """Rig geometry of one patient and batched decoding of edited numbers to ES SVF stacks."""

    def __init__(self, d: dict, numbers: np.ndarray, device):
        self.d, self.numbers, self.device = d, numbers, device
        mask = d["ed_mask"]
        fields, self.slots, _ = patient_fields(mask)
        self.fields = [torch.from_numpy(f).float().to(device) for f in fields]
        self.septum, self.where = septal_angle(mask), positions(mask)
        self.phi = time_basis(np.array([float(d["tau"][int(d["es"])])]), numbers.shape[1] // 2)[0]
        self.rays = {z: rays(mask[z]) for z in self.where}
        self.onehot = F.one_hot(torch.from_numpy(mask).long(), 4).permute(0, 3, 1, 2).float().to(device)
        self.myo = torch.from_numpy(mask == 2).to(device)
        self.roi = torch.from_numpy(np.stack([roi(m) for m in mask])).to(device)

    def decode(self, numbers: list[np.ndarray]) -> torch.Tensor:
        """(E, Z, 2, H, W) ES SVFs from E sets of numbers."""
        c = torch.from_numpy(np.stack([n @ self.phi for n in numbers])).float().to(self.device)
        return torch.stack([torch.einsum("ep,pchw->echw", c[:, idx], b) for b, idx in zip(self.fields, self.slots)], 1)

    def segment_of_pixels(self, z: int) -> np.ndarray:
        """Global LV segment index of every pixel of slice z (by angle around the ED LV centre)."""
        pos, k = self.where[z], self.rays[z]
        n = SECTORS[pos]
        local = (((k["th"] - self.septum - sector_centres(n) + TAU / (2 * n)) % TAU) / (TAU / n)).astype(int) % n
        return OFFSET[pos] + local


def outcome(p: Patient, v: torch.Tensor, endo: bool = True) -> list[dict]:
    """ES metrics of E edited SVF stacks v (E, Z, 2, H, W): Dice, folding, LV area, max |grad u|, endo error per segment."""
    e, z = v.shape[:2]
    u = integrate(v.flatten(0, 1))
    jd = jacobian_det(u).view(e, z, *u.shape[-2:])
    grad = torch.stack(torch.gradient(u, dim=(-2, -1)), 1).flatten(1, 2).norm(dim=1).view(e, z, *u.shape[-2:])
    pred = warp(p.onehot.repeat(e, 1, 1, 1), u).argmax(1).view(e, z, *u.shape[-2:]).cpu().numpy()
    roi = p.roi[None].expand(e, -1, -1, -1)
    rows = []
    for i in range(e):
        row = dict(**{f"dice_{k}": x for k, x in dice(pred[i], p.d["es_mask"]).items()},
                   folded_px=int((jd[i] <= 0).sum()), min_detj=float(jd[i].min()),
                   grad_max=float(grad[i][roi[i]].max()), pred_lv=int((pred[i] == 3).sum()))
        if endo:
            row["endo_err"] = endo_error(p, pred[i])
        rows.append(row)
    return rows


def lv_radius(mask: np.ndarray, k: dict) -> np.ndarray | None:
    """LV outer radius per angular bin around a fixed centre (the ED LV centroid)."""
    lv = mask == 3
    return None if lv.sum() < MIN_PX else _filled(_outer_radius(k["r"], k["th"], lv))


def endo_error(p: Patient, pred: np.ndarray) -> dict[str, np.ndarray]:
    """Per LV segment (16,), mm, mean over wall bins then slices: |predicted - true| ES endocardial radius,
    and the inward endocardial excursion ED -> ES, true and predicted."""
    total, count = np.zeros((3, 16)), np.zeros(16)
    for z, pos in p.where.items():
        k = p.rays[z]
        rp, rt, r0 = (lv_radius(m[z], k) for m in (pred, p.d["es_mask"], p.d["ed_mask"]))
        if rp is None or rt is None:
            continue
        values = np.stack([np.abs(rp - rt), r0 - rt, r0 - rp]) * MM
        sector = sector_of_bins(p.septum, SECTORS[pos])
        for s in range(SECTORS[pos]):
            bins = (sector == s) & k["has_wall"]
            if bins.any():
                total[:, OFFSET[pos] + s] += values[:, bins].mean(1)
                count[OFFSET[pos] + s] += 1
    out = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return dict(err=out[0], excursion_true=out[1], excursion_pred=out[2])


def radial(w: torch.Tensor, k: dict, radius: np.ndarray) -> torch.Tensor:
    """Radial displacement of a batch w (K, 2, H, W) along a curve, (K, bins)."""
    c, s = (torch.tensor(f(BIN_CENTRES), dtype=w.dtype, device=w.device) for f in (np.cos, np.sin))
    r = torch.tensor(radius, dtype=w.dtype, device=w.device)
    h, n = w.shape[-2:]
    x, y = 2 * (k["cx"] + r * c) / (n - 1) - 1, 2 * (k["cy"] + r * s) / (h - 1) - 1
    grid = torch.stack([x, y], -1)[None, None].expand(len(w), -1, -1, -1)
    at = F.grid_sample(w, grid, mode="bilinear", align_corners=True, padding_mode="border")[:, :, 0]
    return at[:, 0] * c + at[:, 1] * s


def segment_thickening(motion: dict, mask: np.ndarray, device) -> tuple[torch.Tensor, np.ndarray]:
    """Thickening per segment (K, 16) in mm, from SVFs motion[z] (K, 2, H, W), and which segments exist."""
    septum, total, count = septal_angle(mask), None, np.zeros(16)
    for z, pos in positions(mask).items():
        k = rays(mask[z])
        w = integrate(-motion[z])
        dt = (radial(w, k, k["epi"]) - radial(w, k, k["endo"])) * MM
        wall = torch.from_numpy(k["has_wall"]).to(device)
        total = torch.zeros(len(w), 16, device=device) if total is None else total
        sector = sector_of_bins(septum, SECTORS[pos])
        for s in range(SECTORS[pos]):
            bins = torch.from_numpy(sector == s).to(device) & wall
            if bins.any():
                seg = OFFSET[pos] + s
                total[:, seg] += dt[:, bins].mean(1)
                count[seg] += 1
    return total / torch.from_numpy(np.maximum(count, 1)).float().to(device), count > 0


def thickening(p: Patient, v: torch.Tensor) -> np.ndarray:
    """ES thickening per segment (E, 16), mm, NaN where the segment is absent."""
    dt, seen = segment_thickening({z: v[:, z] for z in p.where}, p.d["ed_mask"], p.device)
    out = dt.cpu().numpy()
    out[:, ~seen] = np.nan
    return out


def forward(v: torch.Tensor) -> torch.Tensor:
    """Forward displacement of ED points, (E, Z, 2, H, W)."""
    return integrate(-v.flatten(0, 1)).view(v.shape)


# --- locality -------------------------------------------------------------------------------

def neighbours(seg: int) -> set[int]:
    pos = RING[seg]
    n, s = SECTORS[pos], seg - OFFSET[pos]
    return {OFFSET[pos] + (s + d) % n for d in (-1, 0, 1)}


def locality(p: Patient) -> list[dict]:
    segs = [s for s in range(16) if any(RING[s] == pos for pos in p.where.values())]
    v = p.decode([p.numbers] + [edit(p.numbers, {s: 0.0}) for s in segs])
    w = forward(v)
    dt = thickening(p, v)
    labels = torch.full(p.myo.shape, -1, dtype=torch.long, device=p.device)
    for z in p.where:
        labels[z] = torch.from_numpy(p.segment_of_pixels(z)).to(p.device)
    labels = torch.where(p.myo, labels, torch.full_like(labels, -1))
    rows = []
    for i, s in enumerate(segs, 1):
        energy = ((w[i] - w[0]) ** 2).sum(1)  # (Z, H, W)
        myo = energy[p.myo].sum().item()
        near = list(neighbours(s))
        remote = [r for r in range(16) if r not in near and np.isfinite(dt[0, r])]
        rows.append(dict(pid=str(p.d["pid"]), group=str(p.d["group"]), seg=s,
                         share_segment=energy[labels == s].sum().item() / myo,
                         share_neighbourhood=energy[torch.isin(labels, torch.tensor(near, device=p.device))].sum().item() / myo,
                         share_outside_myo=energy[p.roi & ~p.myo].sum().item() / energy[p.roi].sum().item(),
                         remote_dt_change=float(np.mean(np.abs(dt[i, remote] - dt[0, remote]))) if remote else np.nan,
                         target_dt_change=float(dt[i, s] - dt[0, s])))
    return rows


# --- regional-report edits --------------------------------------------------------------------

def reports(truth: pd.DataFrame) -> dict[str, dict[int, str]]:
    """Per patient {segment: akinetic | hypokinetic} from true ES thickening; NOR thresholds from the other folds."""
    truth = truth.assign(rel=truth.dt_true / truth.t_ed)
    fold_of = {pid: f for f in range(5) for pid in fold_patients(f)}
    out = {}
    for pid, rows in truth.groupby("pid"):
        nor = truth[(truth.group == "NOR") & (truth.pid.map(fold_of) != fold_of[pid])]
        cut = nor.groupby("seg").rel.quantile(HYPO / 100)
        flags = {}
        for r in rows.itertuples():
            if not np.isfinite(r.rel):
                continue
            if r.rel < AKINETIC:
                flags[int(r.seg)] = "akinetic"
            elif r.rel < cut.get(r.seg, -np.inf):
                flags[int(r.seg)] = "hypokinetic"
        out[pid] = flags
    return out


def segment_weights(p: Patient, seg: int) -> torch.Tensor:
    """Mask-edit weight of a segment on every slice (Z, H, W): the rig's angular hat inside the ED epicardium, decaying outside."""
    w = np.zeros(p.d["ed_mask"].shape, np.float32)
    pos = RING[seg]
    n = SECTORS[pos]
    for z, where in p.where.items():
        if where != pos or (p.d["ed_mask"][z] == 3).sum() < MIN_PX:
            continue
        k = p.rays[z]
        hat = _periodic_hats(k["th"], n, p.septum + sector_centres(n))[seg - OFFSET[pos]]
        epi = np.interp(k["th"], BIN_CENTRES, k["epi"], period=TAU)
        w[z] = hat * np.exp(-np.maximum(k["r"] - epi, 0) / DECAY)
    return torch.from_numpy(w).to(p.device)


def report_edits(p: Patient, flags: dict[int, str], truth: pd.DataFrame, v_current: torch.Tensor,
                 rng: np.random.Generator) -> list[dict]:
    alpha = {s: ALPHA[f] for s, f in flags.items()}
    present = [s for s in range(16) if np.isfinite(truth.get(s, np.nan))]
    flagged = [s for s in present if s in flags]
    unflagged = [s for s in present if s not in flags]

    # Oracle: one linear step from alpha = 0 and 1 on the flagged segments to the measured thickening.
    dt = thickening(p, p.decode([p.numbers, edit(p.numbers, {s: 0.0 for s in flagged})]))
    oracle = {}
    for s in flagged:
        slope = dt[0, s] - dt[1, s]
        oracle[s] = float(np.clip((truth[s] - dt[1, s]) / slope, -1, 3)) if abs(slope) > 1e-3 else 1.0

    sets = {"rig": p.numbers, "rig_report": edit(p.numbers, alpha), "rig_oracle": edit(p.numbers, oracle)}
    draws = []
    if len(unflagged) >= len(flagged):
        for _ in range(RANDOM_DRAWS):
            segs = rng.choice(unflagged, len(flagged), replace=False)
            draws.append(edit(p.numbers, dict(zip(segs.tolist(), rng.permutation(list(alpha.values()))))))
    v = p.decode(list(sets.values()) + draws)

    factor = torch.ones_like(v_current[:, 0])
    for s, a in alpha.items():
        factor = factor * (1 - (1 - a) * segment_weights(p, s))
    v_cur = torch.stack([v_current, v_current * factor[:, None]])

    res = outcome(p, v) + outcome(p, v_cur)
    names = list(sets) + [f"rig_random_{i}" for i in range(len(draws))] + ["current", "current_mask"]
    dts = np.concatenate([thickening(p, v), thickening(p, v_cur)])
    rows, segments = [], []
    for name, r, t in zip(names, res, dts):
        endo = r.pop("endo_err")
        err = endo["err"]
        thick_err = np.abs(t - np.array([truth.get(s, np.nan) for s in range(16)]))
        if not name.startswith("rig_random_"):
            segments += [dict(pid=str(p.d["pid"]), group=str(p.d["group"]), method=name, seg=s, flag=flags.get(s, "normal"),
                              thickening_pred=t[s], thickening_true=truth[s],
                              **{k: endo[k][s] for k in ("excursion_true", "excursion_pred")}) for s in present]
        rows.append(dict(pid=str(p.d["pid"]), group=str(p.d["group"]), method=name, n_flagged=len(flagged),
                         endo_flagged=np.nanmean(err[flagged]) if flagged else np.nan,
                         endo_unflagged=np.nanmean(err[unflagged]) if unflagged else np.nan,
                         endo_all=np.nanmean(err[present]),
                         thick_err_flagged=np.nanmean(thick_err[flagged]) if flagged else np.nan,
                         ed_lv=int((p.d["ed_mask"] == 3).sum()), gt_lv=int((p.d["es_mask"] == 3).sum()), **r))
    return rows, segments


# --- dose-response -----------------------------------------------------------------------------

def dose(p: Patient) -> list[dict]:
    segs = [s for s in range(16) if any(RING[s] == pos for pos in p.where.values())]
    combos = [(s, a) for s in segs for a in DOSE]
    rows = []
    for chunk in range(0, len(combos), 28):
        part = combos[chunk:chunk + 28]
        v = p.decode([edit(p.numbers, {s: a}) for s, a in part])
        dt = thickening(p, v)
        for (s, a), r, t in zip(part, outcome(p, v, endo=False), dt):
            rows.append(dict(pid=str(p.d["pid"]), group=str(p.d["group"]), seg=s, alpha=a, target_dt=t[s],
                             ed_lv=int((p.d["ed_mask"] == 3).sum()), **{k: r[k] for k in ("pred_lv", "folded_px", "min_detj")}))
    return rows


# --- summary -----------------------------------------------------------------------------------

def ci(x: np.ndarray, n_boot: int = 10000) -> str:
    x = x[np.isfinite(x)]
    boots = np.random.default_rng(0).choice(x, (n_boot, len(x))).mean(1)
    return f"{x.mean():.4f} [{np.percentile(boots, 2.5):.4f}, {np.percentile(boots, 97.5):.4f}] (n={len(x)})"


def summarise(loc: pd.DataFrame, rep: pd.DataFrame, seg: pd.DataFrame, dos: pd.DataFrame) -> None:
    print("\n=== Locality (alpha = 0, every patient-segment) ===")
    per_patient = loc.groupby("pid")[["share_segment", "share_neighbourhood", "share_outside_myo", "remote_dt_change"]].mean()
    for col in per_patient:
        print(f"{col}: {ci(per_patient[col].to_numpy())}")
    print(f"min share_neighbourhood over patient-segments: {loc.share_neighbourhood.min():.4f}")

    print("\n=== Regional-report edits ===")
    rep = rep.assign(dice=rep[["dice_RV", "dice_MYO", "dice_LV"]].mean(axis=1))
    rand = rep[rep.method.str.startswith("rig_random_")]
    rand_mean = rand.groupby(["pid", "group"], as_index=False).mean(numeric_only=True).assign(method="rig_random")
    tab = pd.concat([rep[~rep.method.str.startswith("rig_random_")], rand_mean])
    flagged = tab[tab.n_flagged > 0]
    cols = ["endo_flagged", "endo_unflagged", "endo_all", "thick_err_flagged", "dice", "dice_MYO", "dice_LV", "folded_px", "grad_max"]
    print(f"patients with >= 1 flagged segment: {flagged.pid.nunique()} of {tab.pid.nunique()}")
    print(flagged.groupby("method")[cols].mean().round(4).to_string())
    pooled = flagged.groupby("method")[["ed_lv", "gt_lv", "pred_lv"]].sum()
    print("LV contraction:", (1 - pooled.pred_lv / pooled.ed_lv).round(3).to_dict(),
          "true", round(1 - pooled.gt_lv.iloc[0] / pooled.ed_lv.iloc[0], 3))
    print(flagged.pivot_table(index="group", columns="method", values="endo_flagged").round(3).to_string())
    print(flagged.groupby("group").n_flagged.mean().round(1).to_dict())
    for metric in ("endo_flagged", "dice", "endo_unflagged", "thick_err_flagged"):
        wide = flagged.pivot(index="pid", columns="method", values=metric).dropna(subset=["rig", "rig_report"])
        print(f"[{metric}]", paired(wide, "rig_report", "rig"))
    for metric in ("endo_all", "dice"):
        wide = flagged.pivot(index="pid", columns="method", values=metric).dropna(subset=["rig_random"])
        print(f"[{metric}, location]", paired(wide, "rig_report", "rig_random"))
    for metric in ("endo_flagged", "dice"):
        wide = flagged.pivot(index="pid", columns="method", values=metric)
        print(f"[{metric}, mask edit gain]", paired(wide.assign(gain_rig=wide.rig_report - wide.rig,
                                                               gain_mask=wide.current_mask - wide.current), "gain_rig", "gain_mask"))
        print(f"[{metric}, edited rig vs edited current]", paired(wide, "rig_report", "current_mask"))
    print("Inward endocardial excursion at ES by report flag (mm, mean over patient-segments):")
    print(seg.groupby(["flag", "method"])[["excursion_true", "excursion_pred", "thickening_true", "thickening_pred"]]
          .mean().round(2).unstack("method").stack(0, future_stack=True).to_string())

    print("\n=== Dose-response ===")
    ok = dos.dropna(subset=["target_dt"])
    mono, mono_low, positive, slopes, r2 = [], [], [], [], []
    for _, g in ok.groupby(["pid", "seg"]):
        g = g.sort_values("alpha")
        mono.append(np.all(np.diff(g.target_dt.to_numpy()) > 0))
        mono_low.append(np.all(np.diff(g[g.alpha <= 1].target_dt.to_numpy()) > 0))
        positive.append(g[g.alpha == 1].target_dt.iloc[0] > 0)
        fit = np.polyfit(g.alpha, g.target_dt, 1)
        resid = g.target_dt - np.polyval(fit, g.alpha)
        slopes.append(fit[0])
        r2.append(1 - resid.var() / g.target_dt.var() if g.target_dt.var() > 0 else np.nan)
    print(f"monotone increasing: {np.mean(mono):.4f} of {len(mono)} patient-segments; slope {np.median(slopes):.3f} mm per unit alpha; "
          f"median R2 {np.nanmedian(r2):.4f}")
    mono, mono_low, positive = map(np.array, (mono, mono_low, positive))
    print(f"increasing over alpha <= 1: {mono_low.mean():.4f}; "
          f"unedited thickening > 0: {positive.mean():.4f}, of these increasing: {mono[positive].mean():.4f}, "
          f"over alpha <= 1: {mono_low[positive].mean():.4f}")
    by_alpha = dos.groupby("alpha").agg(folded_px=("folded_px", "sum"), any_folded=("folded_px", lambda x: (x > 0).mean()),
                                        min_detj=("min_detj", "min"))
    lv = dos.groupby("alpha")[["pred_lv", "ed_lv"]].sum()
    by_alpha["lv_area_change"] = (lv.pred_lv / lv.ed_lv - 1).round(4)
    print(by_alpha.to_string())


def main(args):
    torch.manual_seed(0)
    pred = np.load(args.predictions)
    numbers = dict(zip(pred["pids"].tolist(), pred["numbers"]))
    truth = pd.read_csv(args.truth)
    flags = reports(truth)
    truth_by = {pid: dict(zip(g.seg.astype(int), g.dt_true)) for pid, g in truth.groupby("pid")}
    rng = np.random.default_rng(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    loc, rep, seg, dos = [], [], [], []
    start = time.time()
    for fold in range(5):
        current, sigma = load_current(Path(args.runs) / f"current_f{fold}" / "last.pt", args.device)
        patients = load(args.cache, "val", svf=False, features=args.features, fold=fold)
        if args.limit:
            patients = patients[:args.limit]
        for d in patients:
            pid = str(d["pid"])
            with torch.no_grad():
                p = Patient(d, numbers[pid], args.device)
                ed, tau, cond = es_inputs(d, args.device)
                v_current = predict(current, ed, tau, sigma, **cond).float()
                loc += locality(p)
                rows, segments = report_edits(p, flags[pid], truth_by[pid], v_current, rng)
                rep, seg = rep + rows, seg + segments
                dos += dose(p)
        print(f"fold {fold} done, {time.time() - start:.0f} s", flush=True)
    loc, rep, seg, dos = pd.DataFrame(loc), pd.DataFrame(rep), pd.DataFrame(seg), pd.DataFrame(dos)
    loc.to_csv(out / "locality.csv", index=False)
    rep.to_csv(out / "reports.csv", index=False)
    seg.to_csv(out / "report_segments.csv", index=False)
    dos.to_csv(out / "dose.csv", index=False)
    pd.DataFrame([dict(pid=pid, seg=s, flag=f) for pid, fl in flags.items() for s, f in fl.items()]).to_csv(
        out / "report_flags.csv", index=False)
    summarise(loc, rep, seg, dos)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", default="outputs/cv/rig_predictions.npz")
    p.add_argument("--truth", default="outputs/rig/segments.csv")
    p.add_argument("--runs", default="checkpoints/cv")
    p.add_argument("--cache", default="data/processed/acdc")
    p.add_argument("--features", default="data/processed/acdc_cinema")
    p.add_argument("--out", default="outputs/edits")
    p.add_argument("--limit", type=int, default=0, help="patients per fold, for a quick check")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(p.parse_args())
