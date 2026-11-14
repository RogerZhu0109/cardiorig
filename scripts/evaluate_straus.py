"""
STRAUS: the final ACDC-trained models against simulated true motion. Nothing is trained. Three parts: accuracy of every
method, the rig fitted to the true motion, and segment edits against the matched infarct truth.
Writes accuracy.csv (one row per sequence, method and seed), frames.csv (per frame), edits.csv and
edit_reports.csv, and prints the summaries.
Usage: python scripts/evaluate_straus.py --runs checkpoints/final --cache data/processed/straus --features data/processed/straus/cinema
       python scripts/evaluate_straus.py --report outputs/straus   (re-print from the CSVs)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_cv import load_current, load_rig, predict_rig  # noqa: E402
from scripts.evaluate_published import krebs_fields, load_phaseflow, phaseflow_fields  # noqa: E402
from src.data.cache import load  # noqa: E402
from src.models.unet import predict  # noqa: E402
from src.models.krebs import Krebs  # noqa: E402
from src.rig import (OFFSET, SECTORS, TAU, decode, edit, fit_compact, patient_fields, positions, rays,  # noqa: E402
                     sector_centres, septal_angle, time_basis)
from src.utils.metrics import STRUCTURES, dice  # noqa: E402
from src.utils.svf import integrate, jacobian_det, warp  # noqa: E402

MM = 1.5
ACDC_GROUP = dict(healthy="NOR", lbbb="NOR", lad="MINF", lcx="MINF", rca="MINF", rca2="MINF")  # PhaseFlow's label
INFARCTS = ("lad", "lcx", "rca", "rca2")
FLAGGED, ALPHA_RANGE, MIN_HEALTHY = 0.75, (-0.5, 1.5), 1.0
RING = {s: pos for pos in SECTORS for s in range(OFFSET[pos], OFFSET[pos] + SECTORS[pos])}
REFERENCES = ("identity", "teacher", "ceiling", "rig_fit_truth")
MODELS = ("rig", "current", "phaseflow_tau0", "krebs_mean")


def load_straus(cache: str, features: str) -> list[dict]:
    patients = load(cache, "testing", features=features)
    for d in patients:
        d["pid"], d["group"], d["ed"], d["es"] = str(d["pid"]), str(d["group"]), int(d["ed"]), int(d["es"])
        with np.load(Path(cache) / "testing" / f"{d['pid']}.npz") as f:
            d.update(masks=f["masks"], truth=f["truth"].astype(np.float32), aha=f["aha"], anatomy=int(f["anatomy"]))
        d["svf"] = d["svf"].astype(np.float32)
    return patients


def geometry(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per pixel of an ED stack: the outward radial unit vector around the slice's LV centre (Z, 2, H, W)
    and the rig's LV segment of myocardial pixels (Z, H, W), -1 elsewhere."""
    septum, er, seg = septal_angle(mask), np.zeros((len(mask), 2, *mask.shape[1:]), np.float32), np.full(mask.shape, -1)
    for z, pos in positions(mask).items():
        th, n = rays(mask[z])["th"], SECTORS[pos]
        er[z] = np.stack([np.cos(th), np.sin(th)])
        local = (((th - septum - sector_centres(n) + TAU / (2 * n)) % TAU) / (TAU / n)).astype(int) % n
        seg[z] = np.where(mask[z] == 2, OFFSET[pos] + local, -1)
    return er, seg


def segment_radial(w: np.ndarray, er: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Inward radial motion (mm) of the displacement w (Z, 2, H, W) per rig LV segment (16,), NaN if absent."""
    inward = -(w * er).sum(1) * MM
    return np.array([inward[seg == k].mean() if (seg == k).any() else np.nan for k in range(16)])


def centred(w: torch.Tensor, myo: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """w (Z, 2, H, W) minus its mean over the myocardium of each slice, and that mean (Z, 2): the slice's shift."""
    shift = (w * myo[:, None]).sum((2, 3)) / myo.sum((1, 2)).clamp_min(1)[:, None]
    return w - shift[:, :, None, None], shift


def forward(v: torch.Tensor) -> torch.Tensor:
    """Motion of ED points under the SVF stack v (Z, 2, H, W)."""
    return integrate(-v)


@torch.no_grad()
def score_frames(d: dict, v, er: np.ndarray, seg: np.ndarray, device) -> tuple[dict, pd.DataFrame]:
    """Metrics of SVFs v (T, Z, 2, H, W) against the truth: one summary row and per-frame EPE (mm) and Dice."""
    onehot = F.one_hot(torch.from_numpy(d["ed_mask"]).long(), 4).permute(0, 3, 1, 2).float().to(device)
    myo, es, T = torch.from_numpy(d["ed_mask"] == 2).to(device), d["es"], len(d["tau"])
    epe, dsc, folded, sse, sst = np.zeros(T), np.full((T, 3), np.nan), np.zeros(T), 0.0, 0.0
    shape, has_myo = np.zeros(T), myo.any(2).any(1)  # error with each slice's shift removed
    for t in range(1, T):
        vt, truth = torch.as_tensor(v[t]).float().to(device), torch.from_numpy(d["truth"][t]).to(device)
        u, w = integrate(vt), forward(vt)
        err = w - truth
        epe[t] = err.norm(dim=1)[myo].mean().item() * MM
        (w0, shift), (truth0, shift_true) = centred(w, myo), centred(truth, myo)
        shape[t] = (w0 - truth0).norm(dim=1)[myo].mean().item() * MM
        sse += (err ** 2).sum(1)[myo].sum().item()
        sst += (truth ** 2).sum(1)[myo].sum().item()
        dsc[t] = list(dice(warp(onehot, u).argmax(1).cpu().numpy(), d["masks"][t]).values())
        folded[t] = 100 * (jacobian_det(u) <= 0).float().mean().item()
        if t == es:
            radial = np.nanmean(segment_radial(w.cpu().numpy(), er, seg))
            shifts = [x[has_myo].mean().item() * MM for x in (shift.norm(dim=1), shift_true.norm(dim=1), (shift - shift_true).norm(dim=1))]
    per_frame = np.array([np.nanmean(x) if np.isfinite(x).any() else np.nan for x in dsc])
    row = dict(epe_es=epe[es], epe_all=epe[1:].mean(), dice_es=per_frame[es], dice_all=np.nanmean(per_frame[1:]),
               **{f"dice_{k}": x for k, x in zip(STRUCTURES.values(), dsc[es])}, energy=1 - sse / sst,
               radial=radial, radial_true=np.nanmean(segment_radial(d["truth"][es], er, seg)), folded_pct=folded.max(),
               epe_shape_es=shape[es], epe_shape_all=shape[1:].mean(), shift=shifts[0], shift_true=shifts[1], shift_err=shifts[2])
    return row, pd.DataFrame(dict(frame=np.arange(1, T), epe=epe[1:], dice=per_frame[1:]))


def fit_truth(d: dict, fields, slots, start: np.ndarray, device, steps: int = 500, lr: float = 0.05) -> np.ndarray:
    """Compact rig numbers whose motion of ED points best matches the true motion on the walls, all non-ED frames."""
    basis = [torch.from_numpy(b).float().to(device) for b in fields]
    index = [torch.from_numpy(i).to(device) for i in slots]
    phi = torch.from_numpy(time_basis(d["tau"][1:], start.shape[1] // 2)).float().to(device)  # (F, curves)
    truth = torch.from_numpy(d["truth"][1:]).to(device)
    wall = torch.from_numpy(d["aha"] > 0).to(device)
    numbers = torch.tensor(start, dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([numbers], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        c = numbers @ phi.T  # (SHARED, F)
        v = torch.stack([torch.einsum("pf,pchw->fchw", c[i], b) for b, i in zip(basis, index)], 1)  # (F, Z, 2, H, W)
        loss = ((integrate(-v.flatten(0, 1)).view_as(v) - truth) ** 2).sum(2)[:, wall].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return numbers.detach().cpu().numpy()


def decoded(numbers: np.ndarray, d: dict, fields, slots) -> np.ndarray:
    """SVFs (T, Z, 2, H, W) of compact numbers at every frame; frame 0 (ED) is zero by construction."""
    return np.stack([decode(numbers, float(t), fields, slots) for t in d["tau"]]).astype(np.float32)


def opposite(k: int) -> int:
    """The segment across the cavity in the same ring."""
    n, first = SECTORS[RING[k]], OFFSET[RING[k]]
    return first + (k - first + n // 2) % n


@torch.no_grad()
def edit_scores(numbers: np.ndarray, target: dict, healthy: dict, fields, slots, seg: np.ndarray, flagged, device) -> dict:
    """An edited rig motion against the infarct truth at the healthy ES frame and over all frames."""
    v = torch.from_numpy(decoded(numbers, healthy, fields, slots)).to(device)
    myo = torch.from_numpy(seg >= 0).to(device)
    hit = torch.from_numpy(np.isin(seg, flagged)).to(device)
    err = torch.stack([(forward(v[t]) - torch.from_numpy(target["truth"][t]).to(device)).norm(dim=1) * MM
                       for t in range(1, len(v))])  # (F, Z, H, W)
    t = healthy["es"] - 1
    return dict(epe_flagged=err[t][hit].mean().item(), epe_lv=err[t][myo].mean().item(),
                epe_flagged_all=err[:, hit].mean().item(), epe_lv_all=err[:, myo].mean().item(),
                folded_px=int((jacobian_det(integrate(v[t + 1])) <= 0).sum()))


def evaluate(args) -> dict[str, pd.DataFrame]:
    device, runs = args.device, Path(args.runs)
    patients = load_straus(args.cache, args.features)
    rows, frames, fits, predicted, geom, rig_fields = [], [], {}, {}, {}, {}

    def add(d, method, seed, v):
        row, per_frame = score_frames(d, v, *geom[d["pid"]], device)
        base = dict(pid=d["pid"], anatomy=d["anatomy"], condition=d["group"], method=method, seed=seed)
        rows.append(dict(**base, **row))
        frames.append(per_frame.assign(**base))

    for d in patients:  # references: one value per sequence
        geom[d["pid"]] = geometry(d["ed_mask"])
        fields, slots, _ = rig_fields[d["pid"]] = patient_fields(d["ed_mask"])
        ceiling = fit_compact(d)
        fits[d["pid"]] = fit_truth(d, fields, slots, ceiling, device)
        for name, v in (("identity", np.zeros_like(d["svf"])), ("teacher", d["svf"]),
                        ("ceiling", decoded(ceiling, d, fields, slots)), ("rig_fit_truth", decoded(fits[d["pid"]], d, fields, slots))):
            add(d, name, -1, v)
        print(f"references {d['pid']}: fit to truth explains {rows[-1]['energy']:.3f} of the motion energy", flush=True)

    for seed in args.seeds:
        pf = load_phaseflow(runs / f"phaseflow_s{seed}", device)
        kr = Krebs().to(device).eval()
        kr.load_state_dict(torch.load(runs / f"krebs_s{seed}" / "last.pt", map_location=device, weights_only=False)["model"])
        current, sigma = load_current(runs / f"current_s{seed}" / "last.pt", device)
        rig, mean, std = load_rig(runs / f"rig_s{seed}" / "last.pt", device)
        gen = torch.Generator().manual_seed(seed)
        for d in patients:
            fields, slots, _ = rig_fields[d["pid"]]
            numbers, _ = predict_rig(rig, mean, std, d, device, fields, slots)
            predicted[d["pid"], seed] = numbers
            ed = torch.from_numpy(d["cine"][0][:, None].astype(np.float32)).to(device)
            feat = torch.from_numpy(d["feat"].astype(np.float32)).to(device)
            cls = torch.from_numpy(d["cls"].astype(np.float32)).to(device)[None].expand(len(ed), -1)
            with torch.no_grad():
                v_current = torch.stack([predict(current, ed, torch.full((len(ed),), float(t), device=device), sigma,
                                                feat=feat, cls=cls) for t in d["tau"]])
                v_pf = phaseflow_fields(pf, dict(d, group=ACDC_GROUP[d["group"]]), 0.0, device, gen)
                v_kr = krebs_fields(kr, d, None, device)
            assert len(v_pf) == len(v_kr) == len(d["tau"])  # 30 frames: no resampling of the sequence
            for name, v in (("rig", decoded(numbers, d, fields, slots)), ("current", v_current),
                            ("phaseflow_tau0", v_pf), ("krebs_mean", v_kr)):
                add(d, name, seed, v)
        print(f"seed {seed} done", flush=True)

    # the error an ED-only model cannot avoid: the healthy true motion of the same anatomy as the prediction
    by = {(d["anatomy"], d["group"]): d for d in patients}
    for d in patients:
        h = by[d["anatomy"], "healthy"]
        assert np.array_equal(h["ed_mask"], d["ed_mask"])
        myo = d["ed_mask"] == 2
        gap = np.linalg.norm(h["truth"] - d["truth"], axis=2)[:, myo] * MM  # (T, pixels)
        rows.append(dict(pid=d["pid"], anatomy=d["anatomy"], condition=d["group"], method="healthy_truth", seed=-1,
                         epe_es=gap[d["es"]].mean(), epe_all=gap[1:].mean()))

    edits, reports = [], []
    for anatomy in sorted({d["anatomy"] for d in patients}):
        h = by[anatomy, "healthy"]
        (er, seg), (fields, slots, _), t = geom[h["pid"]], rig_fields[h["pid"]], h["es"]
        r_h = segment_radial(h["truth"][t], er, seg)
        for c in INFARCTS:
            d = by[anatomy, c]
            r_c = segment_radial(d["truth"][t], er, seg)
            alpha = np.where(r_h >= MIN_HEALTHY, np.clip(r_c / np.where(r_h >= MIN_HEALTHY, r_h, 1), *ALPHA_RANGE), 1.0)
            flagged = [k for k in range(16) if alpha[k] <= FLAGGED]
            reports += [dict(anatomy=anatomy, condition=c, segment=k, radial_healthy=r_h[k], radial_condition=r_c[k],
                             alpha=alpha[k], flagged=k in flagged) for k in range(16)]
            if not flagged:
                continue
            kinds = dict(none={}, report={k: alpha[k] for k in flagged}, misplaced={opposite(k): alpha[k] for k in flagged})
            kinds["global"] = dict.fromkeys(range(16), float(np.mean([alpha[k] if k in flagged else 1.0 for k in range(16)])))
            bases = [("truth", -1, fits[h["pid"]])] + [("model", s, predicted[h["pid"], s]) for s in args.seeds]
            for base, seed, numbers in bases:
                for kind, change in kinds.items():
                    edits.append(dict(anatomy=anatomy, condition=c, base=base, seed=seed, edit=kind, n_flagged=len(flagged),
                                      **edit_scores(edit(numbers, change), d, h, fields, slots, seg, flagged, device)))
            edits.append(dict(anatomy=anatomy, condition=c, base="fit to infarct truth", seed=-1, edit="none", n_flagged=len(flagged),
                              **edit_scores(fits[d["pid"]], d, h, fields, slots, seg, flagged, device)))
    return dict(accuracy=pd.DataFrame(rows), frames=pd.concat(frames), edits=pd.DataFrame(edits), edit_reports=pd.DataFrame(reports))


def consistent(wide: pd.DataFrame, a: str, b: str) -> str:
    """The 3/3 rule on a per-anatomy table (lower is better)."""
    diff = wide[a] - wide[b]
    verdict = f"{a} better in 3/3" if (diff < 0).all() else f"{b} better in 3/3" if (diff > 0).all() else "no consistent difference"
    return f"{a} - {b}: {diff.mean():+.2f} mm (per anatomy {', '.join(f'{x:+.2f}' for x in diff)}) -> {verdict}"


def report(t: dict[str, pd.DataFrame]) -> None:
    pd.set_option("display.width", 250)
    acc = t["accuracy"].groupby(["pid", "anatomy", "condition", "method"], sort=False).mean(numeric_only=True).reset_index()
    acc["kind"] = acc.condition.map(lambda c: "infarct" if c in INFARCTS else c)
    order = [m for m in (*REFERENCES, *MODELS, "healthy_truth") if m in set(acc.method)]
    teacher, still = acc[acc.method == "teacher"].set_index("pid"), acc[acc.method == "identity"].set_index("pid")
    ok = (teacher.epe_es < still.epe_es) & (teacher.dice_es > still.dice_es)
    print(f"GATE teacher beats no motion on ES EPE and Dice: {'PASS' if ok.all() else 'FAIL'} ({int(ok.sum())}/{len(ok)} sequences)")

    print("\n== 1. Accuracy against true motion (means over sequences; models = mean of 5 seeds) ==")
    for kind in ("healthy", "infarct", "lbbb"):
        cols = ["epe_es", "epe_all", "dice_es", "dice_all", "dice_RV", "dice_MYO", "dice_LV", "energy", "radial", "radial_true", "folded_pct"]
        print(f"-- {kind} ({acc[acc.kind == kind].pid.nunique()} sequences)")
        print(acc[acc.kind == kind].groupby("method")[cols].mean().reindex(order).round(3).to_string())
    healthy = acc[acc.condition == "healthy"].pivot(index="anatomy", columns="method", values="epe_es")
    print("-- healthy, ES EPE (mm) per anatomy\n" + healthy[[m for m in order if m in healthy]].round(2).to_string())
    for m in MODELS:
        print("skill     " + consistent(healthy, m, "identity"))
    for a, b in (("rig", "current"), ("rig", "phaseflow_tau0"), ("rig", "krebs_mean"), *(("teacher", m) for m in MODELS)):
        print("ordering  " + consistent(healthy, a, b))
    pooled = acc.groupby(["kind", "method"])[["radial", "radial_true"]].sum()
    ratio = (pooled.radial / pooled.radial_true).unstack(0)
    print("-- inward radial motion at ES, predicted / true\n" + ratio.reindex(order).round(3).to_string())

    if "epe_shape_es" in acc:
        print("\n== Error with each slice's whole-slice shift removed ==")
        cols = ["epe_es", "epe_shape_es", "epe_shape_all", "shift_true", "shift", "shift_err"]
        for kind in ("healthy", "infarct", "lbbb"):
            print(f"-- {kind}\n" + acc[acc.kind == kind].groupby("method")[cols].mean().reindex(order[:-1]).round(2).to_string())
        shape = acc[acc.condition == "healthy"].pivot(index="anatomy", columns="method", values="epe_shape_es")
        print("-- healthy, shift-removed ES EPE (mm) per anatomy\n" + shape[order[:-1]].round(2).to_string())
        for m in MODELS:
            print("skill     " + consistent(shape, m, "identity"))
        for a, b in (("rig", "current"), ("rig", "phaseflow_tau0"), ("rig", "krebs_mean"), *(("teacher", m) for m in MODELS)):
            print("ordering  " + consistent(shape, a, b))

    print("\n== 2. Rig fitted to the true motion ==")
    fit = acc[acc.method == "rig_fit_truth"]
    print(fit[["pid", "energy", "epe_es", "epe_all", "dice_es"]].round(3).to_string(index=False))
    h = fit[fit.condition == "healthy"]
    print(f"healthy: energy explained min {h.energy.min():.3f} (bar 0.80), ES EPE max {h.epe_es.max():.2f} mm (bar 1.5) -> "
          f"{'PASS' if (h.energy >= 0.80).all() and (h.epe_es <= 1.5).all() else 'FAIL'}")

    print("\n== 3. Segment edits against the matched infarct truth ==")
    rep, e = t["edit_reports"], t["edits"]
    flagged = rep[rep.flagged].groupby(["anatomy", "condition"]).agg(segments=("segment", list), alpha=("alpha", lambda a: [round(x, 2) for x in a]))
    print(flagged.to_string())
    e = e.groupby(["anatomy", "condition", "base", "edit"], sort=False).mean(numeric_only=True).reset_index()
    print(f"pairs with a flagged segment: {e.groupby(['anatomy', 'condition']).ngroups} of {3 * len(INFARCTS)}")
    print(e.groupby(["base", "edit"], sort=False)[["epe_flagged", "epe_lv", "epe_flagged_all", "epe_lv_all", "folded_px"]].mean().round(3).to_string())
    for base in ("truth", "model"):
        w = e[e.base == base].pivot(index=["anatomy", "condition"], columns="edit", values="epe_flagged")
        lv = e[e.base == base].pivot(index=["anatomy", "condition"], columns="edit", values="epe_lv")
        print(f"-- base {base}: flagged EPE (mm) per pair\n" + w.round(2).to_string())
        checks = {"report lowers flagged EPE in >= 83% of pairs": (w.report < w.none).mean() >= 0.83,
                  "mean flagged EPE falls by >= 25%": 1 - w.report.mean() / w.none.mean() >= 0.25,
                  "report beats misplaced in >= 83% of pairs": (w.report < w.misplaced).mean() >= 0.83,
                  "mean LV EPE not higher": lv.report.mean() <= lv.none.mean(),
                  "no folded pixels": e[(e.base == base) & (e.edit == "report")].folded_px.sum() == 0}
        print(f"   report vs none: lower in {(w.report < w.none).sum()}/{len(w)}, mean change {100 * (w.report.mean() / w.none.mean() - 1):+.0f}%; "
              f"vs misplaced: lower in {(w.report < w.misplaced).sum()}/{len(w)}; LV EPE {lv.none.mean():.2f} -> {lv.report.mean():.2f} mm")
        for name, passed in checks.items():
            print(f"   {'ok  ' if passed else 'FAIL'} {name}")
        print(f"   base {base}: {'PASS' if all(checks.values()) else 'FAIL'}")


def main(args):
    names = ("accuracy", "frames", "edits", "edit_reports")
    if args.report:
        return report({n: pd.read_csv(Path(args.report) / f"{n}.csv") for n in names})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = evaluate(args)
    for n in names:
        tables[n].to_csv(out / f"{n}.csv", index=False)
    report(tables)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", help="folder with {rig,current,phaseflow,krebs}_s{seed}")
    p.add_argument("--cache", default="data/processed/straus")
    p.add_argument("--features", default="data/processed/straus/cinema")
    p.add_argument("--out", default="outputs/straus")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--report", help="re-print the summaries from the CSVs in this folder")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(p.parse_args())
