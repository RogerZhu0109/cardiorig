"""
ED -> ES evaluation of one patient's slice stack. Only the ED slices, ED labels and tau_ES are
inputs; ES labels are used for scoring only.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.utils.metrics import dice, lv_areas
from src.utils.svf import integrate, jacobian_det, warp


def score(d: dict, v: torch.Tensor) -> dict:
    """ES metrics of the SVF stack v (Z, 2, S, S) applied to the ED labels."""
    u = integrate(v)
    onehot = F.one_hot(torch.from_numpy(d["ed_mask"]).long(), 4).permute(0, 3, 1, 2).float().to(v.device)
    pred = warp(onehot, u).argmax(1).cpu().numpy()
    jd = jacobian_det(u)
    return dict(**{f"dice_{k}": x for k, x in dice(pred, d["es_mask"]).items()},
                folded_pct=100 * (jd <= 0).float().mean().item(), **lv_areas(d, pred))


def es_inputs(d: dict, device) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """ED slices, tau_ES, and the CineMA conditioning (empty unless the patient has features)."""
    ed = torch.from_numpy(d["cine"][d["ed"]][:, None].astype("float32")).to(device)
    tau = torch.full((len(ed),), float(d["tau"][d["es"]]), device=device)
    cond = {}
    if "feat" in d:
        cond = dict(feat=torch.from_numpy(d["feat"].astype("float32")).to(device),
                    cls=torch.from_numpy(d["cls"].astype("float32")).to(device)[None].expand(len(ed), -1))
    return ed, tau, cond


def endpoint_error(v: torch.Tensor, truth: torch.Tensor, region: torch.Tensor, mm: float = 1.5) -> float:
    """Mean in-plane distance (mm) over region between the motion exp(-v) of ED pixels and their true motion (px)."""
    return ((integrate(-v) - truth).norm(dim=1)[region].mean() * mm).item()
