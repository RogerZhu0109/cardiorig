"""
Anatomical rig for 2D short-axis slices: LV-centred coordinates from an ED mask and a fixed
linear decoder from physiological parameters to an SVF. Conventions follow src/utils/svf.py
(pixel units, ch0 = x): positive endo/epi radial parameters sample outward, i.e. contraction.
Mask labels: 1 RV, 2 MYO, 3 LV. 28 parameters per six-sector slice-frame.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, map_coordinates

TAU = 2 * np.pi
MIN_PX = 20  # below this a slice has no usable LV / RV
BINS = 72  # angular resolution of the boundary radii
DECAY = 6.0  # px, fall-off outside the structure
ROI_DILATION = 3  # px
SECTORS = {"basal": 6, "mid": 6, "apical": 4}
OFFSET = {"basal": 0, "mid": 6, "apical": 12}  # first segment of each ring
RV_SECTORS = 4
SHARED = 2 + 3 * 16 + 2 * RV_SECTORS  # patient-level motions: translation, LV segments, RV sectors
TOKENS = 16 + RV_SECTORS + 1  # LV segments, RV sectors, translation
BIN_CENTRES = (np.arange(BINS) + 0.5) * TAU / BINS


def _centroid(m: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(m)
    return xs.mean(), ys.mean()


def septal_angle(stack: np.ndarray) -> float:
    """Circular mean LV -> RV direction over a stack (Z, H, W); the zero of the sector angle."""
    z = 0j
    for m in stack:
        if (m == 3).sum() >= MIN_PX and (m == 1).sum() >= MIN_PX:
            (lx, ly), (rx, ry) = _centroid(m == 3), _centroid(m == 1)
            z += np.exp(1j * np.arctan2(ry - ly, rx - lx))
    return float(np.angle(z)) % TAU


def _polar(shape: tuple[int, int], cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[:shape[0], :shape[1]]
    dx, dy = xs - cx, ys - cy
    return np.hypot(dx, dy), np.arctan2(dy, dx) % TAU


def _outer_radius(r: np.ndarray, th: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Largest r of region per angular bin, NaN where the region is absent."""
    out = np.full(BINS, np.nan)
    np.fmax.at(out, (th[region] / TAU * BINS).astype(int) % BINS, r[region])
    return out


def _filled(radius: np.ndarray) -> np.ndarray:
    """Circularly interpolate missing bins and smooth over five."""
    ok = np.isfinite(radius)
    full = np.interp(BIN_CENTRES, BIN_CENTRES[ok], radius[ok], period=TAU)
    return np.convolve(np.r_[full[-2:], full, full[:2]], np.ones(5) / 5, "valid")


def rays(mask: np.ndarray) -> dict:
    """LV centre and endo/epi radius per angular bin; has_wall marks bins with real myocardium."""
    lv, myo = mask == 3, mask == 2
    cx, cy = _centroid(lv)
    r, th = _polar(mask.shape, cx, cy)
    endo = _filled(_outer_radius(r, th, lv))
    epi = np.maximum(_filled(_outer_radius(r, th, lv | myo)), endo + 1)
    return dict(cx=cx, cy=cy, r=r, th=th, endo=endo, epi=epi, has_wall=np.isfinite(_outer_radius(r, th, myo)))


def _periodic_hats(phi: np.ndarray, n: int, first: float) -> np.ndarray:
    """Linear partition of unity on n equally spaced periodic nodes, the first at angle `first`."""
    x = ((phi - first) % TAU) / (TAU / n)
    i, f = np.floor(x).astype(int), x - np.floor(x)
    h = np.zeros((n, *phi.shape))
    np.put_along_axis(h, (i % n)[None], (1 - f)[None], 0)
    np.add.at(h, ((i + 1) % n, *np.indices(phi.shape)), f)
    return h


def sector_centres(n: int) -> float:
    """First sector centre relative to the septal direction: AHA 6 split at mid-septum, AHA 4 centred on it."""
    return TAU / 12 if n == 6 else 0.0


def sector_of_bins(septum: float, n: int) -> np.ndarray:
    """Sector index of every angular bin."""
    half = TAU / (2 * n)
    return (((BIN_CENTRES - septum - sector_centres(n) + half) % TAU) / (TAU / n)).astype(int)


def basis(mask: np.ndarray, septum: float, n_sectors: int) -> np.ndarray:
    """Decoder fields (P, 2, H, W): SVF = sum_i c_i B_i. P = 2 without a usable LV."""
    outside = np.exp(-distance_transform_edt(mask == 0) / DECAY)
    zero = np.zeros_like(outside)
    fields = [np.stack([outside, zero]), np.stack([zero, outside])]
    if (mask == 3).sum() < MIN_PX:
        return np.stack(fields)

    k = rays(mask)
    r, th = k["r"], k["th"]
    endo = np.interp(th, BIN_CENTRES, k["endo"], period=TAU)
    epi = np.interp(th, BIN_CENTRES, k["epi"], period=TAU)
    e_r, e_t = np.stack([np.cos(th), np.sin(th)]), np.stack([-np.sin(th), np.cos(th)])

    depth = np.clip((r - endo) / (epi - endo), 0, 1)
    cavity = np.clip(r / endo, 0, 1)
    decay = np.exp(-np.maximum(r - epi, 0) / DECAY)
    hats = _periodic_hats(th, n_sectors, septum + sector_centres(n_sectors))
    for h in hats:
        fields += [h * cavity * (1 - depth) * e_r, h * depth * decay * e_r, h * cavity * decay * e_t]

    rv = mask == 1
    if rv.sum() >= MIN_PX:
        rx, ry = _centroid(rv)
        centre = np.arctan2(ry - k["cy"], rx - k["cx"])
        phi = (th - centre + np.pi) % TAU - np.pi
        lo, hi = phi[rv].min(), phi[rv].max()
        x = np.clip((phi - lo) / (hi - lo), 0, 1) * (RV_SECTORS - 1)
        i = np.minimum(np.floor(x).astype(int), RV_SECTORS - 2)
        wall = _filled(_outer_radius(r, th, rv))
        wall = np.maximum(np.interp(th, BIN_CENTRES, wall, period=TAU), epi + 1)
        across = np.clip((r - epi) / (wall - epi), 0, 1)
        beyond = np.exp(-np.maximum(r - wall, 0) / DECAY)
        side = np.exp(-np.maximum(np.maximum(lo - phi, phi - hi), 0) * r / DECAY)
        for s in range(RV_SECTORS):
            h = (i == s) * (1 - (x - i)) + (i == s - 1) * (x - i)
            fields += [h * across * beyond * side * e_r, h * across * beyond * side * e_t]
    return np.stack(fields)


def spline_basis(mask: np.ndarray, n: int = 4) -> np.ndarray:
    """Generic control: cubic B-spline bumps on an n x n grid over the heart box, 2n^2 fields."""
    ys, xs = np.nonzero(binary_dilation(mask > 0, iterations=ROI_DILATION))
    grid_y, grid_x = np.mgrid[:mask.shape[0], :mask.shape[1]]

    def bump(t: np.ndarray) -> np.ndarray:
        a = np.abs(t)
        return np.where(a < 1, 2 / 3 - a ** 2 + a ** 3 / 2, np.where(a < 2, (2 - a) ** 3 / 6, 0))

    h_y, h_x = max(np.ptp(ys), 1) / (n - 1), max(np.ptp(xs), 1) / (n - 1)
    bumps = [bump((grid_y - (ys.min() + a * h_y)) / h_y) * bump((grid_x - (xs.min() + b * h_x)) / h_x)
             for a in range(n) for b in range(n)]
    zero = np.zeros_like(bumps[0])
    return np.stack([np.stack([b, zero]) for b in bumps] + [np.stack([zero, b]) for b in bumps])


def roi(mask: np.ndarray) -> np.ndarray:
    return binary_dilation(mask > 0, iterations=ROI_DILATION)


def project(fields: np.ndarray, v: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Least-squares fit of v (2, H, W) on region; returns the reconstructed field (2, H, W)."""
    a = fields[:, :, region].reshape(len(fields), -1).T
    c = np.linalg.lstsq(a, v[:, region].ravel(), rcond=1e-6)[0]
    return np.tensordot(c, fields, 1)


def radial_at(w: np.ndarray, k: dict, radius: np.ndarray) -> np.ndarray:
    """Radial component of the displacement w (2, H, W) on the circle-like curve `radius` (per bin)."""
    c, s = np.cos(BIN_CENTRES), np.sin(BIN_CENTRES)
    at = [k["cy"] + radius * s, k["cx"] + radius * c]
    return map_coordinates(w[0], at, order=1) * c + map_coordinates(w[1], at, order=1) * s


def thickening(w: np.ndarray, k: dict) -> np.ndarray:
    """Wall thickness change per bin (px) under w, the forward displacement of ED points."""
    return radial_at(w, k, k["epi"]) - radial_at(w, k, k["endo"])


def positions(mask: np.ndarray) -> dict[int, str]:
    """Slice -> basal/mid/apical: thirds of the slices with an LV; base = end with the larger LV area."""
    lv = [z for z in range(len(mask)) if (mask[z] == 3).sum() >= MIN_PX]
    thirds = np.array_split(lv, 3)
    if (mask[thirds[0]] == 3).sum() < (mask[thirds[-1]] == 3).sum():
        thirds = thirds[::-1]
    return {int(z): pos for pos, third in zip(SECTORS, thirds) for z in third}


def shared_slots(n_fields: int, pos: str) -> np.ndarray:
    """Patient-level motion index of each basis field of a slice in ring pos."""
    n = 3 * SECTORS[pos]
    slots = [0, 1] if n_fields == 2 else [0, 1, *range(2 + 3 * OFFSET[pos], 2 + 3 * OFFSET[pos] + n)]
    if n_fields > 2 + n:
        slots += range(2 + 3 * 16, SHARED)
    return np.array(slots)


def time_basis(tau: np.ndarray, harmonics: int = 2) -> np.ndarray:
    """Smooth cyclic curves that vanish at ED, (frames, 2 * harmonics)."""
    w = TAU * np.outer(tau, np.arange(1, harmonics + 1))
    return np.concatenate([1 - np.cos(w), np.sin(w)], 1)


def patient_fields(mask: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Per slice of an ED stack: decoder fields, their patient-level slots, and the fit region."""
    septum, where = septal_angle(mask), positions(mask)
    fields, slots, regions = [], [], []
    for z, m in enumerate(mask):
        pos = where.get(z, "mid")
        fields.append(basis(m, septum, SECTORS[pos]))
        slots.append(shared_slots(len(fields[-1]), pos))
        regions.append(roi(m))
    return fields, slots, regions


def fit_shared(fields: list[np.ndarray], slots: list[np.ndarray], regions: list[np.ndarray],
               motion: np.ndarray) -> np.ndarray:
    """Least-squares motions shared across slices, (SHARED, F), from motion (F, Z, 2, H, W)."""
    normal, moment = np.zeros((SHARED, SHARED)), np.zeros((SHARED, len(motion)))
    for z, (b, idx, region) in enumerate(zip(fields, slots, regions)):
        a = b[:, :, region].reshape(len(b), -1)
        normal[np.ix_(idx, idx)] += a @ a.T
        moment[idx] += a @ motion[:, z][:, :, region].reshape(len(motion), -1).T
    return np.linalg.lstsq(normal, moment, rcond=1e-6)[0]


def fit_compact(d: dict, harmonics: int = 2) -> np.ndarray:
    """Compact rig numbers of a cached patient, (SHARED, 2 * harmonics)."""
    frames = [t for t in range(len(d["tau"])) if t != int(d["ed"])]
    shared = fit_shared(*patient_fields(d["ed_mask"]), d["svf"][frames].astype(np.float32))
    return shared @ np.linalg.pinv(time_basis(d["tau"][frames], harmonics)).T


def decode(numbers: np.ndarray, tau: float, fields: list[np.ndarray], slots: list[np.ndarray]) -> np.ndarray:
    """SVF stack (Z, 2, H, W) at phase tau from compact numbers (SHARED, 2 * harmonics)."""
    c = numbers @ time_basis(np.array([tau]), numbers.shape[1] // 2)[0]
    return np.stack([np.tensordot(c[idx], b, 1) for b, idx in zip(fields, slots)])


def edit(numbers: np.ndarray, alpha: dict[int, float]) -> np.ndarray:
    """Scale the endo and epi radial curves of LV segments {segment: alpha}; 1 = unchanged, 0 = akinetic."""
    out = numbers.copy()
    for seg, a in alpha.items():
        out[2 + 3 * seg: 4 + 3 * seg] *= a
    return out


def token_maps(fields: list[np.ndarray], slots: list[np.ndarray]) -> np.ndarray:
    """Where each token lives, (Z, TOKENS, H, W): support of its fields; the last token is translation."""
    h, w = fields[0].shape[-2:]
    maps = np.zeros((len(fields), TOKENS, h, w), np.float32)
    for z, (b, idx) in enumerate(zip(fields, slots)):
        norm = np.linalg.norm(b, axis=1)
        for field, slot in zip(norm, idx):
            token = TOKENS - 1 if slot < 2 else (slot - 2) // 3 if slot < 50 else 16 + (slot - 50) // 2
            maps[z, token] = np.maximum(maps[z, token], field)
    return maps
