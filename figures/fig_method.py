"""
Paper Fig. 2 (method): (a) the rig, a fixed decoder from 58 x 4 parameters to motion; (b) the network that
predicts the parameters from the ED stack (src/rig.py, src/models/rig_net.py).
Data: cached ACDC patient062. The parameters shown are the rig fitted to dense registration (the training
target, fit_compact), not a model prediction. CVPR full width (6.875 in); Arial text, STIX sans math,
TrueType fonts embedded.
Usage: python figures/fig_method.py  ->  figures/pdf/method.pdf
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures" / "pdf"
sys.path.insert(0, str(REPO))
from src.rig import SHARED, decode, fit_compact, patient_fields, positions, time_basis, token_maps  # noqa: E402
from src.utils.svf import integrate, warp  # noqa: E402

C = {
    "k": "#1A1A1A", "gray": "#6E6E6E", "edge": "#3A3A3A",
    "green_f": "#D5E8CF", "green_e": "#4E6B45", "green_t": "#35602F",
    "blue_f": "#CFE0F3", "blue_s": "#A9C4E4", "blue_e": "#274C7E", "blue_t": "#1F3F7A",
    "teal_f": "#E3F1F3", "teal_e": "#2F6F76", "teal_t": "#23585E",
    "gray_f": "#EFEFEF", "purple_t": "#5A2A6E", "red_t": "#9B2D2D", "gold": "#E0B43C",
}
SEG_C = ["#B64342", "#E9A6A1", "#E0B43C", "#8BCF8B", "#42949E", "#3775BA"]
RV_C = ["#9A4D8E", "#B97DB0", "#D3A9CC", "#E8D2E4"]
plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
                     "mathtext.fontset": "stixsans", "axes.unicode_minus": False,
                     "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})

# Layout is in design units on a 15 x 7.5 canvas, printed at CVPR text width.
W, H = 15.0, 7.5
PRINT_W = 6.875  # in


def fs(size):
    """Design font size -> printed points (smallest labels print at about 6.5 pt)."""
    return 0.55 * size


def lw_(w):
    """Design line width -> printed points."""
    return 0.6 * w


# ------------------------------------------------------------------ data
with np.load(REPO / "data/processed/acdc/training/patient062.npz") as f:
    d = {k: f[k] for k in f}
ED, ES = int(d["ed"]), int(d["es"])
mask = d["ed_mask"]
where = positions(mask)
ring = {p: [z for z, q in sorted(where.items()) if q == p] for p in ("basal", "mid", "apical")}
ZM = ring["mid"][len(ring["mid"]) // 2]
STACK = [ring["basal"][len(ring["basal"]) // 2], ZM, ring["apical"][len(ring["apical"]) // 2]]
fields, slots, _ = patient_fields(mask)
numbers = fit_compact(d)  # (SHARED, 4)
tau_es = float(d["tau"][ES])
v = decode(numbers, tau_es, fields, slots)[ZM]
teacher = d["svf"][ES, ZM].astype(np.float32)


def norm(img):
    lo, hi = np.percentile(img, [1, 99.5])
    return np.clip((img - lo) / (hi - lo), 0, 1)


M = norm(d["cine"][ED, ZM].astype(np.float32))
u = integrate(torch.from_numpy(v[None]).float())
I_es = warp(torch.from_numpy(M[None, None]).float(), u)[0, 0].numpy()
onehot = F.one_hot(torch.from_numpy(mask[ZM].astype(np.int64)), 4).permute(2, 0, 1)[None].float()
m_es = warp(onehot, u).argmax(1)[0].numpy()
u = u[0].numpy()

ys, xs = np.nonzero(mask[ZM] > 0)
cy, cx, half = int(ys.mean()), int(xs.mean()), 38
CROP = (slice(cy - half, cy + half), slice(cx - half, cx + half))
NC = 2 * half


def cr(a):
    return a[..., CROP[0], CROP[1]]


# ------------------------------------------------------------------ canvas
fig = plt.figure(figsize=(PRINT_W, PRINT_W * H / W))


def layer(z):
    ax = fig.add_axes([0, 0, 1, 1], zorder=z)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.patch.set_alpha(0)
    return ax


cv, ov = layer(0), layer(50)


def sub(x, y, w, h, z=3):
    return fig.add_axes([x / W, y / H, w / W, h / H], zorder=z)


def txt(x, y, s, size=13, color=C["k"], weight="normal", ha="center", va="center", **kw):
    ov.text(x, y, s, fontsize=fs(size), color=color, fontweight="bold" if weight == "bold" else "normal",
            fontstyle="italic" if weight == "italic" else "normal", ha=ha, va=va, zorder=kw.pop("zorder", 10), **kw)


def arrow(p, q, color=C["k"], lw=1.2, rad=0.0, ls="-", hw=3.2):
    hw = 0.6 * hw
    ov.add_patch(FancyArrowPatch(p, q, arrowstyle=f"-|>,head_length={hw * 1.7},head_width={hw}", mutation_scale=1,
                                 lw=lw_(lw), color=color, ls=ls, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}", zorder=9))


def line(xs, ys, color=C["k"], lw=1.1, ls="-", z=8, hop=False):
    effects = [pe.Stroke(linewidth=lw_(lw) + 2.2, foreground="white"), pe.Normal()] if hop else None
    ov.plot(xs, ys, color=color, lw=lw_(lw), ls=ls, solid_capstyle="butt", zorder=z, path_effects=effects)


def trapezoid(x, y, w, h_big, h_small, fc, ec, narrowing=True, lw=1.3):
    hl, hr = (h_big, h_small) if narrowing else (h_small, h_big)
    ov.add_patch(Polygon([(x, y - hl / 2), (x + w, y - hr / 2), (x + w, y + hr / 2), (x, y + hl / 2)],
                         fc=fc, ec=ec, lw=lw_(lw), zorder=3))


def rbox(x, y, w, h, fc, ec, lw=1.2, rs=0.1, ls="-", ax=None, z=3):
    (ax or ov).add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rs}", fc=fc, ec=ec,
                                        lw=lw_(lw), ls=ls, zorder=z))


def circle(x, y, sym, r=0.14):
    ov.add_patch(Circle((x, y), r, fc="white", ec=C["k"], lw=lw_(1.2), zorder=5))
    if sym == "dot":
        ov.add_patch(Circle((x, y), 0.035, fc=C["k"], ec="none", zorder=6))
    else:
        txt(x, y - 0.005, sym, 14, zorder=6)


def cuboid(x, y, w, h, dz, fc=C["blue_f"], ec=C["blue_e"]):
    """Feature map: front face w x h (channels x height), receding depth dz (width)."""
    top = [(x, y + h / 2), (x + dz, y + h / 2 + dz), (x + w + dz, y + h / 2 + dz), (x + w, y + h / 2)]
    side = [(x + w, y - h / 2), (x + w + dz, y - h / 2 + dz), (x + w + dz, y + h / 2 + dz), (x + w, y + h / 2)]
    ov.add_patch(Polygon(top, fc="#E4EEF9", ec=ec, lw=lw_(1.0), zorder=3))
    ov.add_patch(Polygon(side, fc=C["blue_s"], ec=ec, lw=lw_(1.0), zorder=3))
    ov.add_patch(Rectangle((x, y - h / 2), w, h, fc=fc, ec=ec, lw=lw_(1.0), zorder=3))


def snowflake(x, y, r=0.11, color="#5DA5DA"):
    for a in np.deg2rad([90, 30, -30]):
        dx, dy = r * np.cos(a), r * np.sin(a)
        line([x - dx, x + dx], [y - dy, y + dy], color=color, lw=1.6, z=12)
        for sgn in (1, -1):
            bx, by = x + sgn * 0.6 * dx, y + sgn * 0.6 * dy
            for t in (np.pi / 4, -np.pi / 4):
                ca, sa = np.cos(a + t), np.sin(a + t)
                line([bx, bx + sgn * 0.3 * r * ca], [by, by + sgn * 0.3 * r * sa], color=color, lw=1.2, z=12)


def flame(x, y, s=0.26):
    t = np.linspace(0, 2 * np.pi, 120)
    for scale, col, dy in ((1.0, "#F16913", 0.0), (0.55, "#FDBE6E", -0.2)):
        px = x + 0.42 * s * scale * np.sin(t) * np.abs(np.sin(t / 2)) ** 1.6
        py = y + s * (dy + 0.5 * scale * np.cos(t)) + 0.04 * s * np.sin(3 * t) * scale
        ov.add_patch(Polygon(np.c_[px, py], closed=True, fc=col, ec="none", zorder=12))


def frame(ax, color=C["edge"], lw=0.9):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(color)
        sp.set_linewidth(lw_(lw))


def image(x, y, s, arr, z=3):
    ax = sub(x, y, s, s, z=z)
    ax.imshow(cr(arr), cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
    frame(ax)
    return ax


def contours(ax, m, lw=0.9):
    ax.contour(cr(m) == 3, levels=[0.5], colors=C["gold"], linewidths=lw_(lw))
    ax.contour(cr(m) >= 2, levels=[0.5], colors="#79C7D1", linewidths=lw_(lw * 0.9))
    ax.contour(cr(m) == 1, levels=[0.5], colors="#C58BBE", linewidths=lw_(lw * 0.9))


def field(x, y, s, vec, bg=None, color=C["blue_t"], step=6, length=None, faint=0.72):
    ax = sub(x, y, s, s)
    ax.imshow(np.ones((NC, NC)) if bg is None else faint + (1 - faint) * cr(bg), cmap="gray", vmin=0, vmax=1)
    vv = cr(vec)
    gy, gx = np.mgrid[step // 2:NC:step, step // 2:NC:step]
    k = 1.0 if length is None else length * step / max(np.hypot(*vv).max(), 1e-6)
    ax.quiver(gx, gy, k * vv[0][gy, gx], k * vv[1][gy, gx], color=color, angles="xy", scale_units="xy", scale=1,
              width=0.012, headwidth=3.6, headlength=3.6, headaxislength=3.2)
    ax.set_xlim(-0.5, NC - 0.5)
    ax.set_ylim(NC - 0.5, -0.5)
    frame(ax)
    return ax


def grid(x, y, s, disp, step=6, color=C["blue_e"]):
    ax = sub(x, y, s, s)
    uu, t = cr(disp), np.arange(NC)
    for k in range(step // 2, NC, step):
        ax.plot(t + uu[0][k, :], k + uu[1][k, :], color=color, lw=lw_(0.7))
        ax.plot(k + uu[0][:, k], t + uu[1][:, k], color=color, lw=lw_(0.7))
    ax.set_xlim(-0.5, NC - 0.5)
    ax.set_ylim(NC - 0.5, -0.5)
    frame(ax)


def panel(x, y, letter, title, color):
    txt(x, y, letter, 18, weight="bold", ha="left", va="top")
    txt(x + 0.36, y - 0.15, title, 15, color, "bold", ha="left")


def token_colour(t):
    if t < 16:
        return SEG_C[t % 6] if t < 12 else SEG_C[[0, 2, 4, 5][t - 12]]
    return RV_C[t - 16] if t < 20 else "#8A8A8A"


TCOL = [token_colour(t) for t in range(21)]


def token_overlay(ax, z):
    tm = token_maps(fields, slots)[z][:20]
    lab, strength = tm.argmax(0), tm.max(0)
    rgba = np.zeros((*lab.shape, 4))
    for t in range(20):
        sel = (lab == t) & (strength > 0.15)
        rgba[sel] = (*to_rgb(TCOL[t]), 0)
        rgba[sel, 3] = 0.85 * np.clip(strength[sel], 0, 1)
    ax.imshow(rgba[CROP[0], CROP[1]], interpolation="nearest")


# ================================================================== (b) rig network
panel(0.1, 4.35, "b", "Rig network", C["blue_t"])
ya = 1.55
S = 0.95
for i, z in enumerate(STACK[::-1]):
    ox, oy = 0.3 + 0.16 * (2 - i), ya - S / 2 + 0.16 * (2 - i) - 0.16
    ax = image(ox, oy, S, norm(d["cine"][ED, z].astype(np.float32)), z=3 + i)
    contours(ax, mask[z], lw=0.6)
txt(0.3 + (S + 0.32) / 2, ya - 0.98, "ED stack\n+ masks", 13, linespacing=1.1)

# CineMA, frozen: patch features z (into the CNN bottleneck) and the cls token
ey = 3.2
trapezoid(2.0, ey, 0.7, 0.9, 0.56, C["green_f"], C["green_e"])
txt(1.45, ey + 0.2, "CineMA", 14, C["green_t"], "bold")
txt(2.42, ey - 0.02, r"$\mathcal{E}$", 18)
snowflake(2.16, ey + 0.22)
xs0 = 0.3 + 0.16 + S / 2
line([xs0, xs0], [ya + S / 2 + 0.16, ey])
arrow((xs0, ey), (2.0, ey))
zy, cy_ = ey - 0.12, ey + 0.17  # z and cls outputs
ov.add_patch(Rectangle((2.85, zy - 0.13), 0.12, 0.26, fc=C["green_f"], ec=C["green_e"], lw=lw_(1.1), zorder=3))
arrow((2.7, zy), (2.85, zy))
txt(3.07, zy, r"$z$", 15, ha="left")
ov.add_patch(Rectangle((2.85, cy_ - 0.06), 0.12, 0.12, fc=C["green_f"], ec=C["green_e"], lw=lw_(1.1), zorder=3))
arrow((2.7, cy_), (2.85, cy_))
txt(2.91, cy_ + 0.22, "cls", 12, C["green_t"])

# CNN, per slice: shrinking feature maps
cx0 = 1.95
arrow((0.3 + S + 0.34, ya), (cx0, ya))
dims = [(0.07, 1.0, 0.32), (0.13, 0.74, 0.24), (0.22, 0.5, 0.16)]
xx, block_top = cx0, []
for w_, h_, dz in dims:
    cuboid(xx, ya, w_, h_, dz)
    block_top.append((xx + w_ / 2 + dz / 2, ya + h_ / 2 + dz / 2))
    xx += w_ + dz + 0.13
cnn_end = xx - 0.13
txt((cx0 + cnn_end) / 2, ya - 0.82, "CNN (per slice)", 13, C["blue_t"])
flame(1.72, ya + 0.62)
# patch features z join the bottleneck
line([2.91, 2.91], [zy - 0.13, ya + 0.6], color=C["green_e"], hop=True, z=9)
arrow((2.91, ya + 0.6), (2.91, ya + 0.45), color=C["green_e"])

# cls -> MLP -> e: FiLM in every CNN block, added to every token
mx, mw, mh = 3.27, 0.45, 0.28
arrow((2.97, cy_), (mx, cy_), color=C["green_e"])
rbox(mx, cy_ - mh / 2, mw, mh, C["blue_f"], C["blue_e"], rs=0.06)
txt(mx + mw / 2, cy_, "MLP", 11)
flame(mx + mw / 2, cy_ + 0.36, s=0.2)
ex_ = mx + mw + 0.18
line([mx + mw, ex_], [cy_, cy_], color=C["blue_e"])
txt(ex_ + 0.06, cy_ + 0.13, r"$e$", 15, ha="left")
film_y = 2.57
line([ex_, ex_], [film_y, cy_], color=C["blue_e"])
line([block_top[0][0], ex_], [film_y, film_y], color=C["blue_e"], z=7)
for bx_, by_ in block_top:
    arrow((bx_, film_y), (bx_, by_ + 0.02), color=C["blue_e"], lw=1.0, hw=2.6)
txt(3.5, film_y + 0.06, "FiLM", 11, C["blue_t"], va="bottom")

# anatomical pooling with token maps
pl = cnn_end + 1.2
arrow((cnn_end + 0.02, ya), (pl - 0.14, ya))
circle(pl, ya, "dot")
ts = 0.85
tx, ty = pl - ts / 2, ya + 0.45
ax = image(tx, ty, ts, 0.55 + 0.45 * M)
token_overlay(ax, ZM)
arrow((pl, ty - 0.01), (pl, ya + 0.15))
txt(pl + 0.06, ty + ts + 0.26, "anatomical\ntoken pooling", 12, C["teal_t"], linespacing=1.0)
txt(pl, ya - 0.82, "pool", 13, C["k"])

# 21 anatomical tokens, plus e
kx, kh = pl + 0.8, 1.2
cell = kh / 21
arrow((pl + 0.14, ya), (kx, ya))
for t in range(21):
    ov.add_patch(Rectangle((kx, ya + kh / 2 - (t + 1) * cell), 0.16, cell, fc=TCOL[t], ec="white", lw=lw_(0.4),
                           zorder=3))
ov.add_patch(Rectangle((kx, ya - kh / 2), 0.16, kh, fc="none", ec=C["edge"], lw=lw_(1.0), zorder=4))
txt(kx + 0.08, ya - 0.82, "21 tokens", 13)
tc, bus_y, py_ = kx + 0.08, 3.85, ya + kh / 2 + 0.3
line([ex_, ex_], [cy_, bus_y], color=C["blue_e"])
line([ex_, tc], [bus_y, bus_y], color=C["blue_e"])
arrow((tc, bus_y), (tc, py_ + 0.12), color=C["blue_e"])
circle(tc, py_, "+", r=0.12)
arrow((tc, py_ - 0.12), (tc, ya + kh / 2 + 0.01))

# transformer: three stacked layers, one attention fan drawn on the front layer
bxl, bw, bh, off = kx + 0.5, 0.95, 1.0, 0.07
arrow((kx + 0.18, ya), (bxl, ya))
for k in (2, 1, 0):
    rbox(bxl + off * k, ya - bh / 2 + off * k, bw, bh, "#F4F8FD" if k == 0 else "#DCE7F5", C["blue_e"], rs=0.1)
qy = ya + np.linspace(0.3, -0.3, 5)
qx0, qx1 = bxl + 0.22, bxl + bw - 0.22
for b, w_ in zip(qy, (0.7, 2.2, 1.2, 0.6, 1.5)):
    line([qx0, qx1], [qy[2], b], color=C["blue_e"], lw=w_, z=5)
ov.scatter(np.full(5, qx0), qy, s=8, c=["#C9D7EA"] * 2 + [C["blue_e"]] + ["#C9D7EA"] * 2, edgecolors=C["blue_e"],
           linewidths=lw_(0.6), zorder=6)
ov.scatter(np.full(5, qx1), qy, s=8, c="#C9D7EA", edgecolors=C["blue_e"], linewidths=lw_(0.6), zorder=6)
txt(bxl + bw / 2 + off, ya + bh / 2 + 0.38, "Transformer", 14, C["blue_t"], "bold")
txt(bxl + bw / 2 + off, ya - bh / 2 - 0.2, r"$\times 3$ layers", 12, C["gray"])
flame(bxl + bw / 2 + off + 0.88, ya + bh / 2 + 0.4)
tR = bxl + bw + 2 * off  # right edge of the back layer

# shared heads, one per token type
hx = tR + 0.55
groups = [("LV", 0.5), ("RV", 0.0), ("T", -0.5)]
for name, off in groups:
    gy = ya + off
    arrow((tR, ya + 0.1 + 0.6 * off), (hx, gy), lw=0.9)
    trapezoid(hx, gy, 0.42, 0.36, 0.2, C["blue_f"], C["blue_e"], lw=1.1)
    txt(hx + 0.19, gy, name, 11)
txt(hx + 0.21, ya - 0.82, "heads", 13)
flame(hx + 0.66, ya + 0.66)

# the 232 parameters
CMAP = LinearSegmentedColormap.from_list("theta", ["#3775BA", "#F7F7F7", "#B64342"])
nx0, nw, nh = hx + 0.8, 2.6, 0.5
for _, off in groups:
    arrow((hx + 0.42, ya + off), (nx0, ya + off * 0.35), lw=0.9)
A = numbers.T / np.abs(numbers.T).max(1, keepdims=True)
axn = sub(nx0, ya - nh / 2, nw, nh)
axn.imshow(A, cmap=CMAP, vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
for xb in (1.5, 49.5):
    axn.axvline(xb, color=C["k"], lw=lw_(1.0))
frame(axn)
for lo, hi, lab in ((0, 2, "T"), (2, 50, "LV segments"), (50, 58, "RV")):
    a0, a1 = nx0 + nw * lo / SHARED, nx0 + nw * hi / SHARED
    txt((a0 + a1) / 2, ya - nh / 2 - 0.17, lab, 11, C["gray"])
txt(nx0 + nw / 2 - 0.25, ya + nh / 2 + 0.45, r"$\hat\Theta \in \mathbb{R}^{58 \times 4}$", 17)
txt(nx0 + nw / 2 - 0.25, ya + nh / 2 + 0.15, "rig parameters", 13)

# training target and loss
lx = nx0 + nw + 0.95
arrow((nx0 + nw + 0.05, ya), (lx - 0.05, ya), color=C["red_t"], lw=1.4, ls=(0, (4, 2.5)))
txt(lx + 0.05, ya, r"$\mathcal{L}_{\beta\mathrm{NLL}}$", 18, C["red_t"], ha="left")
fs_ = 0.85
fx_, fy_ = lx + 0.1, ya + 0.75
field(fx_, fy_, fs_, teacher, bg=M, color="#5F5F5F", length=1.2)
txt(fx_ + fs_ / 2, fy_ + fs_ + 0.17, "dense registration", 13, C["gray"])
arrow((fx_ + fs_ / 2, fy_ - 0.02), (lx + 0.35, ya + 0.22), color=C["red_t"], lw=1.2, ls=(0, (4, 2.5)))
txt(fx_ + fs_ + 0.1, fy_ + 0.12, r"$\Theta^\ast$", 15, C["gray"], ha="left")

# ================================================================== (a) the rig
panel(0.1, 7.45, "a", "The rig", C["teal_t"])
yb = 5.85
xm = 2.15
# Theta from (b)
x_out, y_link = nx0 + nw - 0.2, 4.62
line([x_out, x_out], [ya + nh / 2, y_link])
line([x_out, xm], [y_link, y_link])
arrow((xm, y_link), (xm, yb - 0.14))
txt(xm + 0.1, y_link + 0.22, r"$\Theta$", 15, ha="left")
circle(xm, yb, r"$\times$")
# time curves
axt = sub(0.45, yb - 0.5, 1.25, 1.0)
tt = np.linspace(0, 1, 200)
for j, col in enumerate(["#0F4D92", "#42949E", "#3775BA", "#9A4D8E"]):
    axt.plot(tt, time_basis(tt)[:, j], color=col, lw=lw_(1.6))
axt.set_xlim(0, 1)
axt.set_ylim(-1.1, 2.2)
axt.set_xticks([0, 1])
axt.set_xticklabels(["ED", "1"], fontsize=fs(11))
axt.tick_params(width=lw_(1.0), length=2)
axt.set_yticks([])
axt.set_xlabel(r"$\tau$", fontsize=fs(14), labelpad=-5)
axt.spines["bottom"].set_linewidth(lw_(1.0))
for sp in ("top", "right", "left"):
    axt.spines[sp].set_visible(False)
txt(1.07, yb + 0.72, r"$b(\tau)$", 15)
arrow((1.75, yb), (xm - 0.14, yb))

# basis fields from the ED mask (one segment shown)
fb, gap = 0.9, 1.0
bx = xm + 0.75
rbox(bx - 0.12, yb - fb / 2 - 0.12, 2 * gap + fb + 0.24, fb + 0.24, C["teal_f"], C["teal_e"], rs=0.1, ax=cv)
arrow((xm + 0.14, yb), (bx - 0.12, yb))
txt((xm + bx) / 2 - 0.05, yb + 0.22, r"$c(\tau)$", 14)
names = ["endo", "epi", "tangential"]
seg = 6 + 3  # a mid-ring segment
for j in range(3):
    k = int(np.nonzero(slots[ZM] == 2 + 3 * seg + j)[0][0])
    field(bx + j * gap, yb - fb / 2, fb, fields[ZM][k], bg=M, color=C["teal_t"], step=5, length=1.9, faint=0.8)
    txt(bx + j * gap + fb / 2, yb - fb / 2 - 0.3, names[j], 12, C["teal_t"])
txt(bx + gap + fb / 2, yb + fb / 2 + 0.32, r"rig basis $B_k$ (from ED mask)", 13, C["teal_t"])
sm = bx + 2 * gap + fb + 0.45
arrow((bx + 2 * gap + fb + 0.12, yb), (sm - 0.14, yb))
circle(sm, yb, r"$\Sigma$")
# velocity field
vx, fs2 = sm + 0.38, 0.95
arrow((sm + 0.14, yb), (vx, yb))
field(vx, yb - fs2 / 2, fs2, v, bg=M, length=1.1)
txt(vx + fs2 / 2, yb - fs2 / 2 - 0.27, r"$v_\tau=\sum_k c_k(\tau)\,B_k$", 13)
# integrate
ex = vx + fs2 + 0.3
arrow((vx + fs2 + 0.03, yb), (ex, yb))
rbox(ex, yb - 0.3, 0.7, 0.6, C["gray_f"], C["edge"])
txt(ex + 0.35, yb, r"$\exp$", 16)
gx = ex + 0.98
arrow((ex + 0.7, yb), (gx, yb))
grid(gx, yb - fs2 / 2, fs2, u)
txt(gx + fs2 / 2, yb - fs2 / 2 - 0.27, r"$\phi_\tau$", 15)
# warp the ED frame
wx = gx + fs2 + 0.3
arrow((gx + fs2 + 0.03, yb), (wx - 0.14, yb))
circle(wx, yb, "dot")
ix = wx + 0.3
arrow((wx + 0.14, yb), (ix, yb))
ax = image(ix, yb - 0.55, 1.1, I_es)
contours(ax, m_es)
txt(ix + 0.55, yb - 0.85, r"$\hat I_\tau = I_{ED}\circ\phi_\tau$", 14)
yt = 6.9
arrow((wx, yt), (wx, yb + 0.15), color=C["red_t"], lw=1.3, ls=(0, (4, 2.5)))
txt(wx + 0.1, yt - 0.02, "ED frame: moved, not generated", 12, C["red_t"], "italic", ha="left")

# legend
gx0, gy0 = ix + 1.55, 5.1
rbox(gx0, gy0, 1.75, 1.0, "none", C["k"], lw=1.2, ls=(0, (4, 2.5)), rs=0.12)
snowflake(gx0 + 0.25, gy0 + 0.72)
txt(gx0 + 0.45, gy0 + 0.72, "frozen", 13, ha="left")
flame(gx0 + 0.25, gy0 + 0.3)
txt(gx0 + 0.45, gy0 + 0.3, "trained", 13, ha="left")

OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "method.pdf")
print("saved", OUT / "method.pdf")
