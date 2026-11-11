"""
Multimodality STRAUS cine loader (Zhou et al., IEEE TMI 2018): 18 simulated cine sequences,
3 anatomies x 6 conditions, with the true motion as one tetrahedral mesh per frame.

Per sequence (data/raw/STRAUS/patientNN_condition/):
    image/cinefrmTT.{mhd,raw}   (I, J, K) int16, 30 frames, 1.25 mm in-plane, ~8 mm slices
    mesh/cinemeshTT.vtk         vertices in mm, fixed connectivity; frame 0 = the reference mesh
and data/raw/STRAUS/RefMeshP{1,2,3}.vtk with the AHA segment of every cell (0 = RV wall, 1-17 = LV).

Geometry (checked by overlay on all three anatomies): a mesh point sits at image index
(point - Offset) / ElementSpacing. The header TransformMatrix does not apply to the meshes.

The stack is brought to the ACDC convention the models were trained on: slice 0 basal, in-plane
quarter turns so that the LV -> RV direction is nearest the ACDC training mean, 1.5 mm, 128 x 128
crop on the ED mask centroid, intensities scaled by ED-frame percentiles. Labels as in ACDC
(1 RV cavity, 2 LV myocardium, 3 LV cavity) come from cutting the mesh with each slice plane.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, distance_transform_edt, label, zoom

from src.data.acdc import center_crop

ACDC_SEPTAL = np.radians(164)  # circular mean LV -> RV angle (x right, y down) over the 100 ACDC training patients
MIN_PX = 20  # smallest cavity kept, as in src/rig.py
RV_WALL = 18  # value of RV-wall pixels in the "aha" map


def read_mhd(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Volume (K, J, I) float32, Offset and ElementSpacing (x, y, z) in mm."""
    h = dict(line.split(" = ") for line in Path(path).read_text().splitlines() if " = " in line)
    dims = [int(x) for x in h["DimSize"].split()]
    vol = np.fromfile(Path(path).with_suffix(".raw"), dtype="<i2").reshape(dims[::-1]).astype(np.float32)
    return vol, np.array(h["Offset"].split(), float), np.array(h["ElementSpacing"].split(), float)


def read_points(path: Path) -> np.ndarray:
    """Vertices (N, 3) of a legacy ASCII vtk mesh."""
    t = Path(path).read_text().split()
    i = t.index("POINTS")
    n = int(t[i + 1])
    return np.array(t[i + 3:i + 3 + 3 * n], float).reshape(n, 3)


def read_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Tetrahedra (M, 4) vertex indices and the AHA segment of every cell (M,)."""
    t = Path(path).read_text().split()
    i = t.index("CELLS")
    m = int(t[i + 1])
    tets = np.array(t[i + 3:i + 3 + 5 * m], int).reshape(m, 5)[:, 1:]
    j = t.index("AHA", t.index("CELL_DATA"))
    return tets, np.array(t[j + 4:j + 4 + m], float).astype(int)


def quarter_turns(xy: np.ndarray, turns: int, width: int, height: int) -> tuple[np.ndarray, int, int]:
    """Pixel coordinates (N, 2) as x, y after np.rot90(image, turns) of a (height, width) image."""
    for _ in range(turns % 4):
        xy = np.stack([xy[:, 1], width - 1 - xy[:, 0]], 1)
        width, height = height, width
    return xy, width, height


def rasterize(v: np.ndarray, tets: np.ndarray, z: float, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Cut a tetrahedral mesh (vertices v (N, 3) as x, y, z) by the plane z. Per pixel centre: the
    index of the tetrahedron that contains it (-1 outside) and its barycentric weights (H, W, 4)."""
    tet, weights = np.full(shape, -1), np.zeros((*shape, 4))
    p = v[tets]
    idx = np.flatnonzero((p[..., 2].min(1) <= z) & (p[..., 2].max(1) >= z))
    edges = (p[idx, 1:] - p[idx, :1]).transpose(0, 2, 1)
    solid = np.abs(np.linalg.det(edges)) > 1e-9
    idx, inv, p = idx[solid], np.linalg.inv(edges[solid]), p[idx[solid]]
    if not len(idx):
        return tet, weights
    lo = np.floor(p[..., :2].min(1)).astype(int)
    span = int(np.ceil((p[..., :2].max(1) - lo).max())) + 1
    for dy in range(span):
        for dx in range(span):
            q = lo + (dx, dy)
            b = np.einsum("mij,mj->mi", inv, np.concatenate([q, np.full((len(q), 1), z)], 1) - p[:, 0])
            bary = np.concatenate([1 - b.sum(1, keepdims=True), b], 1)
            ok = (bary >= -1e-9).all(1) & (q >= 0).all(1) & (q[:, 0] < shape[1]) & (q[:, 1] < shape[0])
            tet[q[ok, 1], q[ok, 0]] = idx[ok]
            weights[q[ok, 1], q[ok, 0]] = bary[ok]
    return tet, weights


def labels(seg: np.ndarray) -> np.ndarray:
    """ACDC labels of one slice from the mesh cross-section seg (-1 outside, 0 RV-labelled wall,
    1-17 LV segment). Cavities are the holes the walls enclose: the LV cavity is bordered mostly by
    LV-labelled wall, the RV cavity is not. The mesh labels the RV insertions 0 across the whole LV
    wall, so RV-labelled wall within the LV wall thickness of the LV cavity counts as myocardium."""
    wall, lv_wall = seg >= 0, seg > 0
    holes, n = label(binary_fill_holes(wall) & ~wall)
    cavity = {True: np.zeros_like(wall), False: np.zeros_like(wall)}  # keyed by "is LV"
    for i in range(1, n + 1):
        hole = holes == i
        is_lv = bool(lv_wall[binary_dilation(hole) & wall].mean() > 0.5)
        if hole.sum() >= max(MIN_PX, cavity[is_lv].sum() + 1):
            cavity[is_lv] = hole
    myo = lv_wall
    if cavity[True].any():
        depth = distance_transform_edt(~cavity[True])
        near, _ = label(wall & (depth <= np.percentile(depth[lv_wall], 90)) | lv_wall)
        myo = np.isin(near, np.unique(near[lv_wall])) & wall
    out = np.zeros(seg.shape, np.uint8)
    out[cavity[False]], out[myo], out[cavity[True]] = 1, 2, 3
    return out


def list_patients(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).glob("patient*") if p.is_dir())


class STRAUSPatient:
    def __init__(self, pdir: Path):
        self.dir = Path(pdir)
        self.pid = self.dir.name
        self.anatomy = (int(self.pid[7:9]) - 1) % 3 + 1  # patients cycle through the 3 anatomies
        self.condition = self.pid.split("_")[1]
        self.n_frames = len(list((self.dir / "image").glob("*.mhd")))
        self.tets, self.aha = read_reference(self.dir.parent / f"RefMeshP{self.anatomy}.vtk")

    def cine(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(T, K, J, I) float32, Offset, ElementSpacing."""
        frames = [read_mhd(self.dir / "image" / f"cinefrm{t:02d}.mhd") for t in range(self.n_frames)]
        return np.stack([f[0] for f in frames]), frames[0][1], frames[0][2]

    def meshes(self) -> np.ndarray:
        """(T, N, 3) vertex positions in mm."""
        return np.stack([read_points(self.dir / "mesh" / f"cinemesh{t:02d}.vtk") for t in range(self.n_frames)])

    def processed(self, size: int = 128, target_mm: float = 1.5) -> dict:
        """
        ACDC-style stack plus the simulated truth, ED = frame 0, ES = smallest true LV cavity:
            cine (T, Z, size, size) in [0, 1];  ed_mask, es_mask (Z, size, size);  tau (T,)
            masks  (T, Z, size, size)     true labels of every frame
            truth  (T, Z, 2, size, size)  true in-plane displacement (px, ch0 = x) of the tissue under
                                          each ED pixel of the walls, 0 elsewhere (forward: ED -> frame)
            through (T, Z, size, size)    its through-plane part in mm, positive toward the base
            aha    (Z, size, size)        AHA segment of that tissue: 1-17 LV, RV_WALL, 0 = no wall
        """
        cine, offset, spacing = self.cine()
        lo, hi = np.percentile(cine[0], [1, 99])  # ED frame only, as for ACDC
        cine = np.clip((cine - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
        T, K, J, I = cine.shape
        pts = (self.meshes() - offset) / spacing  # (T, N, 3) as image index i, j, k

        # in-plane quarter turns from the ED mesh: wall-node centroids give the LV -> RV direction
        node_aha = np.zeros(pts.shape[1], int)
        np.maximum.at(node_aha, self.tets.ravel(), np.repeat(self.aha, 4))
        to_rv = pts[0, node_aha == 0, :2].mean(0) - pts[0, node_aha > 0, :2].mean(0)
        angle = np.arctan2(to_rv[1], to_rv[0])
        turns = int(np.argmin([np.abs(np.angle(np.exp(1j * (angle - r * np.pi / 2 - ACDC_SEPTAL)))) for r in range(4)]))

        cine = zoom(np.rot90(cine, turns, axes=(-2, -1))[:, ::-1], (1, 1, spacing[1] / target_mm, spacing[0] / target_mm), order=1)
        shape = cine.shape[-2:]
        verts = np.empty_like(pts)
        for t in range(T):
            xy, w, h = quarter_turns(pts[t, :, :2], turns, I, J)
            verts[t, :, :2] = xy * [(shape[1] - 1) / (w - 1), (shape[0] - 1) / (h - 1)]  # scipy zoom keeps the corners
            verts[t, :, 2] = pts[t, :, 2]

        masks = np.zeros((T, K, *shape), np.uint8)
        truth, through = np.zeros((T, K, 2, *shape), np.float32), np.zeros((T, K, *shape), np.float32)
        aha = np.zeros((K, *shape), np.uint8)
        ys, xs = np.mgrid[:shape[0], :shape[1]]
        for z in range(K):
            k = K - 1 - z  # slice 0 basal
            for t in range(T):
                tet, bary = rasterize(verts[t], self.tets, k, shape)
                seg = np.where(tet >= 0, self.aha[tet], -1)
                masks[t, z] = labels(seg)
                if t == 0:
                    wall, corners, weights = tet >= 0, self.tets[tet[tet >= 0]], bary[tet >= 0]
                    aha[z][wall] = np.where(seg[wall] > 0, seg[wall], RV_WALL)
            for t in range(T):
                moved = np.einsum("pc,pcd->pd", weights, verts[t][corners]) if wall.any() else np.zeros((0, 3))
                truth[t, z, 0][wall], truth[t, z, 1][wall] = moved[:, 0] - xs[wall], moved[:, 1] - ys[wall]
                through[t, z][wall] = (moved[:, 2] - k) * spacing[2]

        # slices as an annotator would label them at ED: from the most basal closed LV ring to the apex
        ring = [z for z in range(K) if (masks[0, z] == 3).any()]
        keep = np.array([z for z in range(K) if ring and z >= ring[0] and (masks[0, z] == 2).sum() >= MIN_PX])
        _, yy, xx = np.nonzero(masks[0, keep] > 0)
        center = (int(round(yy.mean())), int(round(xx.mean())))
        crop = lambda a: center_crop(a, center, size)  # noqa: E731
        masks = crop(masks[:, keep])
        es = int((masks == 3).sum((1, 2, 3)).argmin())
        return dict(
            pid=self.pid, group=self.condition, anatomy=self.anatomy, ed=0, es=es,
            tau=(np.arange(T) / T).astype(np.float32), z=K - 1 - keep, center=np.array(center), turns=turns,
            cine=crop(cine[:, keep]), ed_mask=masks[0], es_mask=masks[es], masks=masks,
            truth=crop(truth[:, keep]), through=crop(through[:, keep]), aha=crop(aha[keep]),
        )
