import numpy as np

from src.data.straus import labels, quarter_turns, rasterize

# a box split into 6 tetrahedra around its main diagonal
CORNERS = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], float)
TETS = np.array([[0, 1, 3, 7], [0, 3, 2, 7], [0, 2, 6, 7], [0, 6, 4, 7], [0, 4, 5, 7], [0, 5, 1, 7]])


def test_quarter_turns_follow_rot90():
    h, w = 5, 8
    for turns in range(4):
        for x, y in ((0, 0), (7, 0), (2, 3), (6, 4)):
            img = np.zeros((h, w))
            img[y, x] = 1
            (nx, ny), nw, nh = quarter_turns(np.array([[x, y]], float), turns, w, h)[0][0], *quarter_turns(np.zeros((1, 2)), turns, w, h)[1:]
            out = np.rot90(img, turns)
            assert out.shape == (nh, nw) and out[int(ny), int(nx)] == 1


def test_rasterize_box_and_barycentric_weights():
    v = CORNERS * [8.0, 6.0, 2.0] + [2.2, 3.4, -1.0]  # x 2.2-10.2, y 3.4-9.4, z -1..1
    tet, w = rasterize(v, TETS, 0.0, (16, 16))
    ys, xs = np.mgrid[:16, :16]
    inside = (xs > 2.2) & (xs < 10.2) & (ys > 3.4) & (ys < 9.4)
    assert np.array_equal(tet >= 0, inside)
    hit = tet >= 0
    recon = np.einsum("pc,pcd->pd", w[hit], v[TETS[tet[hit]]])
    assert np.abs(recon - np.stack([xs[hit], ys[hit], np.zeros(hit.sum())], 1)).max() < 1e-9
    assert not (rasterize(v, TETS, 1.5, (16, 16))[0] >= 0).any()  # plane outside the box
    # a linear motion of the vertices is reproduced exactly at every pixel
    moved = v @ np.array([[1.1, 0.1, 0], [-0.2, 0.9, 0], [0, 0, 1]]).T + [1.0, -2.0, 0.5]
    disp = np.einsum("pc,pcd->pd", w[hit], moved[TETS[tet[hit]]]) - recon
    assert np.allclose(disp[:, 0], 0.1 * xs[hit] + 0.1 * ys[hit] + 1.0) and np.allclose(disp[:, 2], 0.5)


def test_labels_from_walls():
    ys, xs = np.mgrid[:64, :64]
    r_lv, r_rv = np.hypot(xs - 38, ys - 32), np.hypot(xs - 26, ys - 32)
    lv_wall = (r_lv >= 9) & (r_lv < 14)
    rv_wall = (r_rv >= 18) & (r_rv < 21) & (r_lv >= 14)
    seg = np.where(lv_wall, 7, np.where(rv_wall, 0, -1))
    m = labels(seg)
    assert (m == 2).sum() >= lv_wall.sum() and (m == 2)[lv_wall].all() and abs((m == 3).sum() - np.pi * 81) < 30
    assert (m == 1).sum() > 200 and (m == 2).sum() < lv_wall.sum() + 40  # at most the two insertions of the RV wall
    # a band of the LV ring labelled as RV wall (the RV insertion) still closes the ring
    junction = np.where(lv_wall & (np.abs(ys - 28) < 3) & (xs < 38), 0, seg)
    m2 = labels(junction)
    assert (m2 == 3).sum() == (m == 3).sum() and (m2 == 2)[junction > 0].all() and (m2 == 2)[lv_wall].mean() > 0.97
    open_ring = np.where(lv_wall & (xs < 45), 7, -1)
    assert not (labels(open_ring) == 3).any()
