import numpy as np
import torch

from src.data.cache import VAL_PATIENTS, Pairs
from src.data.acdc import TUNE_PATIENTS


def fake_patient(seed, T=5, Z=3, S=4):
    """Every array entry encodes its (frame, slice) so the pairing can be checked."""
    ed = seed % T
    cine = np.zeros((T, Z, S, S), np.float16)
    svf = np.zeros((T, Z, 2, S, S), np.float16)
    for t in range(T):
        for z in range(Z):
            cine[t, z] = 100 * seed + 10 * t + z
            svf[t, z] = 100 * seed + 10 * t + z
    feat = np.stack([np.full((768, 2, 2), 100 * seed + z, np.float16) for z in range(Z)])
    return dict(ed=ed, tau=((np.arange(T) - ed) % T / T).astype(np.float32), z=np.arange(Z), cine=cine, svf=svf,
                feat=feat, cls=np.full(768, seed, np.float16))


def test_pairs_match_slice_and_phase():
    pts = [fake_patient(1), fake_patient(2, T=4, Z=2)]
    pairs = Pairs(pts, "cpu")
    assert len(pairs) == 4 * 3 + 3 * 2  # non-ED frames x slices
    ed, tau, v, cond = pairs.sample(500, torch.Generator().manual_seed(0))
    code = v[:, 0, 0, 0]
    seed, t, z = code // 100, code % 100 // 10, code % 10
    for i in range(len(code)):
        d = pts[int(seed[i]) - 1]
        assert int(t[i]) != d["ed"]
        assert ed[i, 0, 0, 0] == 100 * seed[i] + 10 * d["ed"] + z[i]  # ED slice of the same patient and z
        assert torch.isclose(tau[i], torch.tensor(d["tau"][int(t[i])]))
        assert cond["feat"][i, 0, 0, 0] == 100 * seed[i] + z[i] and cond["cls"][i, 0] == seed[i]  # same slice


def test_val_split():
    assert len(VAL_PATIENTS) == 10 and not set(VAL_PATIENTS) & set(TUNE_PATIENTS)
