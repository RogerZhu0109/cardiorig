import nibabel as nib
import numpy as np

from src.data.mnms import MnMsPatient, septal_angle, turns_to_acdc
from src.data.straus import ACDC_SEPTAL


def heart(angle: float, shape=(96, 96), lv: int = 8) -> np.ndarray:
    """One ACDC-labelled slice: LV cavity (3) in a myocardial ring (2), RV cavity (1) at angle (x right, y down)."""
    ys, xs = np.mgrid[:shape[0], :shape[1]]
    r = np.hypot(xs - 48, ys - 48)
    m = np.zeros(shape, np.uint8)
    m[np.hypot(xs - 48 - 22 * np.cos(angle), ys - 48 - 22 * np.sin(angle)) < 8] = 1
    m[r < 13], m[r < lv] = 2, 3
    return m


def test_turns_bring_the_septum_to_acdc():
    for deg in range(0, 360, 10):
        m = heart(np.radians(deg))[None]
        assert abs(np.degrees(septal_angle(m)) % 360 - deg) < 1
        turned = np.rot90(m, turns_to_acdc(septal_angle(m)), axes=(-2, -1))
        off = np.degrees(np.angle(np.exp(1j * (septal_angle(turned) - ACDC_SEPTAL))))
        assert abs(off) <= 45.5, (deg, off)


def test_processed_follows_acdc_convention(tmp_path):
    T, Z, ed, es = 6, 5, 4, 1
    acdc = np.zeros((Z, 96, 96), np.uint8)
    acdc[1:4] = heart(np.radians(170))  # slices 1-3 labelled
    acdc[1][acdc[1] == 1] = 0  # first labelled slice (basal after ordering) has no RV
    systole = np.zeros_like(acdc)
    systole[1:4] = heart(np.radians(170), lv=5)
    to_file = lambda m: np.array([0, 3, 2, 1])[m].transpose(2, 1, 0) * 1.00000002  # noqa: E731  M&Ms labels, float noise
    gt = np.zeros((96, 96, Z, T), np.float32)
    gt[..., ed], gt[..., es] = to_file(acdc), to_file(systole)
    cine = np.random.default_rng(0).random((96, 96, Z, T), dtype=np.float32) + (gt[..., ed] > 0)[..., None]
    for flip, mirror in ((False, False), (True, False), (False, True)):
        # slice axis toward the apex (left, anterior, inferior) with det < 0, or its reverse with det > 0
        k = np.array([-1.0, 1.0, -1.0]) * (-1 if flip else 1)
        i = np.array([1.0, 1.0, 0.0])
        j = np.cross(k, i) * (1 if flip != mirror else -1)
        aff = np.eye(4)
        aff[:3, :3] = np.stack([1.5 * i / np.linalg.norm(i), 1.5 * j / np.linalg.norm(j), 10 * k / np.linalg.norm(k)], 1)
        assert (np.linalg.det(aff[:3, :3]) < 0) == (flip == mirror)
        d = tmp_path / f"X{int(flip)}{int(mirror)}"
        d.mkdir()
        nib.save(nib.Nifti1Image(cine, aff), d / f"{d.name}_sa.nii.gz")
        nib.save(nib.Nifti1Image(gt, aff), d / f"{d.name}_sa_gt.nii.gz")
        row = dict(ED=str(ed), ES=str(es), Pathology="NOR", Vendor="A", Centre="1")
        out = MnMsPatient(d, row).processed()
        found = MnMsPatient(d, {**row, "ED": "0", "ES": "0"}).processed()  # frames missing from the table
        assert (found["ed"], found["es"]) == (ed, es) and np.array_equal(found["cine"], out["cine"])
        assert out["mirrored"] == mirror and abs(np.angle(np.exp(1j * (septal_angle(out["ed_mask"]) - ACDC_SEPTAL)))) < np.pi / 4
        assert (out["es_mask"] == 3).sum() < (out["ed_mask"] == 3).sum() == (acdc == 3).sum()  # 1.5 mm already
        assert out["tau"][ed] == 0 and np.isclose(out["tau"][es], 3 / 6)
        if mirror:  # flipped in-plane, then half a turn brings the septum back
            assert out["turns"] == 2 and out["z"].tolist() == [1, 2, 3]
            continue
        assert out["cine"].shape == (T, 3, 128, 128) and out["ed_mask"].shape == (3, 128, 128)
        assert out["z"].tolist() == ([3, 2, 1] if flip else [1, 2, 3])
        basal = 2 if flip else 0  # the slice without RV sits at file index 1
        assert [(m == 1).any() for m in out["ed_mask"]] == [i != basal for i in range(3)]
        assert out["turns"] == 0
        ys, xs = np.nonzero(out["ed_mask"][1] == 3)
        assert abs(ys.mean() - 64) < 6 and abs(xs.mean() - 64) < 6  # crop is centred on the heart
