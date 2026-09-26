import torch

from src.models.consistency import consistency
from src.utils.svf import integrate, warp
from tests.test_augment import smooth


def test_true_field_scores_best():
    """With target = ED o exp(v), the symmetric LNCC is highest at v, and the regulariser is 0 at v = 0."""
    ed, v = smooth(4, 1, 1.0, 0), smooth(4, 2, 3.0, 1)
    target = warp(ed, integrate(v))
    sim_true, _ = consistency(v, ed, target)
    for wrong in (torch.zeros_like(v), 0.5 * v, -v):
        assert (sim_true > consistency(wrong, ed, target)[0]).all()
    assert torch.allclose(consistency(torch.zeros_like(v), ed, target)[1], torch.zeros(4))
