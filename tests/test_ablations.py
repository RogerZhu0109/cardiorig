import numpy as np
import pandas as pd
import torch

from scripts.evaluate_ablations import against_full
from scripts.train_rig import VARIANTS, ablate, build
from src.rig import SHARED, TOKENS


def inputs():
    g = torch.Generator().manual_seed(0)
    x, maps = torch.randn(5, 4, 128, 128, generator=g), torch.rand(5, TOKENS, 128, 128, generator=g)
    return x, maps, torch.tensor([0, 0, 0, 1, 1]), torch.randn(5, 768, 12, 12, generator=g), torch.randn(2, 768, generator=g)


def test_every_variant_runs_and_full_is_unchanged():
    x, maps, owner, feat, cls = inputs()
    for variant in VARIANTS:
        model = build(variant).eval()
        torch.nn.init.normal_(model.lv_head.weight)
        v_maps, v_feat, v_cls = ablate(variant, maps, feat, cls)
        mean, log_var = model(x, v_maps, owner, v_feat, v_cls)
        assert mean.shape == log_var.shape == (2, SHARED, 4)
        changed = variant in ("no_cinema", "uniform_pool")
        assert (v_maps is maps and v_feat is feat and v_cls is cls) != changed
    assert set(build("full").state_dict()) - set(build("no_mixer").state_dict())  # the mixer is what goes


def test_uniform_pool_gives_every_token_the_same_features():
    x, maps, owner, feat, cls = inputs()
    model = build("no_mixer").eval()
    torch.nn.init.zeros_(model.token_id)
    torch.nn.init.normal_(model.lv_head.weight)
    mean, _ = model(x, torch.ones_like(maps), owner, feat, cls)
    lv = mean[:, 2:2 + 16 * 3].reshape(2, 16, 3, 4)
    assert torch.allclose(lv, lv[:, :1].expand_as(lv), atol=1e-5)


def test_verdicts():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.6, 0.9, 100)
    rows = [dict(pid=f"p{i:03d}", method=m, seed=s, dice=base[i] + shift + rng.normal(0, 0.002))
            for m, shift in (("full", 0.0), ("worse", -0.02), ("same", 0.0), ("better", 0.02))
            for s in range(3) for i in range(100)]
    df = pd.DataFrame(rows)
    assert against_full(df, "worse").endswith("-> CONTRIBUTES")
    assert against_full(df, "same").endswith("-> NO MEASURABLE EFFECT")
    assert against_full(df, "better").endswith("-> BETTER WITHOUT")
