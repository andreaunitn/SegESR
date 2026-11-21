"""
Tests for the alignment measures and the degradations of step R.0 (data_tools/feature_alignment.py).

Run with: pytest tests/test_feature_alignment.py
"""

import numpy as np
import pytest
import torch
from PIL import Image

from data_tools.feature_alignment import LEVELS, cknna, degrade, gate, linear_cka, seed_of, tokens

def features(n=64, d=32, seed=0):
    return torch.from_numpy(np.random.default_rng(seed).normal(size=(n, d))).float()

def test_cknna_is_one_for_the_same_features_up_to_rotation_and_scale():
    a = features()
    q, _ = torch.linalg.qr(torch.from_numpy(np.random.default_rng(1).normal(size=(32, 32))).float())
    assert cknna(a, a, k=8) == pytest.approx(1.0, abs=1e-6)
    assert cknna(a, 3.0 * a @ q, k=8) == pytest.approx(1.0, abs=1e-6)

def test_cknna_is_low_for_independent_features_and_grows_with_shared_structure():
    a, b = features(seed=0), features(seed=2)
    independent = cknna(a, b, k=8)
    mixed = cknna(a, torch.cat([a, 0.5 * b], dim=1), k=8)
    assert abs(independent) < 0.3
    assert mixed > independent + 0.3

def test_linear_cka():
    a = features()
    assert linear_cka(a, a) == pytest.approx(1.0, abs=1e-6)
    assert linear_cka(a, 2 * a + 1) == pytest.approx(1.0, abs=1e-6)
    assert linear_cka(a, features(seed=3)) < 0.5

def test_tokens_pool_to_the_grid():
    out = tokens(torch.ones(1, 5, 16, 16), grid=8)
    assert out.shape == (64, 5)

def test_degradations_are_reproducible_and_ordered():
    rng = np.random.default_rng(0)
    hr = Image.fromarray((rng.random((64, 64, 3)) * 255).astype(np.uint8))
    lrs = {lv: np.asarray(degrade(hr, lv, seed_of("a.png", lv)), dtype=np.float32) for lv in LEVELS}
    assert all(lr.shape == (16, 16, 3) for lr in lrs.values())
    assert np.array_equal(lrs["strong"], np.asarray(degrade(hr, "strong", seed_of("a.png", "strong")), np.float32))
    clean = np.asarray(hr.resize((16, 16), Image.BICUBIC), dtype=np.float32)
    errors = [np.mean((lrs[lv] - clean) ** 2) for lv in ("mild", "medium", "strong")]
    assert errors[0] < errors[1] < errors[2]

def test_gate():
    m = lambda c, r: {"cknna": c, "spearman_lpips": r}
    measures = {"mid": {"mild": m(0.5, -0.4), "medium": m(0.45, -0.35), "strong": m(0.35, -0.3)},
                "up0": {"mild": m(0.5, 0.0), "medium": m(0.5, 0.0), "strong": m(0.48, 0.0)}}
    result = gate(measures, ["mid", "up0"])
    assert result["drop_ok"] and result["spearman_ok"]
    assert result["cknna_drop_mild_to_strong"]["mid"] == pytest.approx(0.3)
    measures["mid"]["strong"]["cknna"] = 0.45
    assert not gate(measures, ["mid", "up0"])["drop_ok"]
