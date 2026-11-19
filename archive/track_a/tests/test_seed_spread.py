"""
Tests for the seed part of the error per frequency band (data_tools/seed_spread.py).

Run with: pytest tests/test_seed_spread.py
"""

import json

import cv2
import numpy as np
import pytest

from data_tools.fine_structure import BANDS, band_metrics
from data_tools.seed_spread import aggregate, band_power, image_spread, main

def make_image(size=64, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size] / size
    return 0.3 + 0.3 * x + 0.1 * np.sin(8 * np.pi * y) + 0.02 * rng.standard_normal((size, size))

def test_band_power_adds_up_and_matches_fine_structure():
    a, b = make_image(seed=0), make_image(seed=1)
    power = band_power(a - b)

    assert sum(power.values()) == pytest.approx(np.mean((a - b) ** 2), rel=1e-9)
    psnr = band_metrics(a, b)
    for name, _, _ in BANDS:
        assert 10 * np.log10(1 / power[name]) == pytest.approx(psnr[f"psnr_{name}"])

def test_error_splits_into_systematic_and_seed_parts():
    gt = make_image(seed=0)
    samples = [gt + 0.05 + 0.03 * np.random.default_rng(i).standard_normal(gt.shape) for i in range(1, 5)]
    result = image_spread(samples, gt)
    k = len(samples)

    for name, _, _ in BANDS:
        # E||s - gt||^2 = ||mean - gt||^2 + E||s - mean||^2, with the biased (1/K) variance
        assert result[f"_err_{name}"] == pytest.approx(result[f"_mean_err_{name}"] + result[f"_var_{name}"] * (k - 1) / k)
    # the constant offset is systematic and sits at frequency 0; the noise is seed variance spread over all bands
    assert result["_mean_err_low"] > 10 * result["_var_low"]

def test_identical_samples_have_no_seed_part():
    gt, sr = make_image(seed=0), make_image(seed=1)
    results = aggregate([image_spread([sr, sr, sr], gt)])

    for name, _, _ in BANDS:
        assert results[f"seed_share_{name}"] == pytest.approx(0.0, abs=1e-20)
        assert results[f"psnr_{name}"] == pytest.approx(results[f"psnr_mean_{name}"])

def test_main(tmp_path):
    sr_dir, gt_dir = tmp_path / "sr", tmp_path / "gt"
    gt_dir.mkdir()
    for name in ("a.png", "b.png"):
        gt = (np.clip(np.stack([make_image(seed=0)] * 3, -1), 0, 1) * 255).round().astype(np.uint8)
        cv2.imwrite(str(gt_dir / name), gt)
        for k in range(3):
            (sr_dir / f"sample{k:02d}").mkdir(parents=True, exist_ok=True)
            noise = np.random.default_rng(k).integers(-10, 11, gt.shape)
            cv2.imwrite(str(sr_dir / f"sample{k:02d}" / name), np.clip(gt.astype(int) + noise, 0, 255).astype(np.uint8))

    main(["--sr_dir", str(sr_dir), "--gt_dir", str(gt_dir), "--dataset", "Toy", "--name", "toy",
          "--mean_dir", str(tmp_path / "mean"), "--output_dir", str(tmp_path / "metrics")])
    with open(tmp_path / "metrics" / "Toy" / "seedspread_toy.json") as f:
        results = dict(json.load(f)["Toy"])

    assert sorted(p.name for p in (tmp_path / "mean" / "sample00").iterdir()) == ["a.png", "b.png"]
    for name, _, _ in BANDS:
        assert results[f"seed_share_{name}"] == pytest.approx(1.0, abs=0.1)   # pure noise around the GT: all seed part
        assert results[f"psnr_mean_{name}"] > results[f"psnr_{name}"]

    main(["--sr_dir", str(sr_dir), "--dataset", "Toy", "--name", "nogt", "--output_dir", str(tmp_path / "metrics")])
    with open(tmp_path / "metrics" / "Toy" / "seedspread_nogt.json") as f:
        assert set(dict(json.load(f)["Toy"])) == {f"spread_{name}" for name, _, _ in BANDS}
