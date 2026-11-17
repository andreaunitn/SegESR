"""
Tests for the fine-structure metrics (data_tools/fine_structure.py).

Run with: pytest tests/test_fine_structure.py
"""

import json

import cv2
import numpy as np
import pytest

from data_tools.fine_structure import BANDS, band_metrics, edge_metrics, image_metrics, main
from data_tools.results_table import load_results

def make_image(size=128, seed=0):
    """Smooth gradient with sharp squares and fine stripes, in [0, 1]."""

    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size] / size
    img = 0.3 + 0.3 * x
    img[20:60, 20:60] = 0.9
    img[80:110, 70:120] += 0.2 * (np.sin(np.arange(50) * 2.5)[None, :] > 0)
    return np.clip(img + 0.01 * rng.standard_normal(img.shape), 0, 1)

def blur(img):
    return cv2.GaussianBlur(img, (0, 0), 1.5)

def band_mse(psnr):
    return 10 ** (-psnr / 10)

def test_band_mses_add_up_to_the_image_mse():
    gt = make_image()
    sr = blur(gt)
    results = band_metrics(sr, gt)

    total = sum(band_mse(results[f"psnr_{name}"]) for name, _, _ in BANDS)
    assert total == pytest.approx(np.mean((sr - gt) ** 2), rel=1e-6)

def test_identical_images():
    gt = make_image()
    results = image_metrics(gt.copy(), gt)

    for name, _, _ in BANDS:
        assert results[f"energy_{name}"] == pytest.approx(1.0)
        assert results[f"psnr_{name}"] > 100
    assert results["edge_f"] == pytest.approx(1.0)

def test_blur_removes_high_frequencies_and_edges():
    gt = make_image()
    results = image_metrics(blur(gt), gt)

    assert results["energy_high"] < 0.5                    # detail missing
    assert results["energy_low"] == pytest.approx(1.0, abs=0.02)
    assert results["psnr_high"] < results["psnr_low"]      # the error is fine detail
    assert results["edge_f"] < 1.0

def test_noise_adds_high_frequency_energy():
    gt = make_image()
    noisy = gt + 0.05 * np.random.default_rng(1).standard_normal(gt.shape)

    assert band_metrics(noisy, gt)["energy_high"] > 1.5    # more detail than the GT

def test_edge_tolerance():
    image = make_image()
    gt, shifted = image[:, 8:-8], image[:, 7:-9]          # every edge moved by one pixel, no wrap-around

    assert edge_metrics(shifted, gt, tolerance=2)["edge_f"] > 0.95
    assert edge_metrics(shifted, gt, tolerance=0)["edge_f"] < 0.5

def test_main_writes_results_readable_by_results_table(tmp_path):
    gt_dir, sr_dir = tmp_path / "gt", tmp_path / "sr" / "sample00"
    gt_dir.mkdir()
    sr_dir.mkdir(parents=True)
    for i in range(2):
        gt = (make_image(seed=i) * 255).round().astype(np.uint8)
        cv2.imwrite(str(gt_dir / f"img{i}.png"), cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR))
        sr = (blur(make_image(seed=i)) * 255).round().astype(np.uint8)
        cv2.imwrite(str(sr_dir / f"img{i}.png"), cv2.cvtColor(sr, cv2.COLOR_GRAY2BGR))

    main(["--sr_dir", str(tmp_path / "sr"), "--gt_dir", str(gt_dir), "--dataset", "Toy", "--name", "blurred",
          "--output_dir", str(tmp_path / "metrics")])

    saved = json.loads((tmp_path / "metrics" / "Toy" / "finestruct_blurred.json").read_text())
    assert set(dict(saved["Toy"])) >= {"psnr_high", "energy_high", "edge_f", "edge_psnr"}
    results = load_results(str(tmp_path / "metrics"), prefix="finestruct")
    assert results["Toy"]["blurred"]["energy_high"] < 1.0
