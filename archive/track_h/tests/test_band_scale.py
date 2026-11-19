"""
Tests for the generated-detail scaling (data_tools/band_scale.py).

Run with: pytest tests/test_band_scale.py
"""

import cv2
import numpy as np
import pytest

from data_tools.band_scale import (band_windows, full_band, high_energy, high_pass, main, oracle_gains, replace_low_band,
                                   scale_bands, scale_high_frequencies)

CUTOFF = 1 / 8

def make_image(size=64, seed=0):
    """Smooth color gradient plus fine noise, kept inside [0.2, 0.8] so that scaling does not clip."""

    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size] / size
    base = np.stack([0.3 + 0.2 * x, 0.4 + 0.1 * y, 0.5 - 0.1 * x], axis=-1)
    return base + 0.02 * rng.standard_normal(base.shape)

def test_mask_is_smooth_high_pass():
    mask = high_pass((64, 64), CUTOFF)

    assert mask[0, 0] == 0.0                          # DC kept
    assert mask[0, 32] == 1.0                         # Nyquist fully scaled
    assert np.all((mask >= 0) & (mask <= 1))

def test_alpha_one_is_identity():
    image = make_image()
    assert np.allclose(scale_high_frequencies(image, 1.0, CUTOFF), image)

def test_alpha_scales_the_energy_above_the_cutoff():
    image = make_image()
    region = full_band(image.shape[:2], CUTOFF)
    energy = high_energy(image, region)

    assert high_energy(scale_high_frequencies(image, 0.0, CUTOFF), region) < 1e-6 * energy
    assert high_energy(scale_high_frequencies(image, 2.0, CUTOFF), region) == pytest.approx(4 * energy, rel=1e-6)

def test_low_frequencies_are_kept():
    image = make_image()
    low = high_pass(image.shape[:2], CUTOFF) == 0.0
    out = scale_high_frequencies(image, 0.0, CUTOFF)

    assert high_energy(out - image, low) < 1e-12 * high_energy(image, low)

def test_band_windows_add_up_to_the_high_pass():
    lower, upper = (window for window, _ in band_windows((64, 64), CUTOFF))
    assert np.allclose(lower + upper, high_pass((64, 64), CUTOFF))

def test_equal_gains_match_a_single_alpha():
    image = make_image()
    assert np.allclose(scale_bands(image, [0.5, 0.5], CUTOFF), scale_high_frequencies(image, 0.5, CUTOFF))

def test_oracle_matches_the_gt_energy_per_octave():
    gt = make_image(seed=0)
    sr = gt + 0.03 * np.random.default_rng(1).standard_normal(gt.shape)     # too much fine detail
    gains = oracle_gains(sr, gt, CUTOFF)
    out = scale_bands(sr, gains, CUTOFF)

    assert all(g < 1 for g in gains)
    for _, core in band_windows(gt.shape[:2], CUTOFF):
        assert high_energy(out, core) == pytest.approx(high_energy(gt, core), rel=1e-6)

def test_main_writes_scaled_images(tmp_path):
    sr_dir, gt_dir, out_dir = tmp_path / "sr" / "sample00", tmp_path / "gt", tmp_path / "out"
    sr_dir.mkdir(parents=True)
    gt_dir.mkdir()
    for i in range(2):
        image = (np.clip(make_image(seed=i), 0, 1) * 255).round().astype(np.uint8)
        cv2.imwrite(str(sr_dir / f"img{i}.png"), image)
        cv2.imwrite(str(gt_dir / f"img{i}.png"), image)

    main(["--sr_dir", str(tmp_path / "sr"), "--output_dir", str(out_dir), "--alpha", "0"])
    assert sorted(p.name for p in (out_dir / "sample00").iterdir()) == ["img0.png", "img1.png"]

    main(["--sr_dir", str(tmp_path / "sr"), "--output_dir", str(tmp_path / "oracle"), "--oracle", "--gt_dir", str(gt_dir)])
    assert (tmp_path / "oracle" / "sample00" / "img1.png").exists()

    with pytest.raises(SystemExit):
        main(["--sr_dir", str(tmp_path / "sr"), "--output_dir", str(out_dir)])     # neither --alpha nor --oracle

def test_low_band_replacement():
    image, source = make_image(seed=0), make_image(seed=1)
    out = replace_low_band(image, source, CUTOFF)
    mask = high_pass(image.shape[:2], CUTOFF)

    assert high_energy(out - source, mask == 0.0) < 1e-12 * high_energy(source, mask == 0.0)   # low band from source
    assert high_energy(out - image, mask == 1.0) < 1e-12 * high_energy(image, mask == 1.0)     # high band kept
    assert np.allclose(replace_low_band(image, image, CUTOFF), image)

def test_main_low_from(tmp_path):
    sr_dir, gt_dir = tmp_path / "sr" / "sample00", tmp_path / "gt"
    sr_dir.mkdir(parents=True)
    gt_dir.mkdir()
    sr = (np.clip(make_image(seed=0), 0, 1) * 255).round().astype(np.uint8)
    gt = (np.clip(make_image(seed=1), 0, 1) * 255).round().astype(np.uint8)
    cv2.imwrite(str(sr_dir / "img.png"), sr)
    cv2.imwrite(str(gt_dir / "img.png"), gt)

    main(["--sr_dir", str(tmp_path / "sr"), "--output_dir", str(tmp_path / "out"), "--low_from", str(gt_dir),
          "--cutoff", "0.0625"])
    out = cv2.imread(str(tmp_path / "out" / "sample00" / "img.png")).astype(np.float64)
    low = high_pass(out.shape[:2], 0.0625) == 0.0
    assert high_energy(out - gt, low) < 0.02 * high_energy(out - sr, low)     # up to the 8-bit rounding

    with pytest.raises(SystemExit):
        main(["--sr_dir", str(tmp_path / "sr"), "--output_dir", str(tmp_path / "x"), "--alpha", "0", "--low_from", str(gt_dir)])
