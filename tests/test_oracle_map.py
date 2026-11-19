"""
Tests for the per-tile oracle (data_tools/oracle_map.py) and the curve gap (data_tools/curve_gap.py).

Run with: pytest tests/test_oracle_map.py
"""

import json

import cv2
import numpy as np
import pytest

from data_tools.curve_gap import curve_value, gaps
from data_tools.curve_gap import main as gap_main
from data_tools.oracle_map import blend, luma, main, oracle_choice, tile_errors

def make_gt(h=96, w=128, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(0.5 + 0.2 * rng.standard_normal((h, w, 3)), 0, 1)

def write(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (rgb[..., ::-1] * 255).round().astype(np.uint8))

def read(path):
    return cv2.imread(str(path), cv2.IMREAD_COLOR)[..., ::-1].astype(np.float64) / 255

def psnr(a, b):
    return -10 * np.log10(np.mean((luma(a) - luma(b)) ** 2))

def split_candidates(gt, rng):
    """Candidate a is exact on the left half and noisy on the right, b the opposite."""

    noise = 0.2 * rng.standard_normal(gt.shape)
    w = gt.shape[1]
    a, b = gt.copy(), gt.copy()
    a[:, w // 2:] += noise[:, w // 2:]
    b[:, :w // 2] += noise[:, :w // 2]
    return np.clip(a, 0, 1), np.clip(b, 0, 1)

def test_choice_follows_the_better_candidate():
    gt = make_gt()
    a, b = split_candidates(gt, np.random.default_rng(1))
    choice = oracle_choice(tile_errors([a, b], gt, 32))
    assert choice.shape == (3, 4)
    assert (choice[:, :2] == 0).all() and (choice[:, 2:] == 1).all()

def test_blend_weights_sum_to_one():
    img = np.full((50, 70, 3), 0.3)
    choice = np.random.default_rng(0).integers(0, 3, size=(2, 3))
    out = blend([img, img, img], choice, 32)        # edge tiles smaller than 32
    assert out.shape == img.shape
    assert np.allclose(out, 0.3)

def test_blend_of_a_single_choice_is_that_candidate():
    gt = make_gt()
    a, b = split_candidates(gt, np.random.default_rng(2))
    assert np.allclose(blend([a, b], np.ones((3, 4), dtype=int), 32), b)

def test_shuffle_keeps_the_counts():
    errors = np.random.default_rng(3).random((3, 4, 5))
    choice = oracle_choice(errors)
    shuffled = oracle_choice(errors, "shuffle", np.random.default_rng(0))
    assert sorted(choice.ravel()) == sorted(shuffled.ravel())

def test_main_writes_a_better_composite(tmp_path):
    rng = np.random.default_rng(4)
    gt_dir = tmp_path / "gt"
    for i in range(2):
        gt = make_gt(seed=10 + i)
        a, b = split_candidates(gt, rng)
        write(gt_dir / f"img{i}.png", gt)
        write(tmp_path / "a" / "sample00" / f"img{i}.png", a)
        write(tmp_path / "b" / "sample00" / f"img{i}.png", b)

    out = tmp_path / "oracle"
    main(["--candidates", str(tmp_path / "a"), str(tmp_path / "b"), "--gt_dir", str(gt_dir),
          "--output_dir", str(out), "--tile", "32"])
    for i in range(2):
        gt = read(gt_dir / f"img{i}.png")
        composite = read(out / "sample00" / f"img{i}.png")
        assert psnr(composite, gt) > max(psnr(read(tmp_path / d / "sample00" / f"img{i}.png"), gt) for d in "ab") + 3
    shares = json.loads((out / "choice.json").read_text())["tile_share"]
    assert sum(shares.values()) == pytest.approx(1.0)

def test_main_rejects_missing_images(tmp_path):
    write(tmp_path / "gt" / "x.png", make_gt())
    write(tmp_path / "a" / "sample00" / "x.png", make_gt())
    (tmp_path / "b" / "sample00").mkdir(parents=True)
    with pytest.raises(SystemExit):
        main(["--candidates", str(tmp_path / "a"), str(tmp_path / "b"), "--gt_dir", str(tmp_path / "gt"),
              "--output_dir", str(tmp_path / "out")])

def test_curve_value():
    curve = [(70.0, 25.0), (66.0, 26.0), (68.0, 25.6)]
    assert curve_value(curve, 67.0) == pytest.approx(25.8)
    assert curve_value(curve, 70.0) == pytest.approx(25.0)
    assert curve_value(curve, 71.0) is None

def test_gaps_and_main(tmp_path, capsys):
    results = {"RealSR": {"c0": {"musiq": 66.0, "psnr": 26.0}, "c1": {"musiq": 70.0, "psnr": 25.0},
                          "p": {"musiq": 68.0, "psnr": 26.0}, "far": {"musiq": 60.0, "psnr": 27.0}}}
    rows = gaps(results, ["c0", "c1"], ["p", "far"], "musiq", "psnr")["RealSR"]
    assert rows[0][4] == pytest.approx(0.5)
    assert rows[1][4] is None

    d = tmp_path / "metrics" / "RealSR"
    d.mkdir(parents=True)
    for name, values in results["RealSR"].items():
        (d / f"results_{name}.json").write_text(json.dumps({"RealSR": list(values.items())}))
    gap_main(["--curve", "c0", "c1", "--points", "p", "--metrics_dir", str(tmp_path / "metrics")])
    assert "+0.5000" in capsys.readouterr().out
