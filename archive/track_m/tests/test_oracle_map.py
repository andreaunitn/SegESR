"""
Tests for the per-tile oracle (data_tools/oracle_map.py). Archived with Track M; the tests of the curve gap are
in tests/test_curve_gap.py.

Run from the tag pre-archive-m: pytest tests/test_oracle_map.py
"""

import json

import cv2
import numpy as np
import pytest

from data_tools.oracle_map import blend, budget_price, luma, main, mean_value, oracle_choice, tile_errors

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

def test_price_moves_the_choice_to_higher_values():
    errors = np.array([[[1.0, 1.0]], [[2.0, 1.5]]])          # candidate 0 better in both tiles, by 1 and by 0.5
    assert (oracle_choice(errors) == 0).all()
    assert oracle_choice(errors, price=0.75, values=[0, 1]).tolist() == [[0, 1]]
    assert (oracle_choice(errors, price=2.0, values=[0, 1]) == 1).all()

def test_budget_price_is_the_smallest_that_reaches_the_budget():
    rng = np.random.default_rng(5)
    errors = [rng.random((3, 4, 4)) for _ in range(5)]
    values = [0, 0.5, 1.5]
    for budget in (0.3, 0.8, 1.2, 1.5):
        price = budget_price(errors, values, budget)
        assert mean_value(errors, values, price) >= budget
        assert price == 0 or mean_value(errors, values, price * (1 - 1e-6)) < budget
    with pytest.raises(ValueError):
        budget_price(errors, values, 2.0)

def test_main_with_budgets(tmp_path):
    rng = np.random.default_rng(6)
    gt_dir = tmp_path / "gt"
    for i in range(3):
        gt = make_gt(seed=20 + i)
        a, b = split_candidates(gt, rng)
        write(gt_dir / f"img{i}.png", gt)
        write(tmp_path / "a" / "sample00" / f"img{i}.png", a)
        write(tmp_path / "b" / "sample00" / f"img{i}.png", b)

    for control in ([], ["--control", "shuffle"]):
        out = tmp_path / ("shuffle" if control else "oracle")
        main(["--candidates", str(tmp_path / "a"), str(tmp_path / "b"), "--values", "0", "1", "--budgets", "0.5", "1",
              "--gt_dir", str(gt_dir), "--output_dir", str(out) + "_b{budget}", "--tile", "32", *control])
        for budget, tag in ((0.5, "0.5"), (1.0, "1")):
            info = json.loads((tmp_path / f"{out.name}_b{tag}" / "choice.json").read_text())
            assert info["mean_value"] >= budget and info["budget"] == budget
            assert len(list((tmp_path / f"{out.name}_b{tag}" / "sample00").glob("*.png"))) == 3
        assert info["price"] > 0 and info["tile_share"][str(tmp_path / "b")] == 1.0
    with pytest.raises(SystemExit):
        main(["--candidates", str(tmp_path / "a"), str(tmp_path / "b"), "--values", "0", "1", "--budgets", "0.5",
              "--gt_dir", str(gt_dir), "--output_dir", str(tmp_path / "no_placeholder")])
