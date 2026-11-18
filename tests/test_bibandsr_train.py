"""
Tests for the parts of the BiBandSR training (bibandsr/train.py) that need no model.

Run with: pytest tests/test_bibandsr_train.py
"""

import pytest

from bibandsr.train import FIXED_ARGS, check_config, parse_args, psnr

def test_fixed_args_exist():
    args = parse_args(["--loss", "split", "--output_dir", "x"])
    assert all(hasattr(args, k) for k in FIXED_ARGS)
    assert args.cutoff == 0.125

def test_resume_refuses_changed_settings(tmp_path):
    check_config(tmp_path, parse_args(["--loss", "split", "--output_dir", str(tmp_path)]))
    # Same settings (and a different time budget, which may change): accepted
    check_config(tmp_path, parse_args(["--loss", "split", "--output_dir", str(tmp_path), "--stop_after_minutes", "10"]))
    with pytest.raises(SystemExit):
        check_config(tmp_path, parse_args(["--loss", "pisasr", "--output_dir", str(tmp_path)]))

def test_psnr_of_images_in_minus_one_one():
    assert psnr(4.0 / 100) == pytest.approx(20.0)
