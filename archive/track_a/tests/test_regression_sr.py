"""
Tests for the regression SR baselines (data_tools/rrdbnet.py, data_tools/regression_sr.py).

Run with: pytest tests/test_regression_sr.py
"""

import re

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from data_tools import regression_sr
from data_tools.rrdbnet import RRDBNet, convert_state_dict

def nearest_x4(x):
    return F.interpolate(x, scale_factor=4, mode="nearest")

def test_rrdbnet_upscales_by_four():
    model = RRDBNet(num_block=1).eval()
    with torch.no_grad():
        assert model(torch.rand(1, 3, 8, 12)).shape == (1, 3, 32, 48)

def test_old_esrgan_keys_are_converted():
    state = RRDBNet(num_block=2).state_dict()
    old_names = [(r"^body\.", "RRDB_trunk."), (r"\.rdb(\d)\.", r".RDB\1."), (r"^conv_body\.", "trunk_conv."),
                 (r"^conv_up1\.", "upconv1."), (r"^conv_up2\.", "upconv2."), (r"^conv_hr\.", "HRconv.")]
    old = {}
    for name, value in state.items():
        for new, previous in old_names:
            name = re.sub(new, previous, name)
        old[name] = value
    assert any(re.match(r"RRDB_trunk\.0\.RDB1\.conv1", k) for k in old)

    converted = convert_state_dict(old)
    assert set(converted) == set(state)
    RRDBNet(num_block=2).load_state_dict(converted, strict=True)
    assert set(convert_state_dict({"params_ema": state})) == set(state)

def test_tiled_upscale_matches_whole_image():
    image = torch.rand(1, 3, 45, 70)
    whole = regression_sr.upscale(nearest_x4, image, tile=0)
    tiled = regression_sr.upscale(nearest_x4, image, tile=32, overlap=8)
    assert torch.allclose(whole, tiled)

def test_main_writes_x4_pngs(tmp_path, monkeypatch):
    lr_dir = tmp_path / "lr"
    lr_dir.mkdir()
    Image.fromarray(np.random.default_rng(0).integers(0, 256, (10, 12, 3), dtype=np.uint8)).save(lr_dir / "a.jpg")
    monkeypatch.setattr(regression_sr, "model_path", lambda name, model_dir: "unused")
    monkeypatch.setattr(regression_sr, "load_rrdbnet", lambda path, device: nearest_x4)

    regression_sr.main(["--model", "bsrnet", "--lr_dir", str(lr_dir), "--output_dir", str(tmp_path / "out")])

    out = Image.open(tmp_path / "out" / "sample00" / "a.png")
    assert out.size == (48, 40)
    assert np.array_equal(np.asarray(out)[::4, ::4], np.asarray(Image.open(lr_dir / "a.jpg").convert("RGB")))
