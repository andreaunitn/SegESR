"""
Tests for the VAE ceiling script (data_tools/vae_ceiling.py), with a tiny random VAE.

Run with: pytest tests/test_vae_ceiling.py
"""

import numpy as np
import pytest
import torch
from diffusers import AutoencoderKL
from PIL import Image

from data_tools.vae_ceiling import main, reconstruct

@pytest.fixture
def tiny_vae():
    torch.manual_seed(0)
    vae = AutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=4,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(8, 16), layers_per_block=1, norm_num_groups=8,
    )
    return vae.eval().requires_grad_(False)

def random_image(size=(30, 22), seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))

def test_reconstruction_keeps_size(tiny_vae):
    image = random_image((30, 22))      # not a multiple of the VAE factor (2): padded, then cropped back
    out, encode_mse, final_mse = reconstruct(tiny_vae, image, mode="encode")

    assert out.size == image.size and out.mode == "RGB"
    assert final_mse == encode_mse

def test_optimize_does_not_get_worse(tiny_vae):
    image = random_image((16, 16), seed=1)
    _, encode_mse, final_mse = reconstruct(tiny_vae, image, mode="optimize", steps=20, lr=0.05)

    assert final_mse <= encode_mse
    assert final_mse < encode_mse * 0.99      # 20 steps on a random VAE must reduce the error

def test_main_writes_one_png_per_gt(tiny_vae, tmp_path, monkeypatch):
    gt_dir, out_dir = tmp_path / "gt", tmp_path / "out"
    gt_dir.mkdir()
    for i in range(2):
        random_image((16, 16), seed=i).save(gt_dir / f"img{i}.png")
    monkeypatch.setattr("data_tools.vae_ceiling.load_vae", lambda path, device: tiny_vae.to(device))

    main(["--gt_dir", str(gt_dir), "--output_dir", str(out_dir)])

    written = sorted(p.name for p in (out_dir / "sample00").iterdir())
    assert written == ["img0.png", "img1.png"]
    assert Image.open(out_dir / "sample00" / "img0.png").size == (16, 16)
