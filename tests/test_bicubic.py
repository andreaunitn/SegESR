"""
Tests for the bicubic upsampling of LR images (data_tools/bicubic.py).

Run with: pytest tests/test_bicubic.py
"""

import numpy as np
from PIL import Image

from data_tools.bicubic import main

def test_main_writes_upsampled_pngs(tmp_path):
    lr_dir = tmp_path / "lr"
    lr_dir.mkdir()
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 256, (10, 12, 3), dtype=np.uint8)).save(lr_dir / "a.png")
    Image.fromarray(np.full((8, 8, 3), 100, dtype=np.uint8)).save(lr_dir / "b.jpg")

    main(["--lr_dir", str(lr_dir), "--output_dir", str(tmp_path / "out")])

    out = tmp_path / "out" / "sample00"
    assert sorted(p.name for p in out.iterdir()) == ["a.png", "b.png"]
    assert Image.open(out / "a.png").size == (48, 40)
    assert np.all(np.asarray(Image.open(out / "b.png")) == 100)       # a flat image stays flat
