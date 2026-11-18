"""
Tests for collecting external SR outputs (data_tools/collect_sr.py).

Run with: pytest tests/test_collect_sr.py
"""

import numpy as np
import pytest
from PIL import Image

from data_tools.collect_sr import main

def save(path, width, height, value=100):
    Image.fromarray(np.full((height, width, 3), value, dtype=np.uint8)).save(path)

def test_outputs_are_renamed_and_resized(tmp_path):
    lr, src = tmp_path / "lr", tmp_path / "src"
    lr.mkdir()
    (src / "txt").mkdir(parents=True)
    save(lr / "a.png", 10, 7)
    save(lr / "b.jpg", 8, 8)
    save(src / "a.png", 40, 24)          # rounded down to a multiple of 8: 28 -> 24
    save(src / "b.png", 32, 32)
    (src / "txt" / "a.txt").write_text("a prompt")

    main(["--src_dir", str(src), "--lr_dir", str(lr), "--output_dir", str(tmp_path / "out")])

    out = tmp_path / "out" / "sample00"
    assert sorted(p.name for p in out.iterdir()) == ["a.png", "b.png"]
    assert Image.open(out / "a.png").size == (40, 28)
    assert Image.open(out / "b.png").size == (32, 32)

def test_missing_output_fails(tmp_path):
    lr, src = tmp_path / "lr", tmp_path / "src"
    lr.mkdir()
    src.mkdir()
    save(lr / "a.png", 10, 10)

    with pytest.raises(SystemExit):
        main(["--src_dir", str(src), "--lr_dir", str(lr), "--output_dir", str(tmp_path / "out")])
