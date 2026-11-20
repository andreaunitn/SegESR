"""
Tests for the gap of a model from a curve of global settings (data_tools/curve_gap.py).

Run with: pytest tests/test_curve_gap.py
"""

import json

import pytest

from data_tools.curve_gap import curve_value, gaps
from data_tools.curve_gap import main as gap_main

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
