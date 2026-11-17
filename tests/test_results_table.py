"""
Tests for the metrics table (data_tools/results_table.py).

Run with: pytest tests/test_results_table.py
"""

import json

import pytest

from data_tools.results_table import load_results, main, make_table, matches

def write_result(metrics_dir, dataset, name, values):
    path = metrics_dir / dataset / f"results_{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({dataset: [[metric, value] for metric, value in values.items()]}))

@pytest.fixture
def metrics_dir(tmp_path):
    write_result(tmp_path, "RealSR", "seesr", {"psnr": 25.35, "lpips": 0.3001, "niqe": 5.41})
    write_result(tmp_path, "RealSR", "segesr_checkpoint-15000", {"psnr": 25.90, "lpips": 0.3100, "niqe": 5.20})
    write_result(tmp_path, "DRealSR", "seesr", {"psnr": 28.33, "lpips": 0.3164})
    return tmp_path

def test_load_results(metrics_dir):
    results = load_results(str(metrics_dir))

    assert set(results) == {"RealSR", "DRealSR"}
    assert set(results["RealSR"]) == {"seesr", "segesr_checkpoint-15000"}
    assert results["RealSR"]["seesr"]["lpips"] == pytest.approx(0.3001)

def test_best_value_follows_metric_direction(metrics_dir):
    rows = load_results(str(metrics_dir))["RealSR"]
    table = make_table(rows, "markdown")
    seesr = next(line for line in table.splitlines() if line.startswith("| seesr "))
    segesr = next(line for line in table.splitlines() if line.startswith("| segesr_"))

    assert "**25.90**" in segesr and "**25.35**" not in seesr      # PSNR: higher is better
    assert "**0.3001**" in seesr and "**0.3100**" not in segesr    # LPIPS: lower is better
    assert "**5.20**" in segesr                                    # NIQE: lower is better

def test_columns_in_fixed_order_and_missing_values(metrics_dir):
    write_result(metrics_dir, "RealSR", "extra", {"topiq_nr": 0.6, "psnr": 24.0})
    table = make_table(load_results(str(metrics_dir))["RealSR"], "text")
    header = table.splitlines()[0].split()

    assert header == ["Model", "PSNR↑", "LPIPS↓", "NIQE↓", "TOPIQ_NR"]   # unknown metrics last, not ranked
    extra = next(line for line in table.splitlines() if line.startswith("extra"))
    assert extra.split()[2:4] == ["-", "-"]

def test_single_row_has_no_best(metrics_dir):
    table = make_table(load_results(str(metrics_dir))["DRealSR"], "markdown")

    assert "**" not in table

def test_name_filters():
    assert matches("segesr_checkpoint-15000", [])
    assert matches("segesr_checkpoint-15000", ["checkpoint-15000"])
    assert matches("segesr_checkpoint-15000", ["segesr_*"])
    assert not matches("seesr_finetuned_checkpoint-15000", ["segesr_*"])
    assert not matches("seesr", ["norefresh"])

def test_main_filters_names_and_datasets(metrics_dir, capsys):
    main(["segesr", "--datasets", "RealSR", "--metrics_dir", str(metrics_dir)])
    out = capsys.readouterr().out

    assert "== RealSR" in out and "DRealSR" not in out
    assert "segesr_checkpoint-15000" in out and "\nseesr " not in out

def test_main_without_matches_exits(metrics_dir):
    with pytest.raises(SystemExit):
        main(["nothing-like-this", "--metrics_dir", str(metrics_dir)])
