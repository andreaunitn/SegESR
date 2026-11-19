"""
How far a model is from the curve of the global settings of another model, at the same perceptual score (step M.0).
For each dataset, the curve is made of the results of the --curve names (e.g. one model at several lambda_sem),
ordered by the --x metric (default MUSIQ) and joined by straight lines. Each --points name is compared with the
curve at its own x: gap = its --y metric (default PSNR) minus the curve's, so a positive gap means better than the
global settings at the same perceptual score. A point outside the x range of the curve has no gap.

Usage (standard library only, no container needed):
    python3 data_tools/curve_gap.py --curve freqortho_ls0 freqortho_ls0.5 freqortho_ls1 freqortho_ls1.5 \
        --points freqortho_oracle64 freqortho_shuffle64 --x musiq --y psnr
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_table import load_results  # noqa: E402

def curve_value(curve, x):
    """Linear interpolation of the (x, y) points of a curve at x; None outside its x range."""

    pts = sorted(curve)
    if not pts or x < pts[0][0] or x > pts[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[0][1]   # a single point, at exactly its x

def gaps(results, curve_names, point_names, x_metric, y_metric):
    """{dataset: [(point, x, y, curve y, gap)]}, for the datasets that have every curve name."""

    table = {}
    for dataset, by_name in sorted(results.items()):
        if not all(name in by_name for name in curve_names):
            continue
        curve = [(by_name[n][x_metric], by_name[n][y_metric]) for n in curve_names]
        rows = []
        for name in point_names:
            if name not in by_name:
                continue
            x, y = by_name[name][x_metric], by_name[name][y_metric]
            ref = curve_value(curve, x)
            rows.append((name, x, y, ref, None if ref is None else y - ref))
        table[dataset] = rows
    return table

def main(argv=None):
    parser = argparse.ArgumentParser(description="Gap of models from a curve of global settings at the same x.")
    parser.add_argument("--curve", nargs="+", required=True)
    parser.add_argument("--points", nargs="+", required=True)
    parser.add_argument("--x", default="musiq")
    parser.add_argument("--y", default="psnr")
    parser.add_argument("--metrics_dir", default="metrics")
    args = parser.parse_args(argv)

    table = gaps(load_results(args.metrics_dir), args.curve, args.points, args.x, args.y)
    if not table:
        raise SystemExit("No dataset has results for every curve name.")
    for dataset, rows in table.items():
        print(f"== {dataset}: {args.y} against the curve at the same {args.x}")
        print(f"{'Model':<28}{args.x:>10}{args.y:>10}{'curve':>10}{'gap':>10}")
        for name, x, y, ref, gap in rows:
            ref_s = f"{ref:10.4f}" if ref is not None else f"{'-':>10}"
            gap_s = f"{gap:+10.4f}" if gap is not None else f"{'outside':>10}"
            print(f"{name:<28}{x:10.4f}{y:10.4f}{ref_s}{gap_s}")
    return table

if __name__ == "__main__":
    main()
