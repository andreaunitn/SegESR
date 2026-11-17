"""
Collects the metrics written by `data_tools/metrics.py` (metrics/<dataset>/results_<name>.json) into one table
per dataset, with the best value of each metric marked.

Usage (standard library only, no container needed):
    python3 data_tools/results_table.py                              # every result
    python3 data_tools/results_table.py seesr checkpoint-15000       # names containing "seesr" or "checkpoint-15000"
    python3 data_tools/results_table.py "segesr_*" --datasets RealSR --format markdown

From your machine: cluster/hpc run python3 data_tools/results_table.py [args]
"""

import argparse
import fnmatch
import glob
import json
import os

# Known metrics in display order, with their direction (True: higher is better) and decimals.
# Metrics not listed here are shown after these, in alphabetical order, without a best value.
METRICS = {
    "psnr": (True, 2),
    "ssim": (True, 4),
    "lpips": (False, 4),
    "dists": (False, 4),
    "fid": (False, 2),
    "niqe": (False, 2),
    "clipiqa": (True, 4),
    "musiq": (True, 2),
    "maniqa": (True, 4),
}

def load_results(metrics_dir):
    """Returns {dataset: {name: {metric: value}}} for every metrics/<dataset>/results_<name>.json."""

    results = {}
    for path in sorted(glob.glob(os.path.join(metrics_dir, "*", "results_*.json"))):
        with open(path) as f:
            content = json.load(f)
        name = os.path.basename(path)[len("results_"):-len(".json")]
        for dataset, values in content.items():
            results.setdefault(dataset, {})[name] = {metric: float(value) for metric, value in values}
    return results

def matches(name, patterns):
    """A name is kept if it matches any pattern: a glob (with * ? [) or else a substring."""

    if not patterns:
        return True
    for pattern in patterns:
        if any(c in pattern for c in "*?["):
            if fnmatch.fnmatch(name, pattern):
                return True
        elif pattern in name:
            return True
    return False

def metric_order(rows):
    present = {metric for values in rows.values() for metric in values}
    known = [metric for metric in METRICS if metric in present]
    return known + sorted(present - set(METRICS))

def best_values(rows, metrics):
    """Best value of each known metric over the rows (None when fewer than 2 rows have it)."""

    best = {}
    for metric in metrics:
        values = [values[metric] for values in rows.values() if metric in values]
        if metric in METRICS and len(values) >= 2:
            best[metric] = max(values) if METRICS[metric][0] else min(values)
    return best

def format_value(metric, value):
    return f"{value:.{METRICS.get(metric, (True, 4))[1]}f}"

def header_label(metric):
    if metric not in METRICS:
        return metric.upper()
    return metric.upper() + ("↑" if METRICS[metric][0] else "↓")

def make_table(rows, fmt):
    """Text or markdown table of {name: {metric: value}}; the best value of each column is marked."""

    metrics = metric_order(rows)
    best = best_values(rows, metrics)

    table = [["Model"] + [header_label(metric) for metric in metrics]]
    for name in sorted(rows):
        cells = [name]
        for metric in metrics:
            if metric not in rows[name]:
                cells.append("-")
                continue
            cell = format_value(metric, rows[name][metric])
            if metric in best and format_value(metric, best[metric]) == cell:
                cell = f"**{cell}**" if fmt == "markdown" else f"{cell}*"
            cells.append(cell)
        table.append(cells)

    if fmt == "markdown":
        lines = ["| " + " | ".join(row) + " |" for row in table]
        lines.insert(1, "|" + "|".join("---" for _ in table[0]) + "|")
        return "\n".join(lines)

    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    lines = ["  ".join(cell.ljust(width) if i == 0 else cell.rjust(width) for i, (cell, width) in enumerate(zip(row, widths)))
             for row in table]
    lines.insert(1, "-" * len(lines[0]))
    return "\n".join(lines)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Show the SegESR test metrics as one table per dataset.")
    parser.add_argument("names", nargs="*", help="Keep only the results whose name contains one of these strings "
                        "(or matches one of these globs, e.g. 'segesr_*'). Default: all.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Datasets to show (default: all).")
    parser.add_argument("--metrics_dir", default="metrics", help="Folder written by data_tools/metrics.py.")
    parser.add_argument("--format", choices=["text", "markdown"], default="text",
                        help="text for the terminal (best value marked with *), markdown for docs/EXPERIMENTS.md (best in bold).")
    return parser.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    results = load_results(args.metrics_dir)
    if not results:
        raise SystemExit(f"No results found in {args.metrics_dir}/<dataset>/results_<name>.json")

    shown = 0
    for dataset in sorted(results):
        if args.datasets and dataset not in args.datasets:
            continue
        rows = {name: values for name, values in results[dataset].items() if matches(name, args.names)}
        if not rows:
            continue
        if shown:
            print()
        print(f"## {dataset}" if args.format == "markdown" else f"== {dataset}")
        print(make_table(rows, args.format))
        shown += 1

    if not shown:
        raise SystemExit("No results match the given names / datasets.")

if __name__ == "__main__":
    main()
