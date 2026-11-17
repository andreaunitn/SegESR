"""
Fine-structure metrics of SR outputs against the GT: where in frequency the error is, and how well edges survive.

For every image (Y channel, as PSNR in data_tools/metrics.py, 4-pixel border crop):

Frequency bands (octaves of the spatial frequency f, in cycles per pixel; Nyquist = 1/2):
    low [0, 1/16)   midlow [1/16, 1/8)   midhigh [1/8, 1/4)   high [1/4, 1/2]
  psnr_<band>    PSNR of the error restricted to the band. The band MSEs add up to the image MSE (Parseval),
                 so a low psnr_high means a large part of the error is fine detail.
  energy_<band>  spectral energy of the SR images divided by that of the GT images, in the band, summed over the
                 dataset (a per-image ratio is dominated by nearly flat images, where the GT energy is tiny):
                 < 1 = detail missing (too smooth), > 1 = more detail than the GT (invented texture or noise).
  energy_<band>_median   median over the images of the per-image ratio.

Edges (Canny on the 8-bit Y channel, same thresholds for SR and GT):
  edge_f         F-score of SR edges against GT edges, with a matching tolerance of --edge_tolerance pixels.
  edge_psnr      PSNR inside the thin-structure region: GT edges dilated by --edge_tolerance pixels.

The averages over the images are written to <output_dir>/<dataset>/finestruct_<name>.json, in the same format as
data_tools/metrics.py, and shown with: python3 data_tools/results_table.py --prefix finestruct

Usage:
    python data_tools/fine_structure.py --sr_dir preset/datasets/output/seesr/RealSR \
        --gt_dir preset/datasets/test_datasets/RealSR/test_HR --dataset RealSR --name seesr
On the cluster: sbatch cluster/fine_structure.sh (every output folder, CPU only).
"""

import argparse
import json
import os

import cv2
import numpy as np

BANDS = [("low", 0.0, 1 / 16), ("midlow", 1 / 16, 1 / 8), ("midhigh", 1 / 8, 1 / 4), ("high", 1 / 4, np.inf)]
CANNY_THRESHOLDS = (50, 150)
CROP_BORDER = 4

def load_y(path, crop_border=CROP_BORDER):
    """Y channel in [0, 1] (BT.601, as BasicSR's rgb2ycbcr with y_only=True), with the border cropped."""

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = bgr[..., ::-1].astype(np.float64) / 255.0
    y = (16.0 + 65.481 * rgb[..., 0] + 128.553 * rgb[..., 1] + 24.966 * rgb[..., 2]) / 255.0
    if crop_border:
        y = y[crop_border:-crop_border, crop_border:-crop_border]
    return y

def radial_frequency(shape):
    """Spatial frequency |f| (cycles per pixel) of each coefficient of np.fft.fft2 for an image of this shape."""

    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    return np.sqrt(fy ** 2 + fx ** 2)

def band_metrics(sr, gt):
    """Per-band PSNR of the error and SR/GT spectral energy ratio. The band MSEs sum to the image MSE."""

    n = sr.size
    freq = radial_frequency(sr.shape)
    err_power = np.abs(np.fft.fft2(sr - gt)) ** 2 / n ** 2      # Parseval: sum = mean((sr - gt) ** 2)
    sr_power = np.abs(np.fft.fft2(sr)) ** 2
    gt_power = np.abs(np.fft.fft2(gt)) ** 2

    results = {}
    for name, lo, hi in BANDS:
        band = (freq >= lo) & (freq < hi)
        mse = err_power[band].sum()
        results[f"psnr_{name}"] = 10.0 * np.log10(1.0 / max(mse, 1e-12))
        sr_energy, gt_energy = sr_power[band].sum() / n, gt_power[band].sum() / n
        results[f"energy_{name}"] = sr_energy / max(gt_energy, 1e-12)
        results[f"_sr_energy_{name}"], results[f"_gt_energy_{name}"] = sr_energy, gt_energy
    return results

def canny(y):
    return cv2.Canny(np.clip(np.round(y * 255.0), 0, 255).astype(np.uint8), *CANNY_THRESHOLDS) > 0

def distance_to(mask):
    """Distance of every pixel to the nearest True pixel of mask (large everywhere when mask is empty)."""

    if not mask.any():
        return np.full(mask.shape, np.inf)
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)

def edge_metrics(sr, gt, tolerance=2):
    """Edge F-score (matching within `tolerance` px) and PSNR inside the dilated GT edges."""

    sr_edges, gt_edges = canny(sr), canny(gt)
    if not gt_edges.any() and not sr_edges.any():
        f_score = 1.0
    else:
        precision = (distance_to(gt_edges)[sr_edges] <= tolerance).mean() if sr_edges.any() else 0.0
        recall = (distance_to(sr_edges)[gt_edges] <= tolerance).mean() if gt_edges.any() else 0.0
        f_score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    region = distance_to(gt_edges) <= tolerance
    if region.any():
        mse = np.mean((sr[region] - gt[region]) ** 2)
        edge_psnr = 10.0 * np.log10(1.0 / max(mse, 1e-12))
    else:
        edge_psnr = float("nan")
    return {"edge_f": float(f_score), "edge_psnr": float(edge_psnr)}

def image_metrics(sr, gt, edge_tolerance=2):
    if sr.shape != gt.shape:
        raise ValueError(f"Image shapes are different: {sr.shape}, {gt.shape}")
    return {**band_metrics(sr, gt), **edge_metrics(sr, gt, edge_tolerance)}

def evaluate(sr_dir, gt_dir, edge_tolerance=2):
    """Metrics over the images of sr_dir (matched to gt_dir by file name): PSNRs and edge metrics are averaged,
    energy_<band> is the ratio of the summed energies, energy_<band>_median the median of the per-image ratios."""

    names = sorted(name for name in os.listdir(sr_dir) if not name.startswith("."))
    if not names:
        raise SystemExit(f"No images in {sr_dir}")
    per_image = [image_metrics(load_y(os.path.join(sr_dir, name)), load_y(os.path.join(gt_dir, name)), edge_tolerance)
                 for name in names]
    return aggregate(per_image), len(names)

def aggregate(per_image):
    """Dataset-level metrics from the per-image ones (see evaluate)."""

    values = lambda key: np.array([m[key] for m in per_image], dtype=np.float64)
    results = {}
    for key in per_image[0]:
        if key.startswith("_"):
            continue
        if key.startswith("energy_"):
            band = key[len("energy_"):]
            results[key] = float(values(f"_sr_energy_{band}").sum() / max(values(f"_gt_energy_{band}").sum(), 1e-12))
            results[f"{key}_median"] = float(np.median(values(key)))
        else:
            results[key] = float(np.nanmean(values(key)))
    return results

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Frequency-band and edge metrics of SR outputs against the GT.")
    parser.add_argument("--sr_dir", required=True, help="SR output folder (the sample subfolder is added, as in metrics.py).")
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--sample", default="sample00")
    parser.add_argument("--edge_tolerance", type=float, default=2, help="Edge matching tolerance in pixels.")
    parser.add_argument("--output_dir", default="metrics", help="Results go to <output_dir>/<dataset>/finestruct_<name>.json")
    return parser.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    results, count = evaluate(os.path.join(args.sr_dir, args.sample), args.gt_dir, args.edge_tolerance)

    save_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"finestruct_{args.name}.json")
    with open(save_path, "w") as f:
        json.dump({args.dataset: list(results.items())}, f, indent=4)
    print(f"{args.name} on {args.dataset} ({count} images): " + "  ".join(f"{k}={v:.4f}" for k, v in results.items()))
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    main()
