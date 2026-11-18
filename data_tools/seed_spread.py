"""
How much of the error of a stochastic SR model, in each frequency band, comes from the random seed.

test.py --sample_times K writes K samples per image (sample00 ... sample<K-1>), each with different noise. For each
image and each band of data_tools/fine_structure.py (Y channel, 4-pixel crop), the expected squared error of a
sample against the GT splits exactly into
    E ||s - gt||^2  =  ||mean - gt||^2 (systematic part)  +  E ||s - mean||^2 (seed part),
where mean is the expectation over seeds. The frequencies below the LR Nyquist limit (1/8 cycle per pixel for x4)
are determined by the LR, so an ideal model would have no seed part there.

Metrics (averaged over the images, except the shares):
  spread_<band>      PSNR of the seed variance in the band (unbiased estimate over the K samples; no GT needed):
                     high = the samples agree.
  psnr_<band>        PSNR of the error of a single sample (mean over samples), as in fine_structure.py.
  psnr_mean_<band>   PSNR of the error of the mean of the K samples.
  seed_share_<band>  seed variance / expected error of a sample, summed over the dataset (needs the GT): the
                     fraction of the band error that comes from the seed (an estimate: it can slightly exceed 1
                     when the systematic part is close to 0).

The mean of the samples can be saved as an SR output (--mean_dir), e.g. to take its low band in band_scale.py.

Results go to <output_dir>/<dataset>/seedspread_<name>.json:
    python3 data_tools/results_table.py --prefix seedspread

Usage:
    python data_tools/seed_spread.py --sr_dir preset/datasets/output/seesr_s3/RealSR \
        --gt_dir preset/datasets/test_datasets/RealSR/test_HR --dataset RealSR --name seesr_s3 \
        --mean_dir preset/datasets/output/seesr_s3mean/RealSR
"""

import argparse
import json
import os

import cv2
import numpy as np

try:
    from data_tools.fine_structure import BANDS, CROP_BORDER, radial_frequency
except ImportError:                  # run as a script: python data_tools/seed_spread.py
    from fine_structure import BANDS, CROP_BORDER, radial_frequency

def to_y(rgb, crop_border=CROP_BORDER):
    """Y channel (BT.601, as data_tools/fine_structure.py) of an (H, W, 3) RGB image in [0, 1], border cropped."""

    y = (16.0 + 65.481 * rgb[..., 0] + 128.553 * rgb[..., 1] + 24.966 * rgb[..., 2]) / 255.0
    if crop_border:
        y = y[crop_border:-crop_border, crop_border:-crop_border]
    return y

def band_power(x):
    """Mean squared value of x in each band (Parseval: they add up to mean(x ** 2))."""

    power = np.abs(np.fft.fft2(x)) ** 2 / x.size ** 2
    freq = radial_frequency(x.shape)
    return {name: float(power[(freq >= lo) & (freq < hi)].sum()) for name, lo, hi in BANDS}

def image_spread(samples, gt=None):
    """Per-band seed variance and errors of K >= 2 Y-channel samples of one image (see the module docstring)."""

    k = len(samples)
    if k < 2:
        raise ValueError("At least 2 samples are needed.")
    mean = np.mean(samples, axis=0)
    deviations = [band_power(s - mean) for s in samples]
    results = {f"_var_{name}": sum(d[name] for d in deviations) / (k - 1) for name, _, _ in BANDS}
    if gt is not None:
        errors = [band_power(s - gt) for s in samples]
        mean_error = band_power(mean - gt)
        for name, _, _ in BANDS:
            results[f"_err_{name}"] = float(np.mean([e[name] for e in errors]))
            results[f"_mean_err_{name}"] = mean_error[name]
    return results

def psnr(mse):
    return 10.0 * np.log10(1.0 / np.maximum(mse, 1e-12))

def aggregate(per_image):
    """Dataset-level metrics from the per-image results of image_spread."""

    values = lambda key: np.array([m[key] for m in per_image], dtype=np.float64)
    has_gt = "_err_low" in per_image[0]
    results = {}
    for name, _, _ in BANDS:
        results[f"spread_{name}"] = float(np.mean(psnr(values(f"_var_{name}"))))
        if has_gt:
            results[f"psnr_{name}"] = float(np.mean(psnr(values(f"_err_{name}"))))
            results[f"psnr_mean_{name}"] = float(np.mean(psnr(values(f"_mean_err_{name}"))))
            results[f"seed_share_{name}"] = float(values(f"_var_{name}").sum() / max(values(f"_err_{name}").sum(), 1e-12))
    return results

def read_rgb(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image[..., ::-1].astype(np.float64) / 255.0

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Seed part of the SR error in each frequency band.")
    parser.add_argument("--sr_dir", required=True, help="test.py output folder with sample00, sample01, ...")
    parser.add_argument("--gt_dir", default=None, help="GT folder; without it only spread_<band> is computed.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--mean_dir", default=None, help="Also save the mean of the samples to <mean_dir>/sample00/.")
    parser.add_argument("--output_dir", default="metrics", help="Results go to <output_dir>/<dataset>/seedspread_<name>.json")
    return parser.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    sample_dirs = sorted(os.path.join(args.sr_dir, d) for d in os.listdir(args.sr_dir) if d.startswith("sample"))
    if len(sample_dirs) < 2:
        raise SystemExit(f"Need at least 2 sample folders in {args.sr_dir}, found {len(sample_dirs)}.")
    names = sorted(name for name in os.listdir(sample_dirs[0]) if not name.startswith("."))
    if args.mean_dir:
        os.makedirs(os.path.join(args.mean_dir, "sample00"), exist_ok=True)

    per_image = []
    for name in names:
        rgbs = [read_rgb(os.path.join(d, name)) for d in sample_dirs]
        gt = to_y(read_rgb(os.path.join(args.gt_dir, name))) if args.gt_dir else None
        per_image.append(image_spread([to_y(rgb) for rgb in rgbs], gt))
        if args.mean_dir:
            mean = np.round(np.mean(rgbs, axis=0) * 255.0).astype(np.uint8)
            cv2.imwrite(os.path.join(args.mean_dir, "sample00", name), mean[..., ::-1])
    results = aggregate(per_image)

    save_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"seedspread_{args.name}.json")
    with open(save_path, "w") as f:
        json.dump({args.dataset: list(results.items())}, f, indent=4)
    print(f"{args.name} on {args.dataset} ({len(names)} images, {len(sample_dirs)} samples): "
          + "  ".join(f"{k}={v:.4f}" for k, v in results.items()))
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    main()
