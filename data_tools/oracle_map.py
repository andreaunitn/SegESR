"""
Oracle of a spatially varying fidelity-realism setting (step M.0). From the outputs of one model at several global
settings (e.g. lambda_sem of PiSA-SR or FreqOrtho-SR), it builds for every image the composite that takes, in each
tile, the setting with the lowest error against the GT. This bounds what a per-region map could gain over the best
global setting. The composite is then scored like any SR output (data_tools/metrics.py, data_tools/fine_structure.py).

The choice is made on tiles of --tile HR pixels, by the MSE of the Y channel (as the PSNR of metrics.py), and blended
smoothly: the one-hot choice map is upsampled bilinearly, so neighbouring tiles cross-fade instead of meeting at a
seam. Blending the outputs approximates, in pixel space, a setting that the method would vary in the latent.

--control shuffle keeps how many tiles each setting gets in an image but places them at random: if this control
gains as much as the oracle, the gain comes from mixing the settings, not from choosing where.

Usage:
    python data_tools/oracle_map.py --gt_dir preset/datasets/test_datasets/RealSR/test_HR \
        --candidates preset/datasets/output/freqortho_ls0/RealSR preset/datasets/output/freqortho_ls1/RealSR \
        --output_dir preset/datasets/output/freqortho_oracle64/RealSR --tile 64
Writes <output_dir>/sample00/<image>.png and <output_dir>/choice.json (share of the tiles given to each candidate).
"""

import argparse
import json
import os

import cv2
import numpy as np

def load_rgb(path):
    """RGB in [0, 1], float64."""

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return bgr[..., ::-1].astype(np.float64) / 255.0

def luma(rgb):
    """Y channel (BT.601), as data_tools/fine_structure.py and metrics.py."""

    return (16.0 + 65.481 * rgb[..., 0] + 128.553 * rgb[..., 1] + 24.966 * rgb[..., 2]) / 255.0

def tile_errors(candidates, gt, tile):
    """MSE of the Y channel of each candidate in each tile: array (K, rows, cols). Edge tiles may be smaller."""

    gt_y = luma(gt)
    h, w = gt_y.shape
    rows, cols = -(-h // tile), -(-w // tile)
    errors = np.empty((len(candidates), rows, cols))
    for k, image in enumerate(candidates):
        sq = (luma(image) - gt_y) ** 2
        for i in range(rows):
            for j in range(cols):
                errors[k, i, j] = sq[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile].mean()
    return errors

def blend(candidates, choice, tile):
    """Composite of the candidates from a choice map (rows, cols) of candidate indices: the one-hot maps are
    upsampled bilinearly to the tile grid (rows * tile, cols * tile) and cropped to the image, so each value sits at
    its tile centre and the weights sum to 1 everywhere."""

    h, w = candidates[0].shape[:2]
    rows, cols = choice.shape
    result = np.zeros_like(candidates[0])
    for k, image in enumerate(candidates):
        one_hot = (choice == k).astype(np.float32)
        weight = cv2.resize(one_hot, (cols * tile, rows * tile), interpolation=cv2.INTER_LINEAR)[:h, :w]
        result += weight.astype(np.float64)[..., None] * image
    return np.clip(result, 0.0, 1.0)

def oracle_choice(errors, control=None, rng=None):
    """Index of the candidate with the lowest error in each tile; with control='shuffle' the same choices are
    permuted at random over the tiles."""

    choice = errors.argmin(axis=0)
    if control == "shuffle":
        flat = choice.ravel().copy()
        rng.shuffle(flat)
        choice = flat.reshape(choice.shape)
    return choice

def run(candidate_dirs, gt_dir, output_dir, tile, control=None, sample="sample00", seed=0):
    names = sorted(name for name in os.listdir(gt_dir) if not name.startswith("."))
    if not names:
        raise SystemExit(f"No images in {gt_dir}")
    for d in candidate_dirs:
        missing = [n for n in names if not os.path.exists(os.path.join(d, sample, n))]
        if missing:
            raise SystemExit(f"{d}/{sample} lacks {len(missing)} images, e.g. {missing[0]}")

    rng = np.random.default_rng(seed)
    save_dir = os.path.join(output_dir, sample)
    os.makedirs(save_dir, exist_ok=True)
    counts = np.zeros(len(candidate_dirs), dtype=np.int64)
    for name in names:
        gt = load_rgb(os.path.join(gt_dir, name))
        candidates = [load_rgb(os.path.join(d, sample, name)) for d in candidate_dirs]
        for d, image in zip(candidate_dirs, candidates):
            if image.shape != gt.shape:
                raise ValueError(f"{d}/{sample}/{name}: shape {image.shape}, GT {gt.shape}")
        choice = oracle_choice(tile_errors(candidates, gt, tile), control, rng)
        counts += np.bincount(choice.ravel(), minlength=len(candidate_dirs))
        composite = blend(candidates, choice, tile)
        cv2.imwrite(os.path.join(save_dir, name), (composite[..., ::-1] * 255.0).round().astype(np.uint8))

    shares = {d: float(c / counts.sum()) for d, c in zip(candidate_dirs, counts)}
    with open(os.path.join(output_dir, "choice.json"), "w") as f:
        json.dump({"tile": tile, "control": control, "images": len(names), "tile_share": shares}, f, indent=4)
    return shares

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Per-tile oracle composite of SR outputs at several global settings.")
    parser.add_argument("--candidates", nargs="+", required=True,
                        help="SR output folders of the same model and dataset (each with the sample subfolder).")
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tile", type=int, default=64, help="Tile size in HR pixels.")
    parser.add_argument("--control", choices=["shuffle"], default=None)
    parser.add_argument("--sample", default="sample00")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if len(args.candidates) < 2:
        parser.error("at least two candidates are needed")
    return args

def main(argv=None):
    args = parse_args(argv)
    shares = run(args.candidates, args.gt_dir, args.output_dir, args.tile, args.control, args.sample, args.seed)
    print(f"{args.output_dir} (tile {args.tile}, control {args.control}): "
          + "  ".join(f"{os.path.basename(os.path.dirname(d.rstrip('/')))}={s:.2f}" for d, s in shares.items()))

if __name__ == "__main__":
    main()
