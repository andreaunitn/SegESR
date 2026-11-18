"""
Collects the outputs of an external SR model (e.g. the official PiSA-SR and OSEDiff test scripts) into the layout of
test.py, so that data_tools/metrics.py and data_tools/fine_structure.py can score them:
    <output_dir>/<sample>/<stem>.png, each exactly `scale` x the size of its LR image.

Some scripts round the size down to a multiple of 8; those outputs are resized back with bicubic interpolation,
as test.py does. Files that are not images (e.g. saved prompts) are ignored.

Usage:
    python data_tools/collect_sr.py --src_dir preset/datasets/output/_raw/pisasr/RealSR \
        --lr_dir preset/datasets/test_datasets/RealSR/test_LR --output_dir preset/datasets/output/pisasr/RealSR
"""

import argparse
import os
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def images(folder):
    """{stem: file name} of the images in folder."""

    return {Path(name).stem: name for name in os.listdir(folder) if Path(name).suffix.lower() in IMAGE_EXTENSIONS}

def collect(src_dir, lr_dir, out_dir, scale=4):
    """Copies the outputs to out_dir at `scale` x the LR size. Returns (number of images, number resized);
    exits if an LR image has no output."""

    lr_names, outputs = images(lr_dir), images(src_dir)
    missing = sorted(set(lr_names) - set(outputs))
    if missing:
        raise SystemExit(f"{len(missing)} LR images have no output in {src_dir}, e.g. {missing[:3]}")

    os.makedirs(out_dir, exist_ok=True)
    resized = 0
    for stem in sorted(lr_names):
        width, height = Image.open(os.path.join(lr_dir, lr_names[stem])).size
        image = Image.open(os.path.join(src_dir, outputs[stem])).convert("RGB")
        target = (width * scale, height * scale)
        if image.size != target:
            image = image.resize(target, Image.BICUBIC)
            resized += 1
        image.save(os.path.join(out_dir, f"{stem}.png"))
    return len(lr_names), resized

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Collect external SR outputs into the test.py layout.")
    parser.add_argument("--src_dir", required=True, help="Folder with the outputs of the external script.")
    parser.add_argument("--lr_dir", required=True, help="LR folder: gives the names and the target sizes.")
    parser.add_argument("--output_dir", required=True, help="Results are saved to <output_dir>/<sample>/<stem>.png.")
    parser.add_argument("--sample", default="sample00")
    parser.add_argument("--scale", type=int, default=4)
    return parser.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    out_dir = os.path.join(args.output_dir, args.sample)
    count, resized = collect(args.src_dir, args.lr_dir, out_dir, args.scale)
    print(f"{count} images from {args.src_dir} -> {out_dir} ({resized} resized to x{args.scale} the LR size)")

if __name__ == "__main__":
    main()
