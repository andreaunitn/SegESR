"""
Bicubic upsampling of the LR images, saved like an SR output: how much of the GT the LR itself holds, band by band
(score it with data_tools/fine_structure.py). The upsampling is the one test.py uses for the color fix
(PIL bicubic to `scale` x the LR size).

Usage:
    python data_tools/bicubic.py --lr_dir preset/datasets/test_datasets/RealSR/test_LR \
        --output_dir preset/datasets/output/bicubic/RealSR
"""

import argparse
import os
from pathlib import Path

from PIL import Image

def upsample(image, scale):
    return image.resize((image.size[0] * scale, image.size[1] * scale), Image.BICUBIC)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Bicubic upsampling of LR images, saved like an SR output.")
    parser.add_argument("--lr_dir", required=True)
    parser.add_argument("--output_dir", required=True, help="Results are saved to <output_dir>/<sample>/<stem>.png.")
    parser.add_argument("--sample", default="sample00")
    parser.add_argument("--scale", type=int, default=4)
    return parser.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    out_dir = os.path.join(args.output_dir, args.sample)
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(name for name in os.listdir(args.lr_dir) if not name.startswith("."))
    for name in names:
        image = Image.open(os.path.join(args.lr_dir, name)).convert("RGB")
        upsample(image, args.scale).save(os.path.join(out_dir, f"{Path(name).stem}.png"))
    print(f"{len(names)} images from {args.lr_dir} -> {out_dir} (x{args.scale} bicubic)")

if __name__ == "__main__":
    main()
