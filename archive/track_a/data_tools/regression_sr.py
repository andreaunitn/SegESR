"""
x4 regression SR baselines (trained with a pixel loss, no GAN or diffusion), saved like an SR output. Step T.0 uses
them as the source of the band below the LR Nyquist limit (data_tools/band_scale.py --low_from): can a model trained
for fidelity reconstruct that band better than a diffusion model?

Models (RRDBNet, data_tools/rrdbnet.py), downloaded to --model_dir on first use:
  realesrnet   RealESRNet_x4plus (Real-ESRGAN degradations, L1 loss)
  bsrnet       BSRNet (BSRGAN degradations, L1 loss)

Usage:
    python data_tools/regression_sr.py --model realesrnet --lr_dir preset/datasets/test_datasets/RealSR/test_LR \
        --output_dir preset/datasets/output/realesrnet/RealSR
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    from data_tools.rrdbnet import load_rrdbnet
except ImportError:                  # run as a script: python data_tools/regression_sr.py
    from rrdbnet import load_rrdbnet

MODELS = {
    "realesrnet": ("RealESRNet_x4plus.pth",
                   "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"),
    "bsrnet": ("BSRNet.pth", "https://github.com/cszn/KAIR/releases/download/v1.0/BSRNet.pth"),
}
SCALE = 4

def model_path(name, model_dir):
    filename, url = MODELS[name]
    path = os.path.join(model_dir, filename)
    if not os.path.isfile(path):
        os.makedirs(model_dir, exist_ok=True)
        print(f"Downloading {url} -> {path}")
        torch.hub.download_url_to_file(url, path)
    return path

@torch.no_grad()
def upscale(model, image, tile=0, overlap=16):
    """x4 output of an (1, 3, H, W) tensor in [0, 1]. With tile > 0 the LR is processed in overlapping tiles of
    tile x tile pixels, and the outputs are averaged where they overlap."""

    if tile <= 0 or max(image.shape[-2:]) <= tile:
        return model(image).clamp(0, 1)
    _, c, h, w = image.shape
    out = image.new_zeros(1, c, h * SCALE, w * SCALE)
    weight = torch.zeros_like(out)
    step = tile - overlap
    ys = sorted({min(y, h - tile) for y in range(0, max(h - overlap, 1), step)}) if h > tile else [0]
    xs = sorted({min(x, w - tile) for x in range(0, max(w - overlap, 1), step)}) if w > tile else [0]
    for y in ys:
        for x in xs:
            patch = image[..., y:y + tile, x:x + tile]
            ph, pw = patch.shape[-2:]
            out[..., y * SCALE:(y + ph) * SCALE, x * SCALE:(x + pw) * SCALE] += model(patch)
            weight[..., y * SCALE:(y + ph) * SCALE, x * SCALE:(x + pw) * SCALE] += 1
    return (out / weight).clamp(0, 1)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="x4 regression SR baselines, saved like an SR output.")
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--lr_dir", required=True)
    parser.add_argument("--output_dir", required=True, help="Results are saved to <output_dir>/<sample>/<stem>.png.")
    parser.add_argument("--sample", default="sample00")
    parser.add_argument("--model_dir", default="preset/models/regression")
    parser.add_argument("--tile", type=int, default=512, help="LR tile size (0: whole image).")
    return parser.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_rrdbnet(model_path(args.model, args.model_dir), device)

    out_dir = os.path.join(args.output_dir, args.sample)
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(name for name in os.listdir(args.lr_dir) if not name.startswith("."))
    for name in names:
        lr = np.asarray(Image.open(os.path.join(args.lr_dir, name)).convert("RGB"), dtype=np.float32) / 255.0
        x = torch.from_numpy(lr).permute(2, 0, 1)[None].to(device)
        sr = upscale(model, x, args.tile)[0].permute(1, 2, 0).cpu().numpy()
        Image.fromarray(np.round(sr * 255.0).astype(np.uint8)).save(os.path.join(out_dir, f"{Path(name).stem}.png"))
    print(f"{args.model}: {len(names)} images from {args.lr_dir} -> {out_dir}")

if __name__ == "__main__":
    main()
