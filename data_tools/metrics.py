import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pyiqa
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Make the vendored `basicsr` package importable without installation
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "third_party"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from basicsr.utils.color_util import rgb2ycbcr_pt

# PSNR / SSIM: same implementation as BasicSR's `basicsr.metrics.psnr_ssim` (not vendored)

def _prepare(img, img2, crop_border, test_y_channel):
    assert img.shape == img2.shape, f"Image shapes are different: {img.shape}, {img2.shape}."

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    return img.to(torch.float64), img2.to(torch.float64)

def calculate_psnr_pt(img, img2, crop_border, test_y_channel=False):
    """PSNR of (N, C, H, W) RGB tensors in [0, 1]."""

    img, img2 = _prepare(img, img2, crop_border, test_y_channel)
    mse = torch.mean((img - img2) ** 2, dim=[1, 2, 3])
    return 10. * torch.log10(1. / (mse + 1e-8))

def _ssim_pth(img, img2):
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    window = torch.from_numpy(window).view(1, 1, 11, 11).expand(img.size(1), 1, 11, 11).to(img.dtype).to(img.device)

    mu1 = F.conv2d(img, window, stride=1, padding=0, groups=img.shape[1])
    mu2 = F.conv2d(img2, window, stride=1, padding=0, groups=img2.shape[1])
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img * img, window, stride=1, padding=0, groups=img.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return ssim_map.mean([1, 2, 3])

def calculate_ssim_pt(img, img2, crop_border, test_y_channel=False):
    """SSIM of (N, C, H, W) RGB tensors in [0, 1]."""

    img, img2 = _prepare(img, img2, crop_border, test_y_channel)
    return _ssim_pth(img * 255., img2 * 255.)

def parse_args():
    parser = argparse.ArgumentParser(description="Compute full-reference and no-reference IQA metrics of SR results.")
    parser.add_argument("--sr_dir", type=str, required=True, help="Output folder of test.py (containing 'sample00/').")
    parser.add_argument("--gt_dir", type=str, default=None, help="GT folder. If omitted, only no-reference metrics are computed.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, used for the output path.")
    parser.add_argument("--name", type=str, required=True, help="Run / checkpoint name, used for the output file name.")
    parser.add_argument("--sample", type=str, default="sample00", help="Sample subfolder of `--sr_dir` to evaluate.")
    parser.add_argument("--output_dir", type=str, default="metrics", help="Results are written to <output_dir>/<dataset>/results_<name>.json")
    parser.add_argument("--tb_dir", type=str, default=None, help="Also log the metrics (and a few images) to TensorBoard in this folder.")
    parser.add_argument("--tb_step", type=int, default=0, help="TensorBoard step of the results, e.g. the training step of the checkpoint.")
    parser.add_argument("--tb_images", type=int, default=4, help="Number of SR (| GT) images logged to TensorBoard.")
    return parser.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_preproc = transforms.ToTensor()

    sr_dir = os.path.join(args.sr_dir, args.sample)
    image_files = sorted(os.listdir(sr_dir))
    num_images = len(image_files)

    print("Initializing metric models")
    metrics = {"niqe": [], "maniqa": [], "musiq": [], "clipiqa": []}

    if args.gt_dir is not None:
        metrics.update({"psnr": [], "ssim": [], "lpips": [], "dists": []})
        lpips_iqa_metric = pyiqa.create_metric("lpips", device=device)
        dists_iqa_metric = pyiqa.create_metric("dists", device=device, crop_border=4)
        fid_iqa_metric = pyiqa.create_metric("fid", device=device)

    niqe_iqa_metric = pyiqa.create_metric("niqe", device=device, crop_border=4)
    maniqa_iqa_metric = pyiqa.create_metric("maniqa", device=device)
    musiq_iqa_metric = pyiqa.create_metric("musiq", device=device)
    clip_iqa_metric = pyiqa.create_metric("clipiqa", device=device)

    print(f"Calculating metrics on {num_images} images")
    for i, image in enumerate(image_files):
        sr_image = img_preproc(Image.open(os.path.join(sr_dir, image)).convert("RGB")).unsqueeze(0).to(device)

        if args.gt_dir is not None:
            gt_image = img_preproc(Image.open(os.path.join(args.gt_dir, image)).convert("RGB")).unsqueeze(0).to(device)

            metrics["psnr"].append(calculate_psnr_pt(gt_image, sr_image, crop_border=4, test_y_channel=True).item())
            metrics["ssim"].append(calculate_ssim_pt(gt_image, sr_image, crop_border=4, test_y_channel=True).item())
            metrics["lpips"].append(lpips_iqa_metric(sr_image, gt_image).item())
            metrics["dists"].append(dists_iqa_metric(sr_image, gt_image).item())

        metrics["niqe"].append(niqe_iqa_metric(sr_image).item())
        metrics["maniqa"].append(maniqa_iqa_metric(sr_image).item())
        metrics["musiq"].append(musiq_iqa_metric(sr_image).item())
        metrics["clipiqa"].append(clip_iqa_metric(sr_image).item())

        print(f"Image {i+1}/{num_images}: {image}")

    results = [(metric, float(np.mean(values))) for metric, values in metrics.items()]

    if args.gt_dir is not None:
        results.append(("fid", float(fid_iqa_metric(args.gt_dir, sr_dir))))

    save_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"results_{args.name}.json")
    with open(save_path, "w") as f:
        json.dump({args.dataset: results}, f, indent=4)

    print(f"Saved metrics to {save_path}")
    for metric, value in results:
        print(f"  {metric}: {value:.4f}")

    if args.tb_dir is not None:
        log_to_tensorboard(args, results, sr_dir, image_files)

def log_to_tensorboard(args, results, sr_dir, image_files):
    """Metrics as scalars 'test/<dataset>/<metric>' and the first images as 'test/<dataset>/<image>' (SR | GT)."""

    from torch.utils.tensorboard import SummaryWriter
    from segesr.utils.tensorboard_utils import side_by_side, to_tensorboard_image

    writer = SummaryWriter(args.tb_dir)
    for metric, value in results:
        writer.add_scalar(f"test/{args.dataset}/{metric}", value, args.tb_step)

    for image in image_files[:args.tb_images]:
        panels = [Image.open(os.path.join(sr_dir, image))]
        if args.gt_dir is not None:
            panels.append(Image.open(os.path.join(args.gt_dir, image)))
        writer.add_image(f"test/{args.dataset}/{Path(image).stem}", to_tensorboard_image(side_by_side(panels), max_size=1024),
                         args.tb_step, dataformats="HWC")

    writer.close()
    print(f"Logged to TensorBoard: {args.tb_dir} (step {args.tb_step})")

if __name__ == "__main__":
    main()
