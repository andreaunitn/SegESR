"""
Step R.0(b) of Track R: how well the U-Net features of FiDeSR on an LR match the DINOv2 features of its HR, block by
block, at three degradation levels, and whether that match goes with the quality of the output. Inference only.

For each HR crop (e.g. the 512 x 512 crops of DIV2K) and each level, the LR is made here, reproducibly: Gaussian blur,
bicubic 4x downscaling, Gaussian noise, JPEG (LEVELS). FiDeSR (official settings, empty prompt) runs on the LR; forward
hooks keep the output of every down block, of the middle block and of every up block of its U-Net. DINOv2 ViT-B/14
runs on the HR at 448 x 448 (32 x 32 patches). Both are average-pooled to the same GRID x GRID tokens.

Measures, per block and level:
- cknna: CKNNA (Huh et al. 2024, the measure of REPA) between the U-Net and DINOv2 features of the images, each image
  one mean-pooled vector, with K_DATASET neighbours;
- cka_image / cknna_image: the mean over images of the linear CKA / CKNNA between the U-Net and DINOv2 tokens of one
  image (GRID x GRID tokens, K_IMAGE neighbours);
- spearman_lpips: within each level, the Spearman correlation between cka_image and the LPIPS (AlexNet) of the output
  against the HR. A negative value: images whose features match DINOv2 of the HR better come out closer to the HR.
It writes <output_dir>/alignment.json (the measures and the check of gate R.0) and per_image.csv.

It must run from the FiDeSR code folder, as data_tools/fidesr_test.py:
    cd preset/external/FiDeSR && python <repo>/data_tools/feature_alignment.py --hr_dir <HR crops> \
        --pretrained_model_path <SD 2.1-base> --pretrained_path <fidesr.pkl> --dino_path <dinov2-base> \
        --num_images 500 --output_dir <dir>
"""

import argparse
import csv
import io
import json
import os
import sys
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Blur sigma (HR pixels), noise sigma (out of 255, on the LR), JPEG quality; the scale is 4 at every level
LEVELS = {
    "mild": (0.6, 2.0, 95),
    "medium": (1.2, 8.0, 75),
    "strong": (2.0, 20.0, 40),
}
GRID = 8            # tokens per side: the size of the middle block for a 512 x 512 input
K_DATASET = 10      # neighbours of CKNNA over images, as in REPA
K_IMAGE = 8         # neighbours of CKNNA over the GRID x GRID tokens of one image
DINO_SIZE = 448     # 32 x 32 patches of 14 pixels

def degrade(hr, level, seed, scale=4):
    """The LR of an HR PIL image at one of LEVELS, reproducible for a given seed."""

    blur, noise, quality = LEVELS[level]
    image = hr.filter(ImageFilter.GaussianBlur(radius=blur)) if blur > 0 else hr
    image = image.resize((hr.width // scale, hr.height // scale), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32)
    if noise > 0:
        array = array + np.random.default_rng(seed).normal(0.0, noise, array.shape)
    image = Image.fromarray(np.clip(np.round(array), 0, 255).astype(np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")

def seed_of(name, level):
    return zlib.crc32(f"{name}/{level}".encode())

def hsic_unbiased(K, L):
    """Unbiased HSIC of two n x n kernel matrices (Song et al. 2012), as in the CKNNA code of Huh et al."""

    n = K.shape[0]
    K = K.clone().fill_diagonal_(0)
    L = L.clone().fill_diagonal_(0)
    value = (torch.sum(K * L.T) + torch.sum(K) * torch.sum(L) / ((n - 1) * (n - 2))
             - 2 * torch.sum(K @ L) / (n - 2))
    return value / (n * (n - 3))

def cknna(a, b, k):
    """CKNNA between two sets of n features (n x d_a, n x d_b): CKA restricted to the pairs that are among the k
    nearest neighbours in both spaces (Huh et al., The Platonic Representation Hypothesis, 2024). Features are
    L2-normalised; the kernels are inner products."""

    a = F.normalize(a.double(), dim=1)
    b = F.normalize(b.double(), dim=1)
    K, L = a @ a.T, b @ b.T

    def similarity(K, L):
        n = K.shape[0]
        mask_k = torch.zeros_like(K).scatter_(1, K.clone().fill_diagonal_(-float("inf")).topk(k, dim=1).indices, 1)
        mask_l = torch.zeros_like(L).scatter_(1, L.clone().fill_diagonal_(-float("inf")).topk(k, dim=1).indices, 1)
        mask = mask_k * mask_l
        return hsic_unbiased(mask * K, mask * L)

    kl, kk, ll = similarity(K, L), similarity(K, K), similarity(L, L)
    return float(kl / (torch.sqrt(kk * ll) + 1e-12))

def linear_cka(a, b):
    """Linear CKA between two sets of n features (n x d_a, n x d_b)."""

    a = a.double() - a.double().mean(0, keepdim=True)
    b = b.double() - b.double().mean(0, keepdim=True)
    cross = torch.linalg.norm(b.T @ a) ** 2
    return float(cross / (torch.linalg.norm(a.T @ a) * torch.linalg.norm(b.T @ b) + 1e-12))

def tokens(feature, grid=GRID):
    """A 1 x C x H x W feature map average-pooled to grid x grid tokens: (grid*grid) x C."""

    pooled = F.adaptive_avg_pool2d(feature.float(), grid)
    return pooled[0].flatten(1).T

def spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).correlation)

def gate(measures, blocks):
    """Gate R.0: the dataset CKNNA drops by at least 20% (relative) from mild to strong, and on at least one block the
    mean within-level Spearman between cka_image and LPIPS is at most -0.3."""

    drops = {b: 1 - measures[b]["strong"]["cknna"] / measures[b]["mild"]["cknna"]
             for b in blocks if measures[b]["mild"]["cknna"] > 0}
    rhos = {b: float(np.mean([measures[b][lv]["spearman_lpips"] for lv in LEVELS])) for b in blocks}
    return {
        "cknna_drop_mild_to_strong": drops,
        "mean_spearman_lpips": rhos,
        "drop_ok": any(d >= 0.2 for d in drops.values()),
        "spearman_ok": any(r <= -0.3 for r in rhos.values()),
    }

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--hr_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dino_path", required=True, help="a local copy of facebook/dinov2-base")
    parser.add_argument("--num_images", type=int, default=500, help="N HR crops, evenly spaced in name order")
    parser.add_argument("--pretrained_model_path", required=True)
    parser.add_argument("--pretrained_path", required=True, help="fidesr.pkl")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_known_args(argv)

def main(argv=None):
    args, rest = parse_args(argv)
    # The FiDeSR options (LF/HF strengths, tiling, precision) with the defaults of data_tools/fidesr_test.py
    import fidesr_test
    model_args = fidesr_test.parse_args(["-i", args.hr_dir, "-o", args.output_dir,
                                         "--pretrained_model_path", args.pretrained_model_path,
                                         "--pretrained_path", args.pretrained_path] + rest)
    import lpips
    import torchvision.transforms.functional as TF
    from transformers import Dinov2Model
    from fidesr import FiDeSR_eval
    from src.my_utils.wavelet_color_fix import adain_color_fix

    torch.manual_seed(args.seed)
    model = FiDeSR_eval(model_args)
    model.set_eval()
    unet = model.unet
    blocks = ([f"down{i}" for i in range(len(unet.down_blocks))] + ["mid"]
              + [f"up{i}" for i in range(len(unet.up_blocks))])
    modules = list(unet.down_blocks) + [unet.mid_block] + list(unet.up_blocks)
    captured = {}
    for name, module in zip(blocks, modules):
        module.register_forward_hook(
            lambda _m, _i, out, name=name: captured.__setitem__(name, (out[0] if isinstance(out, tuple) else out)))
    dino = Dinov2Model.from_pretrained(args.dino_path).cuda().eval()
    lpips_net = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    names = sorted(n for n in os.listdir(args.hr_dir) if n.lower().endswith((".png", ".jpg", ".jpeg")))
    names = names[::max(1, len(names) // args.num_images)][:args.num_images]   # spread over the source images
    os.makedirs(args.output_dir, exist_ok=True)
    pooled = {(b, lv): [] for b in blocks for lv in LEVELS}      # one mean-pooled vector per image
    dino_pooled, rows = [], []
    for index, name in enumerate(names):
        hr = Image.open(os.path.join(args.hr_dir, name)).convert("RGB")
        hr_t = TF.to_tensor(hr).unsqueeze(0).cuda()
        with torch.no_grad():
            dino_in = (F.interpolate(hr_t, size=(DINO_SIZE, DINO_SIZE), mode="bicubic", align_corners=False)
                       .clamp(0, 1) - mean) / std
            patches = dino(pixel_values=dino_in).last_hidden_state[:, 1:]
            side = DINO_SIZE // 14
            dino_map = patches.transpose(1, 2).reshape(1, -1, side, side)
        dino_tok = tokens(dino_map)
        dino_pooled.append(dino_tok.mean(0).cpu())
        for level in LEVELS:
            lr = degrade(hr, level, seed_of(name, level))
            image, _, _ = fidesr_test.prepare_input(lr, model_args.process_size, model_args.upscale)
            captured.clear()
            with torch.no_grad():
                c_t = TF.to_tensor(image).unsqueeze(0).cuda() * 2 - 1
                _, output = model(c_t, prompt="")
                sr = adain_color_fix(target=TF.to_pil_image(torch.clip(output * 0.5 + 0.5, 0, 1)[0].cpu()),
                                     source=image)
                sr_t = TF.to_tensor(sr).unsqueeze(0).cuda()
                if sr_t.shape[-2:] != hr_t.shape[-2:]:
                    sr_t = F.interpolate(sr_t, size=hr_t.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
                lpips_value = float(lpips_net(sr_t * 2 - 1, hr_t * 2 - 1))
                psnr = float(10 * torch.log10(1 / torch.mean((sr_t - hr_t) ** 2).clamp_min(1e-10)))
            row = {"image": name, "level": level, "lpips": lpips_value, "psnr": psnr}
            for block in blocks:
                unet_tok = tokens(captured[block])
                pooled[(block, level)].append(unet_tok.mean(0).cpu())
                row[f"cka_{block}"] = linear_cka(unet_tok, dino_tok)
                row[f"cknna_{block}"] = cknna(unet_tok, dino_tok, K_IMAGE)
            rows.append(row)
        if index % 50 == 0:
            print(f"{index + 1}/{len(names)} images", flush=True)

    dino_all = torch.stack(dino_pooled)
    measures = {}
    for block in blocks:
        measures[block] = {}
        for level in LEVELS:
            level_rows = [r for r in rows if r["level"] == level]
            measures[block][level] = {
                "cknna": cknna(torch.stack(pooled[(block, level)]), dino_all, K_DATASET),
                "cka_image": float(np.mean([r[f"cka_{block}"] for r in level_rows])),
                "cknna_image": float(np.mean([r[f"cknna_{block}"] for r in level_rows])),
                "spearman_lpips": spearman([r[f"cka_{block}"] for r in level_rows],
                                           [r["lpips"] for r in level_rows]),
            }
    summary = {
        "images": len(names), "levels": LEVELS, "grid": GRID,
        "output_quality": {lv: {"lpips": float(np.mean([r["lpips"] for r in rows if r["level"] == lv])),
                                "psnr": float(np.mean([r["psnr"] for r in rows if r["level"] == lv]))}
                           for lv in LEVELS},
        "measures": measures,
        "gate": gate(measures, blocks),
    }
    with open(os.path.join(args.output_dir, "alignment.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, "per_image.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'block':<8}" + "".join(f"{'cknna ' + lv:>15}" for lv in LEVELS) + f"{'drop':>8}{'rho':>8}")
    for block in blocks:
        g = summary["gate"]
        print(f"{block:<8}" + "".join(f"{measures[block][lv]['cknna']:15.4f}" for lv in LEVELS)
              + f"{g['cknna_drop_mild_to_strong'].get(block, float('nan')):8.3f}{g['mean_spearman_lpips'][block]:8.3f}")
    print("output quality:", summary["output_quality"])
    print(f"gate R.0(b): drop {'ok' if summary['gate']['drop_ok'] else 'fails'}, "
          f"correlation {'ok' if summary['gate']['spearman_ok'] else 'fails'}")

if __name__ == "__main__":
    main()
