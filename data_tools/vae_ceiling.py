"""
VAE ceiling: how well can any latent diffusion SR model do, at best, with this VAE?

Every GT (HR) image is passed through the VAE and the reconstruction is saved like an SR output
(<output_dir>/sample00/<same file name>), so that data_tools/metrics.py scores it against the GT:

  --mode encode    decoder(encoder(GT)): what the VAE keeps of the GT. A latent SR model whose latent were
                   exactly the encoding of the GT would produce this image.
  --mode optimize  decoder(z*), with z* = argmin_z |decoder(z) - GT|^2, found by gradient descent starting from
                   encoder(GT): the best image the DECODER can produce at all. The gap to `encode` is lost
                   by the encoder, what remains is the decoder's own limit.

Usage:
    python data_tools/vae_ceiling.py --gt_dir preset/datasets/test_datasets/RealSR/test_HR \
        --output_dir preset/datasets/output/vae_sd2_encode/RealSR [--mode optimize --steps 200]
On the cluster: sbatch cluster/vae_ceiling.sh (VAE reconstruction + metrics for several datasets).
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from PIL import Image

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def to_tensor(image):
    """PIL RGB image -> (1, 3, H, W) tensor in [-1, 1]."""

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)

def to_image(tensor):
    """(1, 3, H, W) tensor in [-1, 1] -> PIL RGB image (rounded to 8 bits, like a saved SR output)."""

    array = ((tensor[0].clamp(-1, 1) + 1.0) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)

def pad_to_multiple(x, multiple):
    """Reflect-pads (1, 3, H, W) so that H and W are multiples of `multiple`; returns the padded tensor and (H, W)."""

    h, w = x.shape[-2:]
    pad_h, pad_w = -h % multiple, -w % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (h, w)

@torch.no_grad()
def encode_decode(vae, x):
    """Deterministic VAE reconstruction: decoder(mean of the encoder posterior)."""

    latent = vae.encode(x).latent_dist.mode()
    return vae.decode(latent).sample, latent

def optimize_latent(vae, x, latent, steps, lr):
    """Gradient descent on the latent to minimize the MSE of the decoded image to x. Returns the best image and MSEs."""

    latent = latent.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([latent], lr=lr)
    best_mse, best_image = float("inf"), None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        decoded = vae.decode(latent).sample
        loss = F.mse_loss(decoded, x)
        if loss.item() < best_mse:
            best_mse, best_image = loss.item(), decoded.detach()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        decoded = vae.decode(latent).sample
        loss = F.mse_loss(decoded, x).item()
        if loss < best_mse:
            best_mse, best_image = loss, decoded
    return best_image, best_mse

def reconstruct(vae, image, mode="encode", steps=200, lr=0.05):
    """VAE reconstruction of one PIL image, same size as the input. Returns (PIL image, MSE after encode, final MSE)."""

    device = next(vae.parameters()).device
    x = to_tensor(image).to(device)
    multiple = 2 ** (len(vae.config.block_out_channels) - 1)
    x_padded, (h, w) = pad_to_multiple(x, multiple)

    decoded, latent = encode_decode(vae, x_padded)
    encode_mse = F.mse_loss(decoded, x_padded).item()
    final_mse = encode_mse
    if mode == "optimize":
        decoded, final_mse = optimize_latent(vae, x_padded, latent, steps, lr)

    return to_image(decoded[..., :h, :w]), encode_mse, final_mse

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Reconstruct GT images with a VAE (upper bound of latent SR models).")
    parser.add_argument("--gt_dir", required=True, help="Folder of GT (HR) images.")
    parser.add_argument("--output_dir", required=True, help="Reconstructions are saved to <output_dir>/sample00/.")
    parser.add_argument("--vae_path", default="preset/models/stable-diffusion-2-base",
                        help="Diffusers model folder with a 'vae' subfolder (or a VAE folder itself).")
    parser.add_argument("--mode", choices=["encode", "optimize"], default="encode")
    parser.add_argument("--steps", type=int, default=200, help="Gradient steps on the latent (--mode optimize).")
    parser.add_argument("--lr", type=float, default=0.05, help="Adam learning rate on the latent (--mode optimize).")
    parser.add_argument("--skip_existing", action="store_true", help="Skip images already reconstructed.")
    return parser.parse_args(argv)

def load_vae(path, device):
    subfolder = "vae" if os.path.isdir(os.path.join(path, "vae")) else None
    vae = AutoencoderKL.from_pretrained(path, subfolder=subfolder).to(device, dtype=torch.float32)
    vae.eval().requires_grad_(False)
    return vae

def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = load_vae(args.vae_path, device)

    sample_dir = os.path.join(args.output_dir, "sample00")
    os.makedirs(sample_dir, exist_ok=True)
    names = sorted(name for name in os.listdir(args.gt_dir) if name.lower().endswith(IMAGE_EXTENSIONS))
    if not names:
        raise SystemExit(f"No images found in {args.gt_dir}")

    print(f"VAE reconstruction ({args.mode}) of {len(names)} images from {args.gt_dir} with {args.vae_path} (fp32)")
    for i, name in enumerate(names):
        out_path = os.path.join(sample_dir, os.path.splitext(name)[0] + ".png")
        if args.skip_existing and os.path.exists(out_path):
            continue
        image, encode_mse, final_mse = reconstruct(vae, Image.open(os.path.join(args.gt_dir, name)), args.mode, args.steps, args.lr)
        image.save(out_path)
        detail = f"  optimized MSE {final_mse:.6f}" if args.mode == "optimize" else ""
        print(f"[{i + 1}/{len(names)}] {name}: encode MSE {encode_mse:.6f}{detail}")

if __name__ == "__main__":
    main()
