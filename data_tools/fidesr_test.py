"""
FiDeSR (CVPR'26) inference with a choice of prompt, for step R.0 of Track R.

The official test script (preset/external/FiDeSR/test_fidesr.py) always uses an empty prompt, while the official
training gives the U-Net the RAM tags of the GT in half of the steps (`--null_text_ratio 0.5`) and an empty prompt in
the other half. This script runs the same model, with the same resizing and color fix, and one of three prompts:
    --prompt empty    the official setting ('')
    --prompt ram_lr   RAM tags of the input (the LR resized to the output size), what a model can use at test
    --prompt ram_hr   RAM tags of the GT: an oracle, the tags FiDeSR sees in training
Tags are written as in the FiDeSR training ("<tags>, ", with its default empty --pos_prompt_csd).
It also prints the parameters of each part and the mean time per image of RAM and of the SR model.

It must run from the FiDeSR code folder, whose modules import from the working directory:
    cd preset/external/FiDeSR && python <repo>/data_tools/fidesr_test.py \
        --pretrained_model_path <SD 2.1-base> --pretrained_path <fidesr.pkl> --ram_path <ram_swin_large_14m.pth> \
        --prompt ram_lr --hf_scale 0.2 --lf_scale 0.2 -i <LR dir> -o <output dir>
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.getcwd())

RAM_TRANSFORM = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
EXTENSIONS = ("png", "jpg", "jpeg", "bmp", "tiff")

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input_image", "-i", required=True, help="an LR image or a folder of them")
    parser.add_argument("--output_dir", "-o", required=True)
    parser.add_argument("--prompt", choices=["empty", "ram_lr", "ram_hr"], default="empty")
    parser.add_argument("--hr_dir", default=None, help="folder of the GTs, same file stems as the LRs (ram_hr)")
    parser.add_argument("--ram_path", default=None, help="ram_swin_large_14m.pth (ram_lr, ram_hr)")
    parser.add_argument("--pretrained_model_path", required=True)
    parser.add_argument("--pretrained_path", required=True, help="fidesr.pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--align_method", choices=["wavelet", "adain", "nofix"], default="adain")
    # The remaining options and defaults are those of the official test script
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)
    parser.add_argument("--mixed_precision", default="fp16")
    parser.add_argument("--lf_scale", type=float, default=0.2)
    parser.add_argument("--lf_rc", type=float, default=0.10)
    parser.add_argument("--lf_order", type=int, default=2)
    parser.add_argument("--lf_tau", type=float, default=0.8)
    parser.add_argument("--lf_sharp", type=float, default=10.0)
    parser.add_argument("--lf_dmap_gamma", type=float, default=1.2)
    parser.add_argument("--hf_scale", type=float, default=0.3)
    parser.add_argument("--hf_rc", type=float, default=0.32)
    parser.add_argument("--hf_order", type=int, default=2)
    parser.add_argument("--hf_dmap_gamma", type=float, default=1.2)
    parser.add_argument("--lf_hf_merge_ratio", type=float, default=0.5)
    parser.add_argument("--lrrb_in_ch", type=int, default=8)
    parser.add_argument("--lrrb_mid_ch", type=int, default=64)
    parser.add_argument("--lrrb_growth", type=int, default=32)
    parser.add_argument("--lrrb_res_sc", type=float, default=0.2)
    args = parser.parse_args(argv)
    if args.prompt != "empty" and not args.ram_path:
        parser.error(f"--prompt {args.prompt} needs --ram_path")
    if args.prompt == "ram_hr" and not args.hr_dir:
        parser.error("--prompt ram_hr needs --hr_dir")
    return args

def list_images(path):
    if not os.path.isdir(path):
        return [path]
    names = set()
    for ext in EXTENSIONS:
        names.update(glob.glob(os.path.join(path, f"*.{ext}")) + glob.glob(os.path.join(path, f"*.{ext.upper()}")))
    return sorted(names)

def find_by_stem(folder, stem):
    for ext in EXTENSIONS + tuple(e.upper() for e in EXTENSIONS):
        path = os.path.join(folder, f"{stem}.{ext}")
        if os.path.exists(path):
            return path
    raise SystemExit(f"No GT for {stem} in {folder}")

def prepare_input(image, process_size, upscale):
    """The resizing of the official test script: (LR resized to the model input, resize_flag, original size)."""

    width, height = image.size
    resized = False
    if width < process_size // upscale or height < process_size // upscale:
        scale = (process_size // upscale) / min(width, height)
        image = image.resize((int(scale * width), int(scale * height)))
        resized = True
    image = image.resize((image.size[0] * upscale, image.size[1] * upscale))
    image = image.resize((image.width - image.width % 8, image.height - image.height % 8), Image.LANCZOS)
    return image, resized, (width, height)

def count(module):
    return sum(p.numel() for p in module.parameters())

def ram_tags(model, image01):
    """RAM tags of a 1x3xHxW image in [0, 1] on the GPU, as a FiDeSR prompt, and the time they took."""

    from ram import inference_ram
    torch.cuda.synchronize()
    start = time.time()
    caption = inference_ram(RAM_TRANSFORM(image01).to(torch.float16), model)[0]
    torch.cuda.synchronize()
    return f"{caption}, ", time.time() - start

def main(argv=None):
    args = parse_args(argv)
    from fidesr import FiDeSR_eval
    from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

    torch.manual_seed(args.seed)    # the VAE encoder samples its latent
    model = FiDeSR_eval(args)
    model.set_eval()
    parts = {"unet (LoRA merged)": model.unet, "vae": model.vae, "text_encoder": model.text_encoder,
             "lrrb": model.lrrb}
    ram = None
    if args.prompt != "empty":
        from ram.models.ram_lora import ram as build_ram
        ram = build_ram(pretrained=args.ram_path, pretrained_condition=None, image_size=384, vit="swin_l")
        ram = ram.eval().to("cuda", dtype=torch.float16)
        parts["ram"] = ram
    for name, module in parts.items():
        print(f"parameters, {name}: {count(module) / 1e6:.1f} M")

    names = list_images(args.input_image)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"There are {len(names)} images; prompt: {args.prompt}")
    sr_times, ram_times = [], []
    for path in names:
        image, resized, (width, height) = prepare_input(Image.open(path).convert("RGB"), args.process_size,
                                                        args.upscale)
        stem = os.path.splitext(os.path.basename(path))[0]
        with torch.no_grad():
            c_t = TF.to_tensor(image).unsqueeze(0).cuda() * 2 - 1
            prompt = ""
            if args.prompt == "ram_lr":
                prompt, seconds = ram_tags(ram, c_t * 0.5 + 0.5)
                ram_times.append(seconds)
            elif args.prompt == "ram_hr":
                gt = Image.open(find_by_stem(args.hr_dir, stem)).convert("RGB")
                prompt, seconds = ram_tags(ram, TF.to_tensor(gt).unsqueeze(0).cuda())
                ram_times.append(seconds)
            seconds, output = model(c_t, prompt=prompt)
        sr_times.append(seconds)

        output = transforms.ToPILImage()(torch.clip(output * 0.5 + 0.5, 0, 1)[0].cpu())
        if args.align_method == "adain":
            output = adain_color_fix(target=output, source=image)
        elif args.align_method == "wavelet":
            output = wavelet_color_fix(target=output, source=image)
        if resized:
            output = output.resize((int(args.upscale * width), int(args.upscale * height)))
        output.save(os.path.join(args.output_dir, os.path.basename(path)))
        if prompt:
            with open(os.path.join(args.output_dir, f"{stem}.txt"), "w") as f:
                f.write(prompt)

    skip = 3 if len(sr_times) > 3 else 0    # the first images warm the GPU up, as in the official script
    print(f"mean time per image, SR model: {np.mean(sr_times[skip:]):.4f} s")
    if ram_times:
        print(f"mean time per image, RAM: {np.mean(ram_times[skip:]):.4f} s")

if __name__ == "__main__":
    main()
