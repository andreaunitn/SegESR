"""
BiBandSR step O.1: a short fine-tuning of PiSA-SR from its official weights, with the original losses (the control)
or with the losses split at the LR Nyquist limit.

As in the second stage of the official training (train_pisasr.py of PiSA-SR), only the semantic LoRA is trained, the
pixel LoRA stays frozen, and both are active in the forward pass (the default inference). The losses:
  --loss pisasr   L2 + 2 LPIPS + CSD, all on every frequency (the official ones);
  --loss split    L2 on the band below the cutoff only; LPIPS and CSD with their gradient restricted to the band
                  above the cutoff (stop-gradient on the low band).
CSD acts on the SD latent, which cannot represent the cutoff (the latent only reaches 1/16 cycle per HR pixel), so in
the split mode its step is moved to pixel space through the decoder by a finite difference,
    delta = (decode(z) - decode(z - eps * g)) / eps,   g = the latent step of CSD,
and the loss is mean((x - stopgrad(x - H delta))^2), H = the high-pass filter. Its weight is calibrated at the start
so that, without H, its gradient on the latent has the norm of the latent CSD gradient.

Before training, both modes run the same calibration on the first batches (so that both see the same data and random
numbers); it also measures the share of the CSD and LPIPS gradients below the cutoff (calibration.json).

Outputs in --output_dir: checkpoints/model_<step>.pkl (the PiSA-SR format, read by test_pisasr.py), state.pt (for
resuming), log.jsonl (losses), val.jsonl (fixed validation crops, held out of training), config.json, FINISHED.
With --stop_after_minutes the run saves its state and stops in time for the job limit; running the same command
again resumes it.

Usage (from the repository root, BiBandSR environment, one GPU):
  python -m bibandsr.train --loss split --output_dir preset/train_output/bibandsr/o1_split
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from bibandsr.bands import low_band, low_mse, low_share, project_high

ROOT = Path(__file__).resolve().parents[1]
PISASR_DIR = ROOT / "preset" / "external" / "PiSA-SR"

# Negative prompt of the CSD loss: the default of PiSA-SR (src/my_utils/training_utils.py)
NEG_PROMPT = ("painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, "
              "unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, "
              "jpeg artifacts, deformed, lowres, over-smooth")

# Arguments that must not change when a run is resumed
FIXED_ARGS = ["loss", "gt_dir", "init_weights", "sd_path", "batch_size", "grad_accum", "lr", "lambda_l2",
              "lambda_lpips", "lambda_csd", "cfg_csd", "min_dm_step_ratio", "max_dm_step_ratio", "null_text_ratio",
              "cutoff", "width", "fd_eps", "calib_batches", "val_count", "seed", "resolution"]

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--loss", choices=["pisasr", "split"], required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gt_dir", default="preset/datasets/train_datasets/LSDIR/gt")
    p.add_argument("--init_weights", default="preset/models/pisasr/pisa_sr.pkl")
    p.add_argument("--sd_path", default="preset/models/stable-diffusion-2-1-base")
    p.add_argument("--ram_path", default="preset/models/ram_swin_large_14m.pth")
    p.add_argument("--deg_file", default=str(PISASR_DIR / "src" / "datasets" / "params.yml"),
                   help="Degradation settings (Real-ESRGAN pipeline of PiSA-SR).")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--max_steps", type=int, default=2000, help="Optimizer steps.")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lambda_l2", type=float, default=1.0)
    p.add_argument("--lambda_lpips", type=float, default=2.0)
    p.add_argument("--lambda_csd", type=float, default=1.0)
    p.add_argument("--cfg_csd", type=float, default=7.5)
    p.add_argument("--min_dm_step_ratio", type=float, default=0.02)
    p.add_argument("--max_dm_step_ratio", type=float, default=0.5)
    p.add_argument("--null_text_ratio", type=float, default=0.5)
    p.add_argument("--cutoff", type=float, default=0.125, help="Cycles per HR pixel (1/8: LR Nyquist limit of x4 SR).")
    p.add_argument("--width", type=float, default=0.1, help="Relative width of the smooth transition at the cutoff.")
    p.add_argument("--fd_eps", type=float, default=0.1, help="Finite-difference step of the CSD direction (split).")
    p.add_argument("--calib_batches", type=int, default=16)
    p.add_argument("--val_count", type=int, default=32, help="Last images of gt_dir, held out for validation.")
    p.add_argument("--val_every", type=int, default=250)
    p.add_argument("--save_every", type=int, default=250)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--stop_after_minutes", type=float, default=None,
                   help="Save the state and stop after this time (resume by running the same command again).")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args(argv)

# ---------------------------------------------------------------- data

class TrainSet(Dataset):
    """GT crops degraded online with the Real-ESRGAN pipeline of PiSA-SR; images in [-1, 1]."""

    def __init__(self, files, deg_file, resolution):
        self.files, self.deg_file = files, deg_file
        self.deg = None   # built in each worker
        self.crop = transforms.Compose([transforms.RandomCrop(resolution, pad_if_needed=True, padding_mode="reflect"), transforms.RandomHorizontalFlip()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        if self.deg is None:
            from src.datasets.realesrgan import RealESRGAN_degradation
            self.deg = RealESRGAN_degradation(self.deg_file, device="cpu")
        img = self.crop(Image.open(self.files[i]).convert("RGB"))
        gt, lq = self.deg.degrade_process(np.asarray(img) / 255.0, resize_bak=True)
        return {"gt": gt.squeeze(0).float() * 2 - 1, "lq": lq.squeeze(0).float() * 2 - 1}

def seed_worker(_):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)

def batches(dataset, args, start_step):
    """Endless batches; the order depends on the seed and on the step the run (re)starts from."""
    epoch = 0
    while True:
        gen = torch.Generator().manual_seed(args.seed + 1_000_003 * start_step + epoch)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, generator=gen,
                            num_workers=args.num_workers, worker_init_fn=seed_worker)
        yield from loader
        epoch += 1

def fixed_validation(files, args):
    """The validation crops, degraded once with fixed random numbers (centre crop, no flip)."""
    from src.datasets.realesrgan import RealESRGAN_degradation
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    random.seed(0), np.random.seed(0), torch.manual_seed(0)
    deg = RealESRGAN_degradation(args.deg_file, device="cpu")
    items = []
    for path in files:
        img = transforms.CenterCrop(args.resolution)(Image.open(path).convert("RGB"))
        gt, lq = deg.degrade_process(np.asarray(img) / 255.0, resize_bak=True)
        items.append((gt.float() * 2 - 1, lq.float() * 2 - 1))
    random.setstate(states[0]), np.random.set_state(states[1]), torch.set_rng_state(states[2])
    return items

# ---------------------------------------------------------------- models

def build_models(args, device):
    # The code of PiSA-SR (pisasr, src, its RAM with LoRA) first; basicsr (degradations) from third_party, as SegESR.
    # The ram package of third_party is shadowed by PiSA-SR's.
    sys.path.insert(0, str(PISASR_DIR))
    sys.path.append(str(ROOT / "third_party"))
    from pisasr import CSDLoss, PiSASR
    import lpips
    from ram import inference_ram
    from ram.models.ram_lora import ram

    official = torch.load(args.init_weights, map_location="cpu", weights_only=True)
    ns = SimpleNamespace(pretrained_model_path=args.sd_path, pretrained_model_path_csd=args.sd_path, resume_ckpt=None,
                         lora_rank_unet_pix=official["lora_rank_unet_pix"],
                         lora_rank_unet_sem=official["lora_rank_unet_sem"], timesteps1=1,
                         enable_xformers_memory_efficient_attention=False, null_text_ratio=args.null_text_ratio)
    model = PiSASR(ns)

    # Official LoRA weights (pixel and semantic); the module lists must be those of the official model
    lora = {n: p for n, p in model.unet.named_parameters() if "lora" in n}
    if set(lora) != set(official["state_dict_unet"]):
        raise SystemExit(f"LoRA parameters differ from {args.init_weights}")
    with torch.no_grad():
        for n, p in lora.items():
            p.copy_(official["state_dict_unet"][n])

    model.unet.set_adapter(["default_encoder_pix", "default_decoder_pix", "default_others_pix",
                            "default_encoder_sem", "default_decoder_sem", "default_others_sem"])
    model.set_train_sem()
    if args.gradient_checkpointing:
        model.unet.enable_gradient_checkpointing()
    trainable = [p for n, p in model.unet.named_parameters() if "lora" in n and "sem" in n]
    assert all(p.requires_grad for p in trainable) and trainable

    csd = CSDLoss(args=ns, accelerator=SimpleNamespace(mixed_precision="fp16", device=device))
    csd.requires_grad_(False)
    net_lpips = lpips.LPIPS(net="vgg").to(device)
    net_lpips.requires_grad_(False)

    ram_model = ram(pretrained=args.ram_path, pretrained_condition=None, image_size=384, vit="swin_l")
    ram_model.eval().to(device, dtype=torch.float16)
    ram_tf = transforms.Compose([transforms.Resize((384, 384)),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def captions(x):
        with torch.no_grad():
            return inference_ram(ram_tf(x * 0.5 + 0.5).to(torch.float16), ram_model)

    return model, ns, trainable, csd, net_lpips, captions

def decode(model, z):
    """As in PiSASR.forward: latent (scaled) -> image in [-1, 1]."""
    return model.vae_fix.decode(z / model.vae_fix.config.scaling_factor).sample.clamp(-1, 1)

def csd_step(csd, latents, prompt_embeds, neg_embeds, args):
    """The latent step g of CSD (CSDLoss.cal_csd of PiSA-SR): its loss is mean((z - stopgrad(z - g))^2)."""
    z = latents.detach().float()
    n = csd.sched.config.num_train_timesteps
    t = torch.randint(int(n * args.min_dm_step_ratio), int(n * args.max_dm_step_ratio), (z.shape[0],), device=z.device)
    noisy = csd.sched.add_noise(z, torch.randn_like(z), t)
    with torch.no_grad():
        eps = csd.forward_latent(csd.unet_fix, latents=torch.cat([noisy] * 2).half(), timestep=torch.cat([t] * 2),
                                 prompt_embeds=torch.cat([neg_embeds, prompt_embeds]).half()).float()
        eps_uncond, eps_text = eps.chunk(2)
        real = csd.eps_to_mu(csd.sched, eps_uncond + args.cfg_csd * (eps_text - eps_uncond), noisy, t)
        fake = csd.eps_to_mu(csd.sched, eps_uncond, noisy, t)
    weight = (z - real).abs().mean(dim=[1, 2, 3], keepdim=True)
    return (fake - real) / weight

def csd_pixel_direction(model, z, x, g, args):
    """The CSD step in pixel space, (decode(z) - decode(z - eps g)) / eps ~ J g."""
    with torch.no_grad():
        x_fd = decode(model, z.detach() - args.fd_eps * g.to(z.dtype))
    return (x.detach().float() - x_fd.float()) / args.fd_eps

def prompt_batch(captions, bs):
    # As train_pisasr.py: the RAM tags of the GT, then ", " and the (empty) positive prompt
    return {"prompt": [f"{c}, " for c in captions], "neg_prompt": [NEG_PROMPT] * bs, "null_prompt": [""] * bs}

# ---------------------------------------------------------------- losses

def compute_losses(model, ns, csd, net_lpips, captions, gt, lq, args, csd_scale):
    batch = prompt_batch(captions(gt), gt.shape[0])
    with torch.autocast("cuda", dtype=torch.float16):
        x, z, prompt_embeds, neg_embeds = model(lq, gt, batch=batch, args=ns)
        g = csd_step(csd, z, prompt_embeds, neg_embeds, args)
        xf, gtf = x.float(), gt.float()
        if args.loss == "pisasr":
            l2 = F.mse_loss(xf, gtf)
            lp = net_lpips(xf, gtf).mean()
            lcsd = F.mse_loss(z.float(), (z.float() - g).detach())
        else:
            l2 = low_mse(xf, gtf, args.cutoff, args.width)
            lp = net_lpips(project_high(xf, args.cutoff, args.width), gtf).mean()
            delta = csd_pixel_direction(model, z, x, g, args)
            delta = delta - low_band(delta, args.cutoff, args.width)
            lcsd = csd_scale * F.mse_loss(xf, (xf - delta).detach())
    total = args.lambda_l2 * l2 + args.lambda_lpips * lp + args.lambda_csd * lcsd
    with torch.no_grad():
        diag = {"mse": F.mse_loss(xf, gtf).item(), "mse_low": low_mse(xf, gtf, args.cutoff, args.width).item()}
    return total, {"l2": l2.item(), "lpips": lp.item(), "csd": lcsd.item(), **diag}

def calibrate(model, ns, csd, net_lpips, captions, data, args, device):
    """Scale of the pixel-space CSD (same gradient norm on the latent as the latent CSD, without the high-pass),
    and the share of the CSD and LPIPS gradients below the cutoff."""
    s = 2.0 ** 16   # the gradients pass through the fp16 decoder: scale them up, ratios are unchanged
    ratios, csd_low, lpips_low = [], [], []
    for _ in range(args.calib_batches):
        b = next(data)
        gt, lq = b["gt"].to(device), b["lq"].to(device)
        batch = prompt_batch(captions(gt), gt.shape[0])
        with torch.autocast("cuda", dtype=torch.float16):
            with torch.no_grad():
                _, z, prompt_embeds, neg_embeds = model(lq, gt, batch=batch, args=ns)
            z = z.detach().float().requires_grad_()
            x = decode(model, z)
            g = csd_step(csd, z, prompt_embeds, neg_embeds, args)
            delta = csd_pixel_direction(model, z, x, g, args)
            xf = x.float()
            l_lat = s * F.mse_loss(z, (z - g).detach())
            l_pix = s * F.mse_loss(xf, (xf - delta).detach())
            l_lp = s * net_lpips(xf, gt.float()).mean()
        g_lat = torch.autograd.grad(l_lat, z, retain_graph=True)[0]
        g_pix = torch.autograd.grad(l_pix, z, retain_graph=True)[0]
        g_x = torch.autograd.grad(l_lp, x)[0]
        ratios.append((g_lat.float().norm() / g_pix.float().norm().clamp_min(1e-20)).item())
        csd_low.append(low_share(delta, args.cutoff, args.width))
        lpips_low.append(low_share(g_x, args.cutoff, args.width))
    return {"csd_pixel_scale": float(np.median(ratios)), "csd_pixel_scale_all": ratios,
            "csd_grad_low_share": float(np.mean(csd_low)), "lpips_grad_low_share": float(np.mean(lpips_low))}

def psnr(mse):
    return 10 * math.log10(4.0 / max(mse, 1e-12))   # images in [-1, 1]: peak-to-peak 2

@torch.no_grad()
def validate(model, ns, net_lpips, captions, val, args, device):
    # Same random numbers (VAE sampling) at every validation and in every run; the training ones are left untouched
    py_state = random.getstate()
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(0)
        result = _validate(model, ns, net_lpips, captions, val, args, device)
    random.setstate(py_state)
    return result

def _validate(model, ns, net_lpips, captions, val, args, device):
    ns_val = SimpleNamespace(**{**vars(ns), "null_text_ratio": 0.0})
    tot = {"mse": 0.0, "mse_low": 0.0, "mse_high": 0.0, "lpips": 0.0}
    for gt, lq in val:
        gt, lq = gt[None].to(device), lq[None].to(device)
        # As at test time: the RAM tags of the LR
        batch = {"prompt": captions(lq), "neg_prompt": [NEG_PROMPT], "null_prompt": [""]}
        with torch.autocast("cuda", dtype=torch.float16):
            x, _, _, _ = model(lq, gt, batch=batch, args=ns_val)
        err = x.float() - gt.float()
        err_low = low_band(err, args.cutoff, args.width)
        tot["mse"] += err.pow(2).mean().item()
        tot["mse_low"] += err_low.pow(2).mean().item()
        tot["mse_high"] += (err - err_low).pow(2).mean().item()
        tot["lpips"] += net_lpips(x.float(), gt.float()).mean().item()
    n = len(val)
    return {"psnr": psnr(tot["mse"] / n), "psnr_low": psnr(tot["mse_low"] / n), "psnr_high": psnr(tot["mse_high"] / n),
            "lpips": tot["lpips"] / n}

# ---------------------------------------------------------------- state

def save_state(path, model, optimizer, scaler, step, calibration):
    sem = {n: p.detach().cpu() for n, p in model.unet.named_parameters() if "lora" in n and "sem" in n}
    state = {"step": step, "sem_lora": sem, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
             "calibration": calibration,
             "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                     "cuda": torch.cuda.get_rng_state()}}
    tmp = f"{path}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)

def load_state(path, model, optimizer, scaler):
    state = torch.load(path, map_location="cpu", weights_only=False)   # our own file (it holds the RNG states)
    params = dict(model.unet.named_parameters())
    with torch.no_grad():
        for n, v in state["sem_lora"].items():
            params[n].copy_(v)
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"])
    torch.cuda.set_rng_state(state["rng"]["cuda"])
    return state["step"], state["calibration"]

def append_json(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")

def check_config(out, args):
    """Write config.json on the first start; on a resume, refuse changed settings."""
    path = out / "config.json"
    cfg = {k: getattr(args, k) for k in FIXED_ARGS}
    if path.exists():
        old = json.loads(path.read_text())
        changed = {k: (old.get(k), v) for k, v in cfg.items() if old.get(k) != v}
        if changed:
            raise SystemExit(f"Settings differ from the run in {out} (old, new): {changed}")
    else:
        path.write_text(json.dumps(cfg, indent=1) + "\n")

# ---------------------------------------------------------------- main

def main(argv=None):
    args = parse_args(argv)
    start_time = time.time()
    out = Path(args.output_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    if (out / "FINISHED").exists():
        print(f"{out}: already finished")
        return
    check_config(out, args)
    device = torch.device("cuda")

    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    model, ns, trainable, csd, net_lpips, captions = build_models(args, device)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)
    scaler = torch.amp.GradScaler("cuda")

    files = sorted(str(p) for p in Path(args.gt_dir).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    train_files, val_files = files[:-args.val_count], files[-args.val_count:]
    val = fixed_validation(val_files, args)
    dataset = TrainSet(train_files, args.deg_file, args.resolution)
    print(f"{len(train_files)} training images, {len(val)} validation crops; loss: {args.loss}")

    state_path = out / "state.pt"
    if state_path.exists():
        step, calibration = load_state(state_path, model, optimizer, scaler)
        print(f"Resumed at step {step}")
        data = batches(dataset, args, step)
    else:
        step = 0
        data = batches(dataset, args, step)
        calibration = calibrate(model, ns, csd, net_lpips, captions, data, args, device)
        (out / "calibration.json").write_text(json.dumps(calibration, indent=1) + "\n")
        print(f"Calibration: {calibration}")
        append_json(out / "val.jsonl", {"step": 0, **validate(model, ns, net_lpips, captions, val, args, device)})
    csd_scale = calibration["csd_pixel_scale"]

    sums, n_sums, t_log = {}, 0, time.time()
    torch.cuda.reset_peak_memory_stats()
    while step < args.max_steps:
        for _ in range(args.grad_accum):
            b = next(data)
            loss, parts = compute_losses(model, ns, csd, net_lpips, captions, b["gt"].to(device), b["lq"].to(device),
                                         args, csd_scale)
            scaler.scale(loss / args.grad_accum).backward()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            n_sums += 1
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_every == 0:
            rec = {"step": step, **{k: v / n_sums for k, v in sums.items()},
                   "sec_per_step": (time.time() - t_log) / args.log_every,
                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
            append_json(out / "log.jsonl", rec)
            print(json.dumps(rec), flush=True)
            sums, n_sums, t_log = {}, 0, time.time()
        if step % args.val_every == 0 or step == args.max_steps:
            append_json(out / "val.jsonl", {"step": step, **validate(model, ns, net_lpips, captions, val, args, device)})
        if step % args.save_every == 0 or step == args.max_steps:
            model.save_model(str(out / "checkpoints" / f"model_{step}.pkl"))
            save_state(state_path, model, optimizer, scaler, step, calibration)
        if args.stop_after_minutes and time.time() - start_time > 60 * args.stop_after_minutes and step < args.max_steps:
            save_state(state_path, model, optimizer, scaler, step, calibration)
            print(f"Stopped for the time limit at step {step}; run the same command again to resume.")
            return

    (out / "FINISHED").write_text(f"{step}\n")
    print(f"Finished at step {step}")

if __name__ == "__main__":
    main()
