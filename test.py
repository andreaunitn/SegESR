import os
import sys
from pathlib import Path

# Make `segesr` and the vendored `ram` / `basicsr` packages importable without installation
PROJECT_ROOT = Path(__file__).resolve().parent
for _path in (PROJECT_ROOT, PROJECT_ROOT / "third_party"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import glob
import argparse
from PIL import Image

import torch
from torchvision import transforms

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from ram.models.ram_lora import ram
from ram import inference_ram as inference

from segesr.models.controlnet import ControlNetModel
from segesr.models.unet_2d_condition import UNet2DConditionModel
from segesr.pipelines.pipeline_segesr import StableDiffusionControlNetPipeline
from segesr.utils import compute_sam2_conditions, load_sam2, parse_args_with_config, seg_logits_to_hidden_states
from segesr.utils.color_fix import adain_color_fix, wavelet_color_fix

tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
])

ram_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_segesr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention):

    # Load scheduler, tokenizer and models.
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
    feature_extractor = CLIPImageProcessor.from_pretrained(os.path.join(args.pretrained_model_path, "feature_extractor"))
    unet = UNet2DConditionModel.from_pretrained(args.finetuned_model_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(args.finetuned_model_path, subfolder="controlnet")

    # Freeze everything
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    validation_pipeline = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, feature_extractor=feature_extractor,
        unet=unet, controlnet=controlnet, scheduler=scheduler, safety_checker=None, requires_safety_checker=False,
    )

    validation_pipeline._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    return validation_pipeline

def load_tag_model(args, device="cuda"):
    model = ram(pretrained=args.ram_path,
                pretrained_condition=args.ram_ft_path,
                image_size=384,
                vit="swin_l")
    model.eval()
    model.to(device)

    return model

@torch.no_grad()
def get_validation_prompt(args, image, model, device="cuda"):
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    lq = ram_transforms(lq)
    res = inference(lq, model)
    ram_encoder_hidden_states = model.generate_image_embeds(lq)

    validation_prompt = f"{res[0]}, {args.prompt},"

    return validation_prompt, ram_encoder_hidden_states

def main(args, enable_xformers_memory_efficient_attention=True):
    txt_path = os.path.join(args.output_dir, "txt")
    os.makedirs(txt_path, exist_ok=True)

    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    if args.seed is not None:
        set_seed(args.seed)

    pipeline = load_segesr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention)
    model = load_tag_model(args, accelerator.device)
    sam_generator = load_sam2(
        model_size=args.sam_model_size,
        device=accelerator.device,
        points_per_side=16,
        points_per_batch=128,
        stability_score_thresh=0.9,
    )

    if accelerator.is_main_process:
        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator.manual_seed(args.seed)

        if os.path.isdir(args.image_path):
            image_names = sorted(glob.glob(f"{args.image_path}/*.*"))
        else:
            image_names = [args.image_path]

        for image_idx, image_name in enumerate(image_names):
            print(f"================== process {image_idx} imgs... ===================")
            validation_image = Image.open(image_name).convert("RGB")

            validation_prompt, ram_encoder_hidden_states = get_validation_prompt(args, validation_image, model, accelerator.device)
            validation_prompt += args.added_prompt # clean, extremely detailed, best quality, sharp, clean
            negative_prompt = args.negative_prompt # dirty, messy, low quality, frames, deformed,

            # SAM 2 conditions are computed on the original LR image, as for the training data
            sam_img_embeds, sam_seg_logits = compute_sam2_conditions(validation_image, sam_generator)
            sam2_encoder_hidden_states = sam_img_embeds.to(accelerator.device)
            sam2_segmentation_encoder_hidden_states = seg_logits_to_hidden_states(sam_seg_logits).to(accelerator.device)

            if args.save_prompts:
                txt_save_path = os.path.join(txt_path, f"{Path(image_name).stem}.txt")
                with open(txt_save_path, "w") as file:
                    file.write(validation_prompt)

            print(f"{validation_prompt}")

            ori_width, ori_height = validation_image.size
            rscale = args.upscale
            if ori_width < args.process_size//rscale or ori_height < args.process_size//rscale:
                scale = (args.process_size//rscale)/min(ori_width, ori_height)
                validation_image = validation_image.resize((int(scale*ori_width), int(scale*ori_height)))

            validation_image = validation_image.resize((validation_image.size[0]*rscale, validation_image.size[1]*rscale))
            validation_image = validation_image.resize((validation_image.size[0]//8*8, validation_image.size[1]//8*8))
            width, height = validation_image.size

            print(f"input size: {height}x{width}")

            for sample_idx in range(args.sample_times):
                sample_dir = os.path.join(args.output_dir, f"sample{str(sample_idx).zfill(2)}")
                os.makedirs(sample_dir, exist_ok=True)

                with torch.autocast("cuda"):
                    image = pipeline(
                            validation_prompt, validation_image, num_inference_steps=args.num_inference_steps, generator=generator, height=height, width=width,
                            guidance_scale=args.guidance_scale, negative_prompt=negative_prompt, conditioning_scale=args.conditioning_scale,
                            start_steps=args.start_steps, start_point=args.start_point, ram_encoder_hidden_states=ram_encoder_hidden_states,
                            sam2_segmentation_encoder_hidden_states=sam2_segmentation_encoder_hidden_states,
                            sam2_encoder_hidden_states=sam2_encoder_hidden_states,
                            latent_tiled_size=args.latent_tiled_size, latent_tiled_overlap=args.latent_tiled_overlap,
                            args=args,
                        ).images[0]

                if args.align_method == "wavelet":
                    image = wavelet_color_fix(image, validation_image)
                elif args.align_method == "adain":
                    image = adain_color_fix(image, validation_image)

                # Bring the output back to exactly `upscale` x the original LR size
                image = image.resize((ori_width*rscale, ori_height*rscale))

                image.save(os.path.join(sample_dir, f"{Path(image_name).stem}.png"))

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="SegESR inference script.")
    parser.add_argument("--finetuned_model_path", type=str, default=None, help="Checkpoint folder containing 'unet/' and 'controlnet/'.")
    parser.add_argument("--pretrained_model_path", type=str, default="preset/models/stable-diffusion-2-base")
    parser.add_argument("--ram_path", type=str, default="preset/models/ram_swin_large_14m.pth")
    parser.add_argument("--ram_ft_path", type=str, default=None, help="Path to the DAPE weights.")
    parser.add_argument("--sam_model_size", type=str, default="large", choices=["tiny", "small", "base_plus", "large"])
    parser.add_argument("--prompt", type=str, default="") # user can add self-prompt to improve the results
    parser.add_argument("--added_prompt", type=str, default="clean, high-resolution, 8k")
    parser.add_argument("--negative_prompt", type=str, default="dotted, noise, blur, lowres, smooth")
    parser.add_argument("--image_path", type=str, default=None, help="LR image or folder of LR images.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--guidance_scale", type=float, default=5.5)
    parser.add_argument("--conditioning_scale", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224) # latent size, for 24G
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024) # image size, for 13G
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample_times", type=int, default=1)
    parser.add_argument("--align_method", type=str, choices=["wavelet", "adain", "nofix"], default="adain")
    parser.add_argument("--start_steps", type=int, default=999) # defaults set to 999.
    parser.add_argument("--start_point", type=str, choices=["lr", "noise"], default="lr") # LR Embedding Strategy, choose 'lr latent + 999 steps noise' as diffusion start point.
    parser.add_argument("--save_prompts", action="store_true")
    args = parse_args_with_config(parser, input_args)

    for required in ("finetuned_model_path", "image_path", "output_dir"):
        if getattr(args, required) is None:
            parser.error(f"--{required} is required (on the command line or in --config).")

    return args

if __name__ == "__main__":
    main(parse_args())
