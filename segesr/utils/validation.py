import os
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
from accelerate.logging import get_logger

from segesr.pipelines.pipeline_segesr import StableDiffusionControlNetPipeline
from third_party.ram import inference_ram as inference

logger = get_logger(__name__)

def image_grid(imgs, rows, cols):
    """Combines a list of PIL images into a single grid image."""

    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid

def validation(
        unet,
        controlnet,
        vae,
        text_encoder,
        tokenizer,
        noise_scheduler,
        tiny_vae,
        ram_model,
        sam_loss_fn,
        lpips_loss_fn,
        validation_dataloader,
        global_step,
        args,
        accelerator,
        weight_dtype
):

    """
    Executes validation:
    1. Generates and saves visual super-resolution samples (if enabled).
    2. Computes quantitative validation losses (Diffusion MSE, SAM-2 feature MSE, and LPIPS).
    """

    logger.info("Running validation...")
    val_logs = {}

    # -------------------------------------------------------------------------
    # 1. Visual Image Generation (Qualitative Check)
    # -------------------------------------------------------------------------
    if args.generate_validation_image and accelerator.is_main_process:
        pipeline = StableDiffusionControlNetPipeline(
            vae=accelerator.unwrap_model(vae),
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            unet=accelerator.unwrap_model(unet),
            controlnet=accelerator.unwrap_model(controlnet),
            scheduler=noise_scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=None
        )
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        if args.validation_image and args.validation_prompt:
            val_image_path = args.validation_image[0]
            val_image = Image.open(val_image_path).convert("RGB")

            tensor_transforms = transforms.ToTensor()
            ram_transforms = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            lq_for_ram = tensor_transforms(val_image).unsqueeze(0).to(accelerator.device)
            lq_for_ram = ram_transforms(lq_for_ram)
            ram_tags = inference(lq_for_ram, ram_model)
            ram_embeds = ram_model.generate_image_embeds(lq_for_ram)

            final_prompt = f"{ram_tags[0]}, clean, high-resolution, 8k"
            negative_prompt = "dotted, noise, blur, lowres, smooth"

            width, height = val_image.size
            cond_image = val_image.resize((width * 4, height * 4))

            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed if args.seed else 42)
            logger.info(f"Generating validation sample at step {global_step} with prompt: '{final_prompt}'")

            with torch.autocast("cuda"):
                generated_image = pipeline(
                    prompt=final_prompt,
                    image=cond_image,
                    negative_prompt=negative_prompt,
                    num_inference_steps=50,
                    generator=generator,
                    height=height*4,
                    width=width*4,
                    guidance_scale=5.5,
                    ram_encoder_hidden_states=ram_embeds,
                ).images[0]

            val_dir = os.path.join(args.output_dir, "validation_samples")
            os.makedirs(val_dir, exist_ok=True)
            save_path = os.path.join(val_dir, f"step_{global_step}.png")
            generated_image.save(save_path)
            logger.info(f"Saved validation image to {save_path}")

        del pipeline
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 2. Validation Loss Calculation (Quantitative Check)
    # -------------------------------------------------------------------------

    unet.eval()
    controlnet.eval()

    try:
        if validation_dataloader is not None and len(validation_dataloader) > 0:
            total_val_loss = 0.0
            total_val_diffusion_loss = 0.0
            total_val_sam_loss = 0.0
            total_val_lpips_loss = 0.0

            for val_batch in tqdm(
                validation_dataloader,
                desc="Calculating validation loss",
                disable=not accelerator.is_local_main_process,
            ):

                with torch.no_grad():
                    pixel_values = val_batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                    latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    ).long()
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    encoder_hidden_states = text_encoder(val_batch["input_ids"].to(accelerator.device))[0]
                    ram_hidden = val_batch["ram_values"].to(accelerator.device, dtype=weight_dtype)
                    controlnet_cond = val_batch["conditioning_pixel_values"].to(accelerator.device, dtype=weight_dtype)

                    sam_kwargs = {
                        "sam_encoder_hidden_states": val_batch["sam_img_embeds"].to(accelerator.device, dtype=weight_dtype),
                        "sam_segmentation_hidden_states": val_batch["sam_seg_embeds"].to(accelerator.device, dtype=weight_dtype)
                    } if args.use_sam else {}

                    down_block_res_samples, mid_block_res_sample = controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=controlnet_cond,
                        return_dict=False,
                        image_encoder_hidden_states=ram_hidden,
                        **sam_kwargs,
                    )

                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=[sample.to(dtype=weight_dtype) for sample in down_block_res_samples],
                        mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                        image_encoder_hidden_states=ram_hidden,
                        **sam_kwargs,
                    ).sample

                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type: {noise_scheduler.config.prediction_type}")

                    diffusion_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    batch_loss = diffusion_loss

                    if sam_loss_fn is not None or lpips_loss_fn is not None:
                        alpha_bar_t = noise_scheduler.alphas_cumprod[timesteps]
                        while len(alpha_bar_t.shape) < len(noisy_latents.shape):
                            alpha_bar_t = alpha_bar_t.unsqueeze(-1)

                        pred_x0_latents = (noisy_latents - (1 - alpha_bar_t).sqrt() * model_pred) / alpha_bar_t.sqrt()
                        pred_image_latents = pred_x0_latents / tiny_vae.config.scaling_factor
                        pred_image = tiny_vae.decode(pred_image_latents.to(weight_dtype)).sample

                        sr_rgb = (pred_image.clamp(-1.0, 1.0) + 1.0) / 2.0
                        gt_rgb = (pixel_values.to(torch.float32) + 1.0) / 2.0

                        if sam_loss_fn is not None:
                            sam_loss = sam_loss_fn(sr_rgb, gt_rgb)
                            total_val_sam_loss += sam_loss.item()
                            batch_loss += args.sam_loss_weight * sam_loss

                        if lpips_loss_fn is not None:
                            lpips_loss = lpips_loss_fn(sr_rgb, gt_rgb)
                            total_val_lpips_loss += lpips_loss.item()
                            batch_loss += args.lpips_loss_weight * lpips_loss

                    total_val_diffusion_loss += diffusion_loss.item()
                    total_val_loss += batch_loss.item()

            num_batches = len(validation_dataloader)
            val_logs["val_loss"] = total_val_loss / num_batches
            val_logs["val_diffusion_loss"] = total_val_diffusion_loss / num_batches

            if sam_loss_fn is not None:
                val_logs["val_sam_loss"] = total_val_sam_loss / num_batches

            if lpips_loss_fn is not None:
                val_logs["val_lpips_loss"] = total_val_lpips_loss / num_batches

    finally:
        torch.cuda.empty_cache()
        unet.train()
        controlnet.train()

    return val_logs