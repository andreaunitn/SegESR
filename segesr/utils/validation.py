import os
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
from accelerate.logging import get_logger

from ram import inference_ram as inference
from segesr.utils.diffusion_utils import decode_latents_to_rgb, get_diffusion_target, predict_original_latents
from segesr.utils.sam_utils import MAX_MASKS, compute_sam2_conditions, model_uses_sam2, sam2_model_kwargs, seg_logits_to_hidden_states

logger = get_logger(__name__)

def image_grid(imgs, rows, cols):
    """Combines a list of PIL images into a single grid image."""

    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid

def get_sam_kwargs(batch, device, dtype, config, mask_size, use_clean=None):
    """
    Builds, from a training batch, the SAM 2 inputs of a SegESR UNet/ControlNet with config `config`
    (an empty dict for a model without SAM 2 attentions).

    Args:
        mask_size (tuple): (h, w) latent resolution of the segment masks.
        use_clean (torch.BoolTensor, optional): (B,) samples that use the SAM 2 conditions of the GT
            image ('sam_img_embeds_gt' / 'sam_seg_embeds_gt') instead of the ones of the LR image.
    """

    if not model_uses_sam2(config):
        return {}

    img_embeds = batch["sam_img_embeds"].to(device)
    seg_logits = batch["sam_seg_embeds"].to(device)

    if use_clean is not None and use_clean.any():
        clean_img_embeds = batch["sam_img_embeds_gt"].to(device)
        clean_seg_logits = batch["sam_seg_embeds_gt"].to(device)

        # Pad both mask sets to the same number of (all-zero, i.e. empty) masks before selecting
        num_masks = max(seg_logits.shape[1], clean_seg_logits.shape[1])
        seg_logits = F.pad(seg_logits, (0, 0, 0, 0, 0, num_masks - seg_logits.shape[1]))
        clean_seg_logits = F.pad(clean_seg_logits, (0, 0, 0, 0, 0, num_masks - clean_seg_logits.shape[1]))

        select = use_clean.to(device)[:, None, None, None]
        img_embeds = torch.where(select, clean_img_embeds, img_embeds)
        seg_logits = torch.where(select, clean_seg_logits, seg_logits)

    return sam2_model_kwargs(config, img_embeds, seg_logits, mask_size, dtype)

@torch.no_grad()
def prepare_validation_conditions(image_path, ram_model, sam_generator, device):
    """
    Computes once, before training, the conditions of the fixed validation image: RAM tags and DAPE
    embeddings, and the SAM 2 conditions if `sam_generator` is given. They are kept on the CPU, so the
    RAM model does not need to stay on the GPU during training.
    """

    val_image = Image.open(image_path).convert("RGB")
    ram_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    lq_for_ram = ram_transforms(transforms.ToTensor()(val_image).unsqueeze(0).to(device))

    ram_model.to(device)
    conditions = {
        "image": val_image,
        "tags": inference(lq_for_ram, ram_model)[0],
        "ram_embeds": ram_model.generate_image_embeds(lq_for_ram).cpu(),
    }
    ram_model.to("cpu")

    if sam_generator is not None:
        sam_img_embeds, sam_seg_logits = compute_sam2_conditions(val_image, sam_generator, max_masks=MAX_MASKS)
        conditions["sam2_encoder_hidden_states"] = sam_img_embeds
        conditions["sam2_segmentation_encoder_hidden_states"] = seg_logits_to_hidden_states(sam_seg_logits)

    return conditions

def validation(
        unet,
        controlnet,
        vae,
        text_encoder,
        tokenizer,
        noise_scheduler,
        tiny_vae,
        validation_conditions,
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
    if args.generate_validation_image and accelerator.is_main_process and validation_conditions is not None:
        # Imported here: the pipeline itself imports `segesr.utils`
        from segesr.pipelines.pipeline_segesr import StableDiffusionControlNetPipeline

        pipeline = StableDiffusionControlNetPipeline(
            vae=accelerator.unwrap_model(vae),
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            unet=accelerator.unwrap_model(unet),
            controlnet=accelerator.unwrap_model(controlnet),
            scheduler=noise_scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False
        )
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        # Conditions of the validation image, computed once before training (prepare_validation_conditions)
        val_image = validation_conditions["image"]
        ram_embeds = validation_conditions["ram_embeds"].to(accelerator.device)

        sam_kwargs = {}
        if model_uses_sam2(unet.config):
            sam_kwargs = {
                key: validation_conditions[key].to(accelerator.device)
                for key in ("sam2_encoder_hidden_states", "sam2_segmentation_encoder_hidden_states")
            }

        user_prompt = args.validation_prompt[0] if args.validation_prompt else ""
        final_prompt = ", ".join(p for p in [validation_conditions["tags"], user_prompt, "clean, high-resolution, 8k"] if p)
        negative_prompt = "dotted, noise, blur, lowres, smooth"

        width, height = val_image.size
        cond_image = val_image.resize((width * 4, height * 4))

        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed if args.seed is not None else 42)
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
                **sam_kwargs,
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
                    sam_kwargs = get_sam_kwargs(val_batch, accelerator.device, weight_dtype, unet.config, mask_size=latents.shape[-2:])

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

                    target = get_diffusion_target(noise_scheduler, latents, noise, timesteps)
                    diffusion_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    batch_loss = diffusion_loss

                    if sam_loss_fn is not None or lpips_loss_fn is not None:
                        pred_x0_latents = predict_original_latents(noise_scheduler, noisy_latents, model_pred, timesteps)
                        sr_rgb = decode_latents_to_rgb(tiny_vae, pred_x0_latents, weight_dtype)
                        gt_rgb = (pixel_values.to(torch.float32) + 1.0) / 2.0

                        if sam_loss_fn is not None:
                            sam_loss = sam_loss_fn(sr_rgb, gt_rgb)
                            total_val_sam_loss += sam_loss.item()
                            batch_loss = batch_loss + args.sam_loss_weight * sam_loss

                        if lpips_loss_fn is not None:
                            lpips_loss = lpips_loss_fn(sr_rgb, gt_rgb)
                            total_val_lpips_loss += lpips_loss.item()
                            batch_loss = batch_loss + args.lpips_loss_weight * lpips_loss

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
