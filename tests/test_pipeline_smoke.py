"""
CPU smoke tests of the SegESR models and pipeline with tiny random weights: a training step
(ControlNet + UNet, gradient checkpointing, batched variable mask counts, clean SAM 2 conditions)
and the denoising loop (classifier-free guidance, latent tiling, segment routing, SAM 2 refresh),
for the SeeSR baseline, PAFB and the full SegESR architecture.

No pretrained weights or GPU are needed; SAM 2 is replaced by a stub.

Run with: pytest tests/test_pipeline_smoke.py
"""

import pytest
import torch
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import CLIPTextConfig, CLIPTextModel

import segesr.pipelines.pipeline_segesr as pipeline_module
from segesr.dataloaders.paired_dataset import collate_fn
from segesr.models.controlnet import ControlNetModel
from segesr.models.unet_2d_condition import UNet2DConditionModel
from segesr.pipelines.pipeline_segesr import StableDiffusionControlNetPipeline
from segesr.utils.validation import get_sam_kwargs

TEXT_DIM = 32
DAPE_DIM = 512
SAM_DIM = 256

SEESR = {}
PAFB = dict(attention_fusion="parallel")
FULL = dict(attention_fusion="parallel", use_sam2_image_attention=True, use_sam2_segmentation_attention=True, segment_routing=True)
NO_ROUTING = {**FULL, "segment_routing": False}

def make_models(**architecture):
    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        sample_size=16,
        block_out_channels=(32, 64),
        layers_per_block=1,
        cross_attention_dim=TEXT_DIM,
        attention_head_dim=8,
        norm_num_groups=8,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        use_image_cross_attention=True,
        **architecture,
    )
    controlnet = ControlNetModel.from_unet(unet, use_image_cross_attention=True)
    return unet, controlnet

def stub_sam_conditions(image, sam_generator, max_masks=None):
    """Stands in for `compute_sam2_conditions`: 3 masks (top half, bottom half, left strip)."""

    logits = torch.full((3, 1, 256, 256), -10.0)
    logits[0, :, :128] = 10.0
    logits[1, :, 128:] = 10.0
    logits[2, :, :, :100] = 10.0
    return torch.randn(1, SAM_DIM, 64, 64), logits

@pytest.mark.parametrize("architecture", [SEESR, PAFB, FULL], ids=["seesr", "pafb", "full"])
def test_training_step(architecture):
    unet, controlnet = make_models(**architecture)
    unet.enable_gradient_checkpointing()
    controlnet.enable_gradient_checkpointing()
    unet.train()
    controlnet.train()

    examples = [
        {
            "sam_img_embeds": torch.randn(SAM_DIM, 64, 64),
            "sam_seg_embeds": torch.randn(num_lr, 256, 256) * 5,
            "sam_img_embeds_gt": torch.randn(SAM_DIM, 64, 64),
            "sam_seg_embeds_gt": torch.randn(num_gt, 256, 256) * 5,
        }
        for num_lr, num_gt in [(3, 7), (11, 2)]
    ]
    batch = collate_fn(examples)

    latents = torch.randn(2, 4, 16, 16)
    timesteps = torch.tensor([10, 900])
    text = torch.randn(2, 7, TEXT_DIM)
    dape = torch.randn(2, 4, DAPE_DIM)
    sam_kwargs = get_sam_kwargs(batch, "cpu", torch.float32, unet.config, mask_size=latents.shape[-2:], use_clean=torch.tensor([True, False]))
    assert bool(sam_kwargs) == (architecture is FULL)

    down_block_res_samples, mid_block_res_sample = controlnet(
        latents, timesteps, encoder_hidden_states=text, controlnet_cond=torch.rand(2, 3, 128, 128),
        return_dict=False, image_encoder_hidden_states=dape, **sam_kwargs,
    )
    model_pred = unet(
        latents, timesteps, encoder_hidden_states=text,
        down_block_additional_residuals=down_block_res_samples, mid_block_additional_residual=mid_block_res_sample,
        image_encoder_hidden_states=dape, **sam_kwargs,
    ).sample
    model_pred.square().mean().backward()
    assert model_pred.shape == latents.shape

    if architecture is not SEESR:
        # The fusion convs learn from the first step
        assert all(m.fusion_conv.weight.grad.abs().sum() > 0 for m in unet.modules() if hasattr(m, "fusion_conv"))

    if architecture is FULL:
        # The SAM 2 branches start with zero fusion weight: their attentions get gradients once the fusion
        # conv has moved, but they are already part of the graph
        for model in (unet, controlnet):
            grads = [p.grad for name, p in model.named_parameters() if "sam2_" in name]
            assert grads and all(g is not None for g in grads)

@pytest.mark.parametrize(
    "architecture, size, tile_size, refresh_timesteps",
    [
        (SEESR, 256, 24, [700, 400]),       # baseline: no SAM 2 at all, the refresh is ignored
        (PAFB, 128, 96, None),              # 16x16 latents, not tiled
        (FULL, 128, 96, None),
        (FULL, 256, 24, [700, 400]),        # 32x32 latents, 24x24 tiles, two SAM 2 refreshes
        (NO_ROUTING, 256, 24, [700, 400]),  # SAM 2 attentions without segment routing
    ],
    ids=["seesr-tiled-refresh", "pafb", "full", "full-tiled-refresh", "no-routing-tiled-refresh"],
)
def test_pipeline_denoising_loop(monkeypatch, architecture, size, tile_size, refresh_timesteps):
    unet, controlnet = make_models(**architecture)
    unet.eval()
    controlnet.eval()
    vae = AutoencoderKL(
        block_out_channels=(8, 8, 8, 8), down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4, latent_channels=4, norm_num_groups=8,
    ).eval()
    text_encoder = CLIPTextModel(CLIPTextConfig(hidden_size=TEXT_DIM, intermediate_size=37, num_attention_heads=4, num_hidden_layers=1)).eval()

    pipeline = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=None, unet=unet, controlnet=controlnet,
        scheduler=DDPMScheduler(num_train_timesteps=1000), safety_checker=None, feature_extractor=None,
        requires_safety_checker=False,
    )
    pipeline.set_progress_bar_config(disable=True)

    sam_calls = []
    def sam_stub(image, sam_generator, max_masks=None):
        sam_calls.append(image.size)
        return stub_sam_conditions(image, sam_generator, max_masks)
    monkeypatch.setattr(pipeline_module, "compute_sam2_conditions", sam_stub)

    unet_inputs = []
    unet_forward = unet.forward
    def spy(*args, **kwargs):
        unet_inputs.append({k: v.shape for k, v in kwargs.items() if k.startswith("sam2_")})
        return unet_forward(*args, **kwargs)
    monkeypatch.setattr(unet, "forward", spy)

    uses_sam2 = architecture in (FULL, NO_ROUTING)
    with torch.no_grad():
        images = pipeline(
            prompt=None, image=torch.rand(1, 3, size, size),
            prompt_embeds=torch.randn(1, 7, TEXT_DIM), negative_prompt_embeds=torch.randn(1, 7, TEXT_DIM),
            num_inference_steps=6, guidance_scale=5.0, height=size, width=size, start_point="lr", start_steps=999,
            ram_encoder_hidden_states=torch.randn(1, 4, DAPE_DIM),
            sam2_encoder_hidden_states=torch.randn(1, SAM_DIM, 64, 64) if uses_sam2 else None,
            sam2_segmentation_encoder_hidden_states=torch.randn(1, 5, 256, 256) * 5 if uses_sam2 else None,
            sam_generator=object(), sam_refresh_timesteps=refresh_timesteps,
            latent_tiled_size=tile_size, latent_tiled_overlap=4, args=True, output_type="np",
        ).images

    assert images.shape == (1, size, size, 3)
    assert torch.isfinite(torch.from_numpy(images)).all()

    # One SAM 2 refresh per threshold, on the full-size decoded image, for models with SAM 2 attentions only
    assert sam_calls == ([(size, size)] * len(refresh_timesteps or []) if uses_sam2 else [])

    # Only the SAM 2 inputs of the architecture; masks of the current tile, duplicated for classifier-free guidance
    latent_tile = min(tile_size, size // 8)
    expected = {}
    if uses_sam2:
        expected = {"sam2_encoder_hidden_states": torch.Size([2, SAM_DIM, 64, 64]),
                    "sam2_segmentation_encoder_hidden_states": torch.Size([2, 8, SAM_DIM])}
        if architecture["segment_routing"]:
            expected["sam2_segmentation_masks"] = torch.Size([2, 8, latent_tile, latent_tile])
    assert unet_inputs == [expected] * len(unet_inputs)
