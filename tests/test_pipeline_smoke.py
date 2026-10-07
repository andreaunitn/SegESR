"""
CPU smoke tests of the SegESR models and pipeline with tiny random weights: a training step
(ControlNet + UNet, gradient checkpointing, batched variable mask counts, clean SAM 2 conditions)
and the denoising loop (classifier-free guidance, latent tiling, segment routing, SAM 2 refresh).

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

def make_models():
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

def test_training_step_backpropagates_into_segmentation_attention():
    unet, controlnet = make_models()
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
    sam_kwargs = get_sam_kwargs(batch, "cpu", torch.float32, True, mask_size=latents.shape[-2:], use_clean=torch.tensor([True, False]))

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

    unet_grads = [p.grad for name, p in unet.named_parameters() if "sam2_segmentation_attentions" in name]
    assert unet_grads and all(g is not None and g.abs().sum() > 0 for g in unet_grads)

    # The ControlNet output convs are zero-initialized, so its inner gradients are exactly zero at the
    # first step: only check that the segmentation attentions are part of the graph
    controlnet_grads = [p.grad for name, p in controlnet.named_parameters() if "sam2_segmentation_attentions" in name]
    assert controlnet_grads and all(g is not None for g in controlnet_grads)

@pytest.mark.parametrize(
    "size, tile_size, refresh_timesteps, segment_routing",
    [
        (128, 96, None, True),          # 16x16 latents, not tiled, no refresh
        (256, 24, [700, 400], True),    # 32x32 latents, 24x24 tiles, two SAM 2 refreshes
        (256, 24, [700, 400], False),   # same, without segment routing
    ],
)
def test_pipeline_denoising_loop(monkeypatch, size, tile_size, refresh_timesteps, segment_routing):
    unet, controlnet = make_models()
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

    unet_masks = []
    unet_forward = unet.forward
    def spy(*args, **kwargs):
        masks = kwargs["sam2_segmentation_masks"]
        unet_masks.append(None if masks is None else masks.shape)
        return unet_forward(*args, **kwargs)
    monkeypatch.setattr(unet, "forward", spy)

    with torch.no_grad():
        images = pipeline(
            prompt=None, image=torch.rand(1, 3, size, size),
            prompt_embeds=torch.randn(1, 7, TEXT_DIM), negative_prompt_embeds=torch.randn(1, 7, TEXT_DIM),
            num_inference_steps=6, guidance_scale=5.0, height=size, width=size, start_point="lr", start_steps=999,
            ram_encoder_hidden_states=torch.randn(1, 4, DAPE_DIM),
            sam2_encoder_hidden_states=torch.randn(1, SAM_DIM, 64, 64),
            sam2_segmentation_encoder_hidden_states=torch.randn(1, 5, 256, 256) * 5,
            segment_routing=segment_routing,
            sam_generator=object() if refresh_timesteps else None, sam_refresh_timesteps=refresh_timesteps,
            latent_tiled_size=tile_size, latent_tiled_overlap=4, args=True, output_type="np",
        ).images

    assert images.shape == (1, size, size, 3)
    assert torch.isfinite(torch.from_numpy(images)).all()

    # One SAM 2 refresh per threshold, on the full-size decoded image
    assert sam_calls == [(size, size)] * len(refresh_timesteps or [])

    # Masks of the current tile (or of the whole latents), duplicated for classifier-free guidance
    latent_tile = min(tile_size, size // 8)
    expected = [torch.Size([2, 8, latent_tile, latent_tile]) if segment_routing else None] * len(unet_masks)
    assert unet_masks == expected
