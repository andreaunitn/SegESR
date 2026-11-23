"""
CPU smoke tests of the SegESR models and pipeline with tiny random weights: a training step
(ControlNet + UNet, gradient checkpointing, batched variable mask counts)
and the denoising loop (classifier-free guidance, latent tiling),
for the SeeSR baseline, PAFB and the full SegESR architecture.

No pretrained weights or GPU are needed; the SAM 2 outputs are random tensors.

Run with: pytest tests/test_pipeline_smoke.py
"""

import pytest
import torch
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import CLIPTextConfig, CLIPTextModel

from segesr.dataloaders.paired_dataset import collate_fn
from segesr.models.controlnet import ControlNetModel
from segesr.models.unet_2d_condition import UNet2DConditionModel
from segesr.pipelines.pipeline_segesr import StableDiffusionControlNetPipeline
from segesr.utils.validation import get_sam_kwargs

TEXT_DIM = 32
DAPE_DIM = 512
SAM_DIM = 1152  # last Hiera stage of SAM 2.1 large, 32x32

SEESR = {}
PAFB = dict(attention_fusion="parallel")
FULL = dict(attention_fusion="parallel", use_sam2_image_attention=True, use_sam2_segmentation_attention=True)

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

@pytest.mark.parametrize("architecture", [SEESR, PAFB, FULL], ids=["seesr", "pafb", "full"])
def test_training_step(architecture):
    unet, controlnet = make_models(**architecture)
    unet.enable_gradient_checkpointing()
    controlnet.enable_gradient_checkpointing()
    unet.train()
    controlnet.train()

    examples = [
        {
            "sam_img_embeds": torch.randn(SAM_DIM, 32, 32),
            "sam_seg_embeds": torch.randn(num_lr, 256, 256) * 5,
        }
        for num_lr in (3, 11)
    ]
    batch = collate_fn(examples)

    latents = torch.randn(2, 4, 16, 16)
    timesteps = torch.tensor([10, 900])
    text = torch.randn(2, 7, TEXT_DIM)
    dape = torch.randn(2, 4, DAPE_DIM)
    sam_kwargs = get_sam_kwargs(batch, "cpu", torch.float32, unet.config)
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
        # The fusion conv starts as the average of the four branches: the UNet SAM 2 attentions learn from the
        # first step; the ControlNet ones are in the graph but get zero gradients until its zero convs move
        for model in (unet, controlnet):
            grads = [p.grad for name, p in model.named_parameters() if "sam2_" in name]
            assert grads and all(g is not None for g in grads)
        unet_grads = [p.grad for name, p in unet.named_parameters() if "sam2_" in name and name.endswith(".weight")]
        assert any(g.abs().sum() > 0 for g in unet_grads)

@pytest.mark.parametrize(
    "architecture, size, tile_size",
    [
        (SEESR, 256, 24),   # baseline: no SAM 2 at all, tiled
        (PAFB, 128, 96),    # 16x16 latents, not tiled
        (FULL, 128, 96),
        (FULL, 256, 24),    # 32x32 latents, 24x24 tiles
    ],
    ids=["seesr-tiled", "pafb", "full", "full-tiled"],
)
def test_pipeline_denoising_loop(monkeypatch, architecture, size, tile_size):
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

    unet_inputs = []
    unet_forward = unet.forward
    def spy(*args, **kwargs):
        unet_inputs.append({k: v.shape for k, v in kwargs.items() if k.startswith("sam2_")})
        return unet_forward(*args, **kwargs)
    monkeypatch.setattr(unet, "forward", spy)

    uses_sam2 = architecture is FULL
    with torch.no_grad():
        images = pipeline(
            prompt=None, image=torch.rand(1, 3, size, size),
            prompt_embeds=torch.randn(1, 7, TEXT_DIM), negative_prompt_embeds=torch.randn(1, 7, TEXT_DIM),
            num_inference_steps=6, guidance_scale=5.0, height=size, width=size, start_point="lr", start_steps=999,
            ram_encoder_hidden_states=torch.randn(1, 4, DAPE_DIM),
            sam2_encoder_hidden_states=torch.randn(1, SAM_DIM, 32, 32) if uses_sam2 else None,
            sam2_segmentation_encoder_hidden_states=torch.randn(1, 5, 256, 256) * 5 if uses_sam2 else None,
            latent_tiled_size=tile_size, latent_tiled_overlap=4, args=True, output_type="np",
        ).images

    assert images.shape == (1, size, size, 3)
    assert torch.isfinite(torch.from_numpy(images)).all()

    # Only the SAM 2 inputs of the architecture, the same for every tile and timestep, duplicated for
    # classifier-free guidance; the 5 masks become one 128x128 map, read as 128 row tokens
    expected = {}
    if uses_sam2:
        expected = {"sam2_encoder_hidden_states": torch.Size([2, SAM_DIM, 32, 32]),
                    "sam2_segmentation_encoder_hidden_states": torch.Size([2, 128, 128])}
    assert unet_inputs == [expected] * len(unet_inputs)

def test_training_step_with_low_precision_frozen_weights():
    """As in train.py: frozen weights in the mixed-precision dtype, trainable ones float32, forward under autocast."""

    from segesr.utils.weight_utils import cast_frozen_params

    unet, controlnet = make_models(**FULL)
    unet.enable_gradient_checkpointing()
    controlnet.enable_gradient_checkpointing()
    unet.train()
    controlnet.train()

    for model in (unet, controlnet):
        model.requires_grad_(False)
        for name, param in model.named_parameters():
            if any(key in name for key in ("image_attentions", "sam2_", "fusion_conv")):
                param.requires_grad_(True)

    num_cast = cast_frozen_params([unet, controlnet], torch.bfloat16)
    assert num_cast > 0
    trainable = [p for m in (unet, controlnet) for p in m.parameters() if p.requires_grad]
    assert all(p.dtype == torch.float32 for p in trainable)
    assert all(p.dtype == torch.bfloat16 for m in (unet, controlnet) for p in m.parameters() if not p.requires_grad)

    batch = collate_fn([{"sam_img_embeds": torch.randn(SAM_DIM, 32, 32), "sam_seg_embeds": torch.randn(4, 256, 256) * 5}])
    latents = torch.randn(1, 4, 16, 16)
    sam_kwargs = get_sam_kwargs(batch, "cpu", torch.bfloat16, unet.config)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        down, mid = controlnet(latents, torch.tensor([500]), encoder_hidden_states=torch.randn(1, 7, TEXT_DIM),
                               controlnet_cond=torch.rand(1, 3, 128, 128), return_dict=False,
                               image_encoder_hidden_states=torch.randn(1, 4, DAPE_DIM), **sam_kwargs)
        model_pred = unet(latents, torch.tensor([500]), encoder_hidden_states=torch.randn(1, 7, TEXT_DIM),
                          down_block_additional_residuals=down, mid_block_additional_residual=mid,
                          image_encoder_hidden_states=torch.randn(1, 4, DAPE_DIM), **sam_kwargs).sample
    model_pred.float().square().mean().backward()

    assert torch.isfinite(model_pred.float()).all()
    assert all(p.grad is not None and p.grad.dtype == torch.float32 for p in trainable)

class FrozenImageEncoder(torch.nn.Module):
    """Stand-in for the SAM 2 image encoder: a random conv, returning the output key SamPerceptualLoss reads."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=4, stride=4)

    def forward(self, x):
        return {"vision_features": self.conv(x)}

def test_sam_loss_alone_trains_the_sam_modules():
    """L_sam alone (Eq. 6-7, as in train.py) gives non-zero gradients to the SAM 2 attentions and the fusion convs."""

    from segesr.losses.sam_loss import SamPerceptualLoss
    from segesr.utils.diffusion_utils import decode_latents_to_rgb, predict_original_latents

    unet, controlnet = make_models(**FULL)
    unet.enable_gradient_checkpointing()
    controlnet.enable_gradient_checkpointing()
    unet.train()
    controlnet.train()

    vae = AutoencoderKL(
        block_out_channels=(8, 8, 8, 8), down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4, latent_channels=4, norm_num_groups=8,
    )
    vae.requires_grad_(False)
    vae.enable_gradient_checkpointing()
    vae.train()  # gradient checkpointing of the decoder is active only in training mode, as in train.py
    sam_loss_fn = SamPerceptualLoss(FrozenImageEncoder())

    batch = collate_fn([{"sam_img_embeds": torch.randn(SAM_DIM, 32, 32), "sam_seg_embeds": torch.randn(4, 256, 256) * 5}])
    sam_kwargs = get_sam_kwargs(batch, "cpu", torch.float32, unet.config)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    noisy_latents, timesteps = torch.randn(1, 4, 16, 16), torch.tensor([300])
    text, dape = torch.randn(1, 7, TEXT_DIM), torch.randn(1, 4, DAPE_DIM)

    down, mid = controlnet(noisy_latents, timesteps, encoder_hidden_states=text, controlnet_cond=torch.rand(1, 3, 128, 128),
                           return_dict=False, image_encoder_hidden_states=dape, **sam_kwargs)
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text, down_block_additional_residuals=down,
                      mid_block_additional_residual=mid, image_encoder_hidden_states=dape, **sam_kwargs).sample

    sr_rgb = decode_latents_to_rgb(vae, predict_original_latents(noise_scheduler, noisy_latents, model_pred, timesteps), torch.float32)
    assert sr_rgb.requires_grad
    sam_loss = sam_loss_fn(sr_rgb, torch.rand(1, 3, 128, 128))
    sam_loss.backward()

    assert sam_loss.item() > 0
    for kind in ("sam2_image_attentions", "sam2_segmentation_attentions", "fusion_conv"):
        grads = [p.grad for name, p in unet.named_parameters() if kind in name and name.endswith(".weight")]
        assert grads and any(g is not None and g.abs().sum() > 0 for g in grads), kind
    # The frozen decoder and encoder only pass the gradient through
    assert all(p.grad is None for p in vae.parameters())
    assert all(p.grad is None for p in sam_loss_fn.image_encoder.parameters())
