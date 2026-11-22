"""
Tests for the modular SegESR architecture: the SeeSR default, the PAFB fusion and the SAM 2 components
switched on one at a time, their initialization, and how the architecture is stored in checkpoints.

Run with: pytest tests/test_architecture.py
"""

from pathlib import Path

import pytest
import torch

from segesr.models.controlnet import ControlNetModel
from segesr.models.unet_2d_blocks import CrossAttnDownBlock2D
from segesr.models.unet_2d_condition import UNet2DConditionModel
from segesr.utils.config import load_config
from segesr.models.unet_2d_blocks import SAM2_IMAGE_EMBED_DIM, SAM2_MASK_MAP_SIZE
from segesr.utils.sam_utils import MASK_MAP_SIZE, aggregate_mask_logits, seg_logits_to_tokens

REPO_ROOT = Path(__file__).resolve().parents[1]

PAFB = dict(attention_fusion="parallel")
FULL = dict(attention_fusion="parallel", use_sam2_image_attention=True, use_sam2_segmentation_attention=True)

def make_unet(**architecture):
    return UNet2DConditionModel(
        sample_size=16,
        block_out_channels=(32, 64),
        layers_per_block=1,
        cross_attention_dim=32,
        attention_head_dim=8,
        norm_num_groups=8,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        use_image_cross_attention=True,
        **architecture,
    ).eval()

def unet_inputs():
    generator = torch.Generator().manual_seed(0)
    embeds = torch.randn(1, SAM2_IMAGE_EMBED_DIM, 32, 32, generator=generator)
    tokens = seg_logits_to_tokens(torch.randn(1, 5, 256, 256, generator=generator) * 5)
    return dict(
        sample=torch.randn(1, 4, 16, 16, generator=generator),
        timestep=torch.tensor([500]),
        encoder_hidden_states=torch.randn(1, 7, 32, generator=generator),
        image_encoder_hidden_states=torch.randn(1, 4, 512, generator=generator),
        sam2_encoder_hidden_states=embeds,
        sam2_segmentation_encoder_hidden_states=tokens,
    )

def test_sequential_fusion_is_seesr():
    """Default block = SeeSR: resnet, text attention, then DAPE attention on its output."""

    torch.manual_seed(0)
    block = CrossAttnDownBlock2D(
        in_channels=32, out_channels=32, temb_channels=32, num_layers=1, resnet_groups=8, num_attention_heads=2,
        cross_attention_dim=8, add_downsample=False, use_image_cross_attention=True, image_cross_attention_dim=16,
    ).eval()
    hidden_states, temb = torch.randn(1, 32, 8, 8), torch.randn(1, 32)
    text, dape = torch.randn(1, 4, 8), torch.randn(1, 3, 16)

    with torch.no_grad():
        output = block(hidden_states, temb, encoder_hidden_states=text, image_encoder_hidden_states=dape)[0]
        expected = block.resnets[0](hidden_states, temb)
        expected = block.attentions[0](expected, encoder_hidden_states=text, return_dict=False)[0]
        expected = block.image_attentions[0](expected, encoder_hidden_states=dape, return_dict=False)[0]

    torch.testing.assert_close(output, expected)
    assert not hasattr(block, "fusion_conv")

@pytest.mark.parametrize("architecture, num_branches", [
    (PAFB, 2),
    (dict(PAFB, use_sam2_image_attention=True), 3),
    (dict(PAFB, use_sam2_segmentation_attention=True), 3),
    (FULL, 4),
], ids=["pafb", "sica", "smca", "full"])
def test_pafb_fusion_starts_as_average_of_its_branches(architecture, num_branches):
    """As in the ECCV 2026 model, the fusion conv starts as the plain average of all the attention branches."""

    block = make_unet(**architecture).down_blocks[0]
    weight, channels = block.fusion_conv.weight, block.fusion_conv.out_channels
    index = torch.arange(channels)

    assert weight.shape[1] == num_branches * channels
    for branch in range(num_branches):
        assert torch.all(weight[index, index + branch * channels, 1, 1] == 1.0 / num_branches)
    assert torch.isclose(weight.abs().sum(), torch.tensor(float(channels)))  # nothing else
    assert torch.all(block.fusion_conv.bias == 0)

def test_smca_attends_to_one_token_per_logit_row():
    """SMCA reads the aggregated 128x128 mask map as 128 row tokens: one more mask changes the output."""

    torch.manual_seed(0)
    unet, inputs = make_unet(**FULL), unet_inputs()
    assert inputs["sam2_segmentation_encoder_hidden_states"].shape == (1, 128, 128)

    more = torch.randn(1, 6, 256, 256) * 5
    with torch.no_grad():
        output = unet(**inputs).sample
        other = unet(**{**inputs, "sam2_segmentation_encoder_hidden_states": seg_logits_to_tokens(more)}).sample
    assert output.shape == other.shape and not torch.allclose(output, other)

def test_mask_logits_are_aggregated_as_in_eq_3():
    """Bilinear upsampling to H x W, then pixel-wise maximum over the N masks; padding maps are ignored."""

    assert MASK_MAP_SIZE == SAM2_MASK_MAP_SIZE == 128

    logits = torch.full((1, 3, 256, 256), -5.0)
    logits[0, 0, :128] = 4.0      # top half
    logits[0, 1, :, :128] = 2.0   # left half
    logits[0, 2] = 0.0            # batch padding: all zeros
    aggregated = aggregate_mask_logits(logits)

    assert aggregated.shape == (1, 128, 128)
    assert aggregated[0, 10, 100] == 4.0    # top right: mask 0
    assert aggregated[0, 100, 10] == 2.0    # bottom left: mask 1
    assert aggregated[0, 100, 100] == -5.0  # bottom right: no mask, the padding map (0) is not the maximum
    torch.testing.assert_close(seg_logits_to_tokens(logits), aggregated)

    assert torch.all(aggregate_mask_logits(torch.zeros(2, 4, 256, 256)) == 0)  # no mask at all

def test_architecture_is_saved_with_the_checkpoint(tmp_path):
    torch.manual_seed(0)
    unet = make_unet(**FULL)
    unet.save_pretrained(tmp_path / "unet")
    loaded = UNet2DConditionModel.from_pretrained(tmp_path / "unet").eval()

    for key, value in FULL.items():
        assert loaded.config[key] == value
    with torch.no_grad():
        inputs = unet_inputs()
        torch.testing.assert_close(loaded(**inputs).sample, unet(**inputs).sample)

def test_controlnet_copies_the_unet_architecture():
    controlnet = ControlNetModel.from_unet(make_unet(**FULL), use_image_cross_attention=True)
    for key, value in FULL.items():
        assert controlnet.config[key] == value
    assert hasattr(controlnet.mid_block, "sam2_segmentation_attentions")

def test_invalid_architectures_are_rejected():
    with pytest.raises(ValueError, match="parallel"):
        make_unet(use_sam2_image_attention=True)
    with pytest.raises(ValueError, match="attention_fusion"):
        make_unet(attention_fusion="concat")

def test_missing_sam2_inputs_are_rejected():
    inputs = unet_inputs()
    with pytest.raises(ValueError, match="sam2_segmentation_encoder_hidden_states"):
        make_unet(**FULL)(**{**inputs, "sam2_segmentation_encoder_hidden_states": None})
    with pytest.raises(ValueError, match="sam2_encoder_hidden_states"):
        make_unet(**FULL)(**{**inputs, "sam2_encoder_hidden_states": None})

def test_segesr_configs_follow_the_paper():
    """Stage 1 and stage 2 of the ECCV 2026 training (Sec. 4.1), on top of the SeeSR baseline config."""

    seesr = load_config(REPO_ROOT / "configs" / "train_seesr.yaml")
    stage1 = load_config(REPO_ROOT / "configs" / "train_segesr.yaml")
    stage2 = load_config(REPO_ROOT / "configs" / "train_segesr_stage2.yaml")

    assert seesr["attention_fusion"] == "sequential"
    assert not any(seesr[key] for key in ("use_sam2_image_attention", "use_sam2_segmentation_attention", "use_sam_loss"))

    for stage in (stage1, stage2):
        for key, value in FULL.items():
            assert stage[key] == value
        assert stage["use_sam_loss"] and stage["sam_loss_weight"] == 0.1
        assert stage["train_batch_size"] == 1 and stage["gradient_accumulation_steps"] == 4
        assert "base_config" not in stage

    # Stage 1: only the new modules, 20K steps at 9e-7
    assert stage1["max_train_steps"] == 20000 and stage1["finetune_lr"] == 9.0e-7
    assert not stage1["train_unet_dape_attention"] and not stage1["train_controlnet_dape_attention"]
    assert stage1["init_sam_from_dape"]

    # Stage 2: everything, 28K steps from the last stage 1 checkpoint, 2e-7 (SeeSR) and 5e-7 (new)
    assert stage2["max_train_steps"] == 28000
    assert stage2["learning_rate"] == 2.0e-7 and stage2["finetune_lr"] == 5.0e-7
    assert stage2["train_unet_dape_attention"] and stage2["train_controlnet_dape_attention"]
    assert not stage2["init_sam_from_dape"]
    stage1_last = f"{stage1['output_dir']}/checkpoint-{stage1['max_train_steps']}"
    assert stage2["seesr_model_path"] == stage2["controlnet_model_name_or_path"] == stage1_last
    assert stage2["output_dir"] != stage1["output_dir"]

def test_resuming_a_checkpoint_with_another_architecture_fails(tmp_path):
    from accelerate import Accelerator
    from segesr.utils.checkpoint_utils import register_checkpoint_hooks

    def prepared(**architecture):
        accelerator = Accelerator(cpu=True)
        register_checkpoint_hooks(accelerator)
        unet = make_unet(**architecture)
        controlnet = ControlNetModel.from_unet(unet, use_image_cross_attention=True)
        return accelerator, accelerator.prepare(unet, controlnet)

    accelerator, _ = prepared(**FULL)
    accelerator.save_state(tmp_path / "checkpoint-10")

    accelerator, _ = prepared(**FULL)
    accelerator.load_state(tmp_path / "checkpoint-10")  # same architecture: fine

    accelerator, _ = prepared(**{**FULL, "use_sam2_segmentation_attention": False})
    with pytest.raises(ValueError, match="use_sam2_segmentation_attention"):
        accelerator.load_state(tmp_path / "checkpoint-10")
