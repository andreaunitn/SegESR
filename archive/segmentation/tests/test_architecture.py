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
from segesr.utils.sam_utils import build_segment_conditions

REPO_ROOT = Path(__file__).resolve().parents[1]

PAFB = dict(attention_fusion="parallel")
FULL = dict(attention_fusion="parallel", use_sam2_image_attention=True, use_sam2_segmentation_attention=True, segment_routing=True)

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
    embeds = torch.randn(1, 256, 64, 64, generator=generator)
    tokens, masks = build_segment_conditions(embeds, torch.randn(1, 5, 256, 256, generator=generator) * 5, (16, 16))
    return dict(
        sample=torch.randn(1, 4, 16, 16, generator=generator),
        timestep=torch.tensor([500]),
        encoder_hidden_states=torch.randn(1, 7, 32, generator=generator),
        image_encoder_hidden_states=torch.randn(1, 4, 512, generator=generator),
        sam2_encoder_hidden_states=embeds,
        sam2_segmentation_encoder_hidden_states=tokens,
        sam2_segmentation_masks=masks,
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

def test_pafb_fusion_starts_as_average_of_text_and_dape():
    block = make_unet(**PAFB).down_blocks[0]
    weight, channels = block.fusion_conv.weight, block.fusion_conv.out_channels
    index = torch.arange(channels)

    assert weight.shape[1] == 2 * channels
    assert torch.all(weight[index, index, 1, 1] == 0.5) and torch.all(weight[index, index + channels, 1, 1] == 0.5)
    assert weight.abs().sum() == channels  # nothing else
    assert torch.all(block.fusion_conv.bias == 0)

@pytest.mark.parametrize("architecture", [
    dict(PAFB, use_sam2_image_attention=True),
    dict(PAFB, use_sam2_segmentation_attention=True),
    FULL,
], ids=["sica", "smca", "full"])
def test_sam2_components_start_without_influence(architecture):
    """With the same shared weights, adding SAM 2 attentions does not change the PAFB output at initialization."""

    pafb, model = make_unet(**PAFB), make_unet(**architecture)

    model.load_state_dict({k: v for k, v in pafb.state_dict().items() if "fusion_conv.weight" not in k}, strict=False)
    pafb_modules = dict(pafb.named_modules())
    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, "fusion_conv"):
                text_dape_weight = pafb_modules[name].fusion_conv.weight
                module.fusion_conv.weight[:, : text_dape_weight.shape[1]] = text_dape_weight

        inputs = unet_inputs()
        torch.testing.assert_close(model(**inputs).sample, pafb(**inputs).sample)

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
    with pytest.raises(ValueError, match="sam2_segmentation_masks"):
        make_unet(**FULL)(**{**inputs, "sam2_segmentation_masks": None})
    with pytest.raises(ValueError, match="sam2_encoder_hidden_states"):
        make_unet(**FULL)(**{**inputs, "sam2_encoder_hidden_states": None})

    # A model without segment routing ignores the masks
    no_routing = make_unet(**{**FULL, "segment_routing": False})
    with torch.no_grad():
        torch.testing.assert_close(no_routing(**inputs).sample, no_routing(**{**inputs, "sam2_segmentation_masks": None}).sample)

def test_segesr_config_builds_on_the_seesr_config():
    seesr = load_config(REPO_ROOT / "configs" / "train_seesr.yaml")
    segesr = load_config(REPO_ROOT / "configs" / "train_segesr.yaml")

    assert seesr["attention_fusion"] == "sequential"
    assert not any(seesr[key] for key in ("use_sam2_image_attention", "use_sam2_segmentation_attention", "segment_routing", "use_sam_loss"))
    for key, value in FULL.items():
        assert segesr[key] == value
    assert segesr["use_sam_loss"] and segesr["clean_sam_prob"] > 0

    # Every other setting is shared, so the two runs differ only by the SegESR components
    components = set(FULL) | {"use_sam_loss", "clean_sam_prob", "output_dir"}
    assert {k: v for k, v in seesr.items() if k not in components} == {k: v for k, v in segesr.items() if k not in components}
    assert "base_config" not in segesr

def test_resuming_a_checkpoint_with_another_architecture_fails(tmp_path):
    """Routing on/off has identical parameters: without the check, a run would silently resume another variant."""

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

    accelerator, _ = prepared(**{**FULL, "segment_routing": False})
    with pytest.raises(ValueError, match="segment_routing"):
        accelerator.load_state(tmp_path / "checkpoint-10")
