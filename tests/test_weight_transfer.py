"""
Tests for the DAPE -> SAM 2 attention weight transfer (segesr.utils.weight_utils).

They build tiny SegESR cross-attention blocks on CPU, where the SAM 2 attentions have a smaller
cross-attention dim than the DAPE ones, so both the direct copy and the slicing path are exercised.

Run with: pytest tests/test_weight_transfer.py
"""

import pytest
import torch
from torch import nn

from accelerate import Accelerator

from segesr.models.unet_2d_blocks import CrossAttnDownBlock2D
from segesr.utils.weight_utils import init_sam_weights, unfreeze_params, verify_weights

SAM_MODULES = ["sam2_image_attentions", "sam2_segmentation_attentions"]

CHANNELS = 32
DAPE_DIM = 64
SAM_DIM = 32

def make_block(use_sam2=True):
    return CrossAttnDownBlock2D(
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        temb_channels=CHANNELS,
        num_layers=1,
        resnet_groups=8,
        num_attention_heads=2,
        cross_attention_dim=16,
        add_downsample=False,
        use_image_cross_attention=True,
        image_cross_attention_dim=DAPE_DIM,
        use_sam2=use_sam2,
        seg_cross_attention_dim=SAM_DIM,
    )

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_blocks = nn.ModuleList([make_block(use_sam2=True), make_block(use_sam2=False)])

@pytest.fixture(scope="module")
def accelerator():
    return Accelerator(cpu=True)

@pytest.fixture
def model():
    torch.manual_seed(0)
    return TinyModel()

def expected_from_source(source_param, target_param):
    """Mirror of the copy/slice rule used by init_sam_weights."""

    if source_param.shape == target_param.shape:
        return source_param
    if target_param.dim() > 1:
        return source_param[:, :target_param.shape[1]]
    return source_param[:target_param.shape[0]]

def test_sam_attentions_are_initialized_from_dape(model, accelerator):
    init_sam_weights(model, accelerator, SAM_MODULES)

    block = model.down_blocks[0]
    source_params = dict(block.image_attentions[0].named_parameters())

    sliced_keys = 0
    for module_name in SAM_MODULES:
        target_params = dict(getattr(block, module_name)[0].named_parameters())
        assert target_params.keys() == source_params.keys()

        for name, target_param in target_params.items():
            source_param = source_params[name]
            if source_param.shape != target_param.shape:
                sliced_keys += 1
            torch.testing.assert_close(target_param, expected_from_source(source_param, target_param))

    # The cross-attention key/value projections take SAM_DIM inputs instead of DAPE_DIM
    assert sliced_keys > 0

def test_init_does_not_touch_dape_attentions(model, accelerator):
    before = {k: v.clone() for k, v in model.down_blocks[0].image_attentions.state_dict().items()}

    init_sam_weights(model, accelerator, SAM_MODULES)

    after = model.down_blocks[0].image_attentions.state_dict()
    for key, value in before.items():
        torch.testing.assert_close(after[key], value)

def test_init_only_touches_requested_modules(model, accelerator):
    block = model.down_blocks[0]
    seg_before = {k: v.clone() for k, v in block.sam2_segmentation_attentions.state_dict().items()}

    init_sam_weights(model, accelerator, ["sam2_image_attentions"])

    seg_after = block.sam2_segmentation_attentions.state_dict()
    for key, value in seg_before.items():
        torch.testing.assert_close(seg_after[key], value)

def test_verify_weights_passes_after_init(model, accelerator):
    init_sam_weights(model, accelerator, SAM_MODULES)
    assert verify_weights(model, accelerator, SAM_MODULES) is True

def test_verify_weights_fails_on_random_init(model, accelerator):
    assert verify_weights(model, accelerator, SAM_MODULES) is False

def test_verify_weights_fails_when_no_module_matches(model, accelerator):
    assert verify_weights(model, accelerator, ["not_an_attention_module"]) is False

def test_blocks_without_sam_are_skipped(model, accelerator):
    block_without_sam = model.down_blocks[1]
    assert not hasattr(block_without_sam, "sam2_image_attentions")

    # Must not raise on blocks that have no SAM 2 attentions
    init_sam_weights(model, accelerator, SAM_MODULES)

def test_unfreeze_params_respects_exclusions(model, accelerator):
    model.requires_grad_(False)

    # Tag attentions only: DAPE and SAM 2 attentions end with "attentions" too and must be excluded
    assert unfreeze_params(model, "Tiny", "attentions", exclude_keywords=["image", "sam2"])

    block = model.down_blocks[0]
    assert all(p.requires_grad for p in block.attentions.parameters())
    assert not any(p.requires_grad for p in block.image_attentions.parameters())
    assert not any(p.requires_grad for p in block.sam2_image_attentions.parameters())
    assert not any(p.requires_grad for p in block.sam2_segmentation_attentions.parameters())

def test_unfreeze_params_reports_missing_modules(model, accelerator):
    model.requires_grad_(False)
    assert unfreeze_params(model, "Tiny", "does_not_exist") is False
    assert not any(p.requires_grad for p in model.parameters())
