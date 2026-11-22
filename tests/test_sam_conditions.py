"""
Tests for the SAM 2 inputs of the SegESR UNet/ControlNet: the training batch (variable number of masks) and
the kwargs built for each architecture.

Run with: pytest tests/test_sam_conditions.py
"""

from types import SimpleNamespace

import torch

from segesr.dataloaders.paired_dataset import collate_fn
from segesr.utils.sam_utils import pad_seg_logits, seg_logits_to_tokens
from segesr.utils.validation import get_sam_kwargs

SAM_DIM = 16

def config(sica, smca):
    return SimpleNamespace(use_sam2_image_attention=sica, use_sam2_segmentation_attention=smca)

def test_collate_pads_variable_number_of_masks():
    examples = [
        {"pixel_values": torch.zeros(3, 8, 8), "sam_seg_embeds": torch.ones(2, 256, 256)},
        {"pixel_values": torch.zeros(3, 8, 8), "sam_seg_embeds": torch.ones(5, 256, 256)},
    ]
    batch = collate_fn(examples)

    assert batch["pixel_values"].shape == (2, 3, 8, 8)
    assert batch["sam_seg_embeds"].shape == (2, 5, 256, 256)
    assert torch.all(batch["sam_seg_embeds"][0, 2:] == 0)
    torch.testing.assert_close(batch["sam_seg_embeds"], pad_seg_logits([e["sam_seg_embeds"] for e in examples]))

def test_get_sam_kwargs_builds_only_the_inputs_of_the_model():
    batch = {"sam_img_embeds": torch.randn(2, SAM_DIM, 64, 64), "sam_seg_embeds": torch.randn(2, 3, 256, 256)}

    full = get_sam_kwargs(batch, "cpu", torch.float32, config(True, True))
    torch.testing.assert_close(full["sam2_encoder_hidden_states"], batch["sam_img_embeds"])
    torch.testing.assert_close(full["sam2_segmentation_encoder_hidden_states"], seg_logits_to_tokens(batch["sam_seg_embeds"]))
    assert full["sam2_segmentation_encoder_hidden_states"].shape == (2, 128, 128)

    assert set(get_sam_kwargs(batch, "cpu", torch.float32, config(True, False))) == {"sam2_encoder_hidden_states"}
    assert set(get_sam_kwargs(batch, "cpu", torch.float32, config(False, True))) == {"sam2_segmentation_encoder_hidden_states"}
    assert get_sam_kwargs(batch, "cpu", torch.float32, config(False, False)) == {}
