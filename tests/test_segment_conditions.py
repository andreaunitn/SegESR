"""
Tests for the SAM 2 segment tokens (segesr.utils.sam_utils.build_segment_conditions) and the
segment-routed segmentation attention of the SegESR UNet blocks.

Run with: pytest tests/test_segment_conditions.py
"""

import pytest
import torch

from segesr.dataloaders.paired_dataset import collate_fn
from segesr.models.unet_2d_blocks import CrossAttnDownBlock2D, segment_routed_attention
from segesr.utils.sam_utils import SEGMENT_TOKENS_MULTIPLE, build_segment_conditions, pad_seg_logits
from segesr.utils.validation import get_sam_kwargs

CHANNELS = 32
SAM_DIM = 16
LOGIT = 20.0  # saturated mask logits, so sigmoid(logits) is ~binary

def half_masks(size=256):
    """Two disjoint masks: left and right half of the image, as (1, 2, size, size) logits."""

    logits = torch.full((1, 2, size, size), -LOGIT)
    logits[:, 0, :, : size // 2] = LOGIT
    logits[:, 1, :, size // 2:] = LOGIT
    return logits

def make_block():
    return CrossAttnDownBlock2D(
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        temb_channels=CHANNELS,
        num_layers=1,
        resnet_groups=8,
        num_attention_heads=2,
        cross_attention_dim=8,
        add_downsample=False,
        use_image_cross_attention=True,
        image_cross_attention_dim=8,
        use_sam2=True,
        seg_cross_attention_dim=SAM_DIM,
    ).eval()

def test_segment_tokens_are_mask_averaged_embeddings():
    embeds = torch.randn(1, SAM_DIM, 64, 64)
    tokens, masks = build_segment_conditions(embeds, half_masks(), mask_size=(32, 48))

    assert tokens.shape == (1, SEGMENT_TOKENS_MULTIPLE, SAM_DIM)
    assert masks.shape == (1, SEGMENT_TOKENS_MULTIPLE, 32, 48)

    # Null token: zero vector covering everything
    assert torch.all(tokens[0, 0] == 0)
    assert torch.all(masks[0, 0] == 1)

    torch.testing.assert_close(tokens[0, 1], embeds[0, :, :, :32].mean(dim=(1, 2)), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(tokens[0, 2], embeds[0, :, :, 32:].mean(dim=(1, 2)), atol=1e-4, rtol=1e-4)

    # Masks follow the image extent at the requested resolution
    assert masks[0, 1, :, :20].min() > 0.99 and masks[0, 1, :, 28:].max() < 0.01

    # Padding tokens are empty
    assert torch.all(tokens[0, 3:] == 0) and torch.all(masks[0, 3:] == 0)

def test_all_zero_logits_are_no_mask():
    logits = torch.cat([half_masks(), torch.zeros(1, 1, 256, 256)], dim=1)
    tokens, masks = build_segment_conditions(torch.randn(1, SAM_DIM, 64, 64), logits, mask_size=(16, 16))

    assert torch.all(tokens[0, 3] == 0)
    assert torch.all(masks[0, 3] == 0)

def test_token_count_is_padded_to_multiple():
    logits = torch.randn(1, 9, 256, 256)
    tokens, masks = build_segment_conditions(torch.randn(1, SAM_DIM, 64, 64), logits, mask_size=(8, 8))

    assert tokens.shape[1] == 16 and masks.shape[1] == 16

@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_block_forward_is_invariant_to_mask_order(gradient_checkpointing):
    torch.manual_seed(0)
    block = make_block()
    block.gradient_checkpointing = gradient_checkpointing

    embeds = torch.randn(2, SAM_DIM, 64, 64)
    logits = torch.randn(2, 5, 256, 256) * 5

    def forward(seg_logits):
        tokens, masks = build_segment_conditions(embeds, seg_logits, mask_size=(32, 32))
        hidden_states = torch.randn(2, CHANNELS, 16, 16, generator=torch.Generator().manual_seed(1))
        return block(
            hidden_states,
            temb=torch.zeros(2, CHANNELS),
            encoder_hidden_states=torch.randn(2, 4, 8, generator=torch.Generator().manual_seed(2)),
            image_encoder_hidden_states=torch.randn(2, 4, 8, generator=torch.Generator().manual_seed(3)),
            sam2_encoder_hidden_states=embeds,
            sam2_segmentation_encoder_hidden_states=tokens,
            sam2_segmentation_masks=masks,
        )[0]

    with torch.no_grad():
        reference = forward(logits)
        permuted = forward(logits[:, torch.randperm(5)])

    assert reference.shape == (2, CHANNELS, 16, 16)
    torch.testing.assert_close(reference, permuted, atol=1e-5, rtol=1e-5)

def test_routing_restricts_each_pixel_to_its_segments():
    torch.manual_seed(0)
    attn = make_block().sam2_segmentation_attentions[0]

    embeds = torch.randn(1, SAM_DIM, 64, 64)
    tokens, masks = build_segment_conditions(embeds, half_masks(), mask_size=(16, 16))
    hidden_states = torch.randn(1, CHANNELS, 16, 16)

    changed_tokens = tokens.clone()
    changed_tokens[0, 2] += 10.0  # token of the right half

    with torch.no_grad():
        out = segment_routed_attention(attn, hidden_states, tokens, masks)
        out_changed = segment_routed_attention(attn, hidden_states, changed_tokens, masks)
        out_unrouted = segment_routed_attention(attn, hidden_states, tokens, None)
        out_unrouted_changed = segment_routed_attention(attn, hidden_states, changed_tokens, None)

    diff = (out - out_changed).abs()
    left_diff, right_diff = diff[..., :8].max(), diff[..., 8:].max()

    assert right_diff > 1e-2
    assert left_diff < 1e-3 * right_diff

    # Without routing, the change reaches every pixel
    assert (out_unrouted - out_unrouted_changed).abs()[..., :8].max() > 1e-2

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

def test_get_sam_kwargs_selects_clean_conditions_per_sample():
    lr_logits = torch.cat([half_masks(), half_masks()])                      # 2 masks per image
    gt_logits = torch.full((2, 3, 256, 256), -LOGIT)
    gt_logits[:, :, :128] = LOGIT                                              # 3 masks per image
    batch = {
        "sam_img_embeds": torch.randn(2, SAM_DIM, 64, 64),
        "sam_seg_embeds": lr_logits,
        "sam_img_embeds_gt": torch.randn(2, SAM_DIM, 64, 64),
        "sam_seg_embeds_gt": gt_logits,
    }

    kwargs = get_sam_kwargs(batch, "cpu", torch.float32, True, mask_size=(16, 16), use_clean=torch.tensor([False, True]))

    torch.testing.assert_close(kwargs["sam2_encoder_hidden_states"][0], batch["sam_img_embeds"][0])
    torch.testing.assert_close(kwargs["sam2_encoder_hidden_states"][1], batch["sam_img_embeds_gt"][1])

    masks = kwargs["sam2_segmentation_masks"]
    assert masks.shape == (2, SEGMENT_TOKENS_MULTIPLE, 16, 16)
    assert masks[0, 3].max() < 1e-6        # LR sample: the 3rd mask slot is padding
    assert masks[1, 3, :8].min() > 0.99    # GT sample: 3rd mask covers the top half

    unrouted = get_sam_kwargs(batch, "cpu", torch.float32, True, mask_size=(16, 16), segment_routing=False)
    assert unrouted["sam2_segmentation_masks"] is None
