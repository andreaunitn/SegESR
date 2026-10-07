import contextlib

import numpy as np
import torch
import torch.nn.functional as F

SAM2_MODELS = {
    "tiny": "facebook/sam2.1-hiera-tiny",
    "small": "facebook/sam2.1-hiera-small",
    "base_plus": "facebook/sam2.1-hiera-base-plus",
    "large": "facebook/sam2.1-hiera-large",
}

# Shape of the mask decoder low-res logits SegESR uses as segmentation embeddings.
SEG_LOGITS_SIZE = 256

# Maximum number of masks kept per image, sorted by area (as `data_tools/sam_processing.py --max_seg`).
MAX_MASKS = 150

# The segment token axis is padded to a multiple of this (xformers attention bias alignment).
SEGMENT_TOKENS_MULTIPLE = 8

def load_sam2(model_size="large", device=None, **kwargs):
    """
    Loads a SAM 2.1 automatic mask generator from the Hugging Face Hub.
    Its `.predictor.model.image_encoder` is the Hiera encoder used for the SAM 2 perceptual loss.
    Extra kwargs (points_per_side, stability_score_thresh, ...) are forwarded to the generator.
    """

    if model_size not in SAM2_MODELS:
        raise ValueError(f"Invalid SAM 2 model size '{model_size}'. Choose from {list(SAM2_MODELS)}.")

    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    return SAM2AutomaticMaskGenerator.from_pretrained(SAM2_MODELS[model_size], device=str(device), **kwargs)

def sam2_autocast(sam_generator):
    """bf16 autocast context for SAM 2 inference (no-op on CPU)."""

    device = sam_generator.predictor.device
    if torch.device(device).type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()

def get_mask_logits_from_anns(sam_generator, anns):
    """
    Computes per-mask embeddings as the MASK DECODER's raw low-res logits, re-predicting
    each mask from its generation prompt. Requires `predictor.set_image` to have been called.

    Returns:
        torch.Tensor: (N, 1, 256, 256) logits on CPU.
    """

    if not anns:
        return torch.empty(0, 1, SEG_LOGITS_SIZE, SEG_LOGITS_SIZE)

    predictor = sam_generator.predictor
    all_mask_logits = []

    for ann in anns:
        point_coords = np.asarray(ann["point_coords"], dtype=np.float32)
        point_labels = np.ones(len(point_coords), dtype=np.int32)

        _, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

        # Keep the candidate whose predicted IoU is closest to the one of the generated mask
        best_idx = int(np.argmin(np.abs(np.asarray(scores).reshape(-1) - ann["predicted_iou"])))
        all_mask_logits.append(torch.from_numpy(np.asarray(logits[best_idx])).unsqueeze(0))

    return torch.stack(all_mask_logits, dim=0)

@torch.no_grad()
def compute_sam2_conditions(image, sam_generator, max_masks=None):
    """
    Computes the two SAM 2 conditions consumed by the SegESR UNet/ControlNet.

    Args:
        image (PIL.Image.Image | np.ndarray): RGB image.
        sam_generator: generator returned by `load_sam2`.
        max_masks (int, optional): keep only the largest `max_masks` masks.

    Returns:
        img_embeds (torch.Tensor): (1, 256, 64, 64) Hiera image embedding.
        seg_logits (torch.Tensor): (N, 1, 256, 256) mask decoder logits, sorted by mask area.
            N is at least 1 (an all-zero map is used when no mask is found).
    """

    image_np = np.array(image.convert("RGB")) if hasattr(image, "convert") else np.asarray(image)

    with sam2_autocast(sam_generator):
        masks = sam_generator.generate(image_np)
        masks = sorted(masks, key=lambda m: m["area"], reverse=True)
        if max_masks is not None:
            masks = masks[:max_masks]

        # `generate` resets the predictor, so the full image is encoded again here
        sam_generator.predictor.set_image(image_np)
        img_embeds = sam_generator.predictor.get_image_embedding().cpu()
        seg_logits = get_mask_logits_from_anns(sam_generator, masks)

    if seg_logits.shape[0] == 0:
        seg_logits = torch.zeros(1, 1, SEG_LOGITS_SIZE, SEG_LOGITS_SIZE)

    return img_embeds, seg_logits

def seg_logits_to_hidden_states(seg_logits):
    """(N, 1, 256, 256) per-mask logits -> (1, N, 256, 256) batch expected by `build_segment_conditions`."""

    return seg_logits.squeeze(1).unsqueeze(0)

def pad_seg_logits(seg_logits_list):
    """
    Stacks per-image (N_i, 256, 256) mask logits into a (B, max N_i, 256, 256) batch.
    Padding maps are all-zero, which `build_segment_conditions` treats as "no mask".
    """

    max_masks = max(logits.shape[0] for logits in seg_logits_list)
    batch = seg_logits_list[0].new_zeros(len(seg_logits_list), max_masks, *seg_logits_list[0].shape[1:])
    for i, logits in enumerate(seg_logits_list):
        batch[i, :logits.shape[0]] = logits
    return batch

def build_segment_conditions(img_embeds, seg_logits, mask_size, pad_to=SEGMENT_TOKENS_MULTIPLE):
    """
    Turns the SAM 2 outputs into the inputs of the segmentation attention (SMCA): one token per
    segment plus its binary mask, so that each latent pixel attends only to the segments covering it
    (see `segment_routed_attention` in the UNet blocks).

    Each segment token is the Hiera embedding averaged inside the mask (weighted by the mask
    probabilities); the masks are the mask logits > 0, as in SAM 2. Token 0 is an all-zero
    "null" token that covers the whole image: pixels outside every mask attend to it, which
    gives a zero update. All-zero logit maps (empty images, batch padding) are treated as no mask.
    The token axis is zero-padded to a multiple of `pad_to`, as xformers requires for attention biases.

    Args:
        img_embeds (torch.Tensor): (B, C, 64, 64) Hiera image embeddings.
        seg_logits (torch.Tensor): (B, N, 256, 256) mask decoder logits.
        mask_size (tuple): (h, w) resolution of the returned masks, i.e. of the latents.

    Returns:
        tokens (torch.Tensor): (B, K, C) segment tokens, K = 1 + N padded to a multiple of `pad_to`.
        masks (torch.Tensor): (B, K, h, w) binary segment masks.
    """

    embeds = img_embeds.float()
    seg_logits = seg_logits.float()
    batch_size, channels = embeds.shape[:2]

    valid = (seg_logits.flatten(2).abs().amax(dim=-1) > 0).float()[..., None, None]

    # SAM 2 resizes the input to a square, so its embeddings and logits share the image extent
    pool_masks = torch.sigmoid(F.interpolate(seg_logits, size=embeds.shape[-2:], mode="bilinear", align_corners=False)) * valid
    areas = pool_masks.flatten(2).sum(dim=-1, keepdim=True)
    tokens = torch.einsum("bnhw,bchw->bnc", pool_masks, embeds) / areas.clamp_min(1e-6)

    masks = (F.interpolate(seg_logits, size=tuple(mask_size), mode="bilinear", align_corners=False) > 0).float() * valid

    tokens = torch.cat([tokens.new_zeros(batch_size, 1, channels), tokens], dim=1)
    masks = torch.cat([masks.new_ones(batch_size, 1, *masks.shape[-2:]), masks], dim=1)

    num_padding = -tokens.shape[1] % pad_to
    if num_padding:
        tokens = F.pad(tokens, (0, 0, 0, num_padding))
        masks = F.pad(masks, (0, 0, 0, 0, 0, num_padding))

    return tokens, masks
