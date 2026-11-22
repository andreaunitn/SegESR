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

# N of Eq. 3: maximum number of masks kept per image, sorted by area (as `data_tools/sam_processing.py --max_seg`).
MAX_MASKS = 150

# H x W of Eq. 3: side of the aggregated mask map (the LR size of the x4 protocol, 128 -> 512).
MASK_MAP_SIZE = 128

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
    """
    bf16 autocast context for SAM 2 inference on GPUs with native bf16 (Ampere or newer).
    Older GPUs (e.g. Turing) and the CPU run SAM 2 in float32.
    """

    device = torch.device(sam_generator.predictor.device)
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8:
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
        img_embeds (torch.Tensor): (1, 1152, 32, 32) output of the last Hiera stage (Eq. 2 of the paper:
            1/32 of the 1024x1024 input of SAM 2).
        seg_logits (torch.Tensor): (N, 1, 256, 256) mask decoder logits, sorted by mask area.
            N is at least 1 (an all-zero map is used when no mask is found).
    """

    image_np = np.array(image.convert("RGB")) if hasattr(image, "convert") else np.asarray(image)

    with sam2_autocast(sam_generator):
        masks = sam_generator.generate(image_np)
        masks = sorted(masks, key=lambda m: m["area"], reverse=True)
        if max_masks is not None:
            masks = masks[:max_masks]

        # `generate` resets the predictor, so the full image is encoded again here; the last Hiera stage is
        # read from the trunk, before the FPN neck
        trunk_outputs = []
        hook = sam_generator.predictor.model.image_encoder.trunk.register_forward_hook(
            lambda module, inputs, outputs: trunk_outputs.append(outputs[-1])
        )
        try:
            sam_generator.predictor.set_image(image_np)
        finally:
            hook.remove()
        img_embeds = trunk_outputs[-1].float().cpu()
        seg_logits = get_mask_logits_from_anns(sam_generator, masks)

    if seg_logits.shape[0] == 0:
        seg_logits = torch.zeros(1, 1, SEG_LOGITS_SIZE, SEG_LOGITS_SIZE)

    return img_embeds, seg_logits

def seg_logits_to_hidden_states(seg_logits):
    """(N, 1, 256, 256) per-mask logits of one image -> (1, N, 256, 256) batch."""

    return seg_logits.squeeze(1).unsqueeze(0)

def pad_seg_logits(seg_logits_list):
    """
    Stacks per-image (N_i, 256, 256) mask logits into a (B, max N_i, 256, 256) batch, padding with all-zero
    maps, which `aggregate_mask_logits` leaves out of the maximum.
    """

    max_masks = max(logits.shape[0] for logits in seg_logits_list)
    batch = seg_logits_list[0].new_zeros(len(seg_logits_list), max_masks, *seg_logits_list[0].shape[1:])
    for i, logits in enumerate(seg_logits_list):
        batch[i, :logits.shape[0]] = logits
    return batch

def aggregate_mask_logits(seg_logits, size=MASK_MAP_SIZE):
    """
    Eq. 3 of the paper: bilinear upsampling of the N mask logits to H x W and pixel-wise maximum.
    All-zero logit maps (batch padding) are left out of the maximum; an image without masks gives all zeros.

    Args:
        seg_logits (torch.Tensor): (B, N, 256, 256) mask decoder logits.

    Returns:
        torch.Tensor: (B, size, size) aggregated soft mask map.
    """

    logits = F.interpolate(seg_logits.float(), size=(size, size), mode="bilinear", align_corners=False)
    valid = seg_logits.flatten(2).abs().amax(dim=-1) > 0
    logits = logits.masked_fill(~valid[..., None, None], float("-inf"))
    aggregated = logits.amax(dim=1)
    return torch.where(torch.isfinite(aggregated), aggregated, torch.zeros_like(aggregated))

def seg_logits_to_tokens(seg_logits):
    """
    SMCA context: the aggregated mask map of Eq. 3, one token per row (H tokens of W values).

    Args:
        seg_logits (torch.Tensor): (B, N, 256, 256) mask decoder logits.

    Returns:
        torch.Tensor: (B, 128, 128) tokens.
    """

    return aggregate_mask_logits(seg_logits).contiguous()

def model_uses_sam2(config):
    """Whether a SegESR UNet/ControlNet with this config is conditioned on SAM 2."""

    return config.use_sam2_image_attention or config.use_sam2_segmentation_attention

def sam2_model_kwargs(config, img_embeds, seg_logits, dtype):
    """
    Builds the SAM 2 inputs of a SegESR UNet/ControlNet from the SAM 2 outputs, with only the entries its
    config uses (an empty dict for a model without SAM 2 attentions).

    Args:
        config: UNet/ControlNet config (`use_sam2_image_attention`, `use_sam2_segmentation_attention`).
        img_embeds (torch.Tensor): (B, 1152, 32, 32) last Hiera stage.
        seg_logits (torch.Tensor): (B, N, 256, 256) mask decoder logits.
    """

    kwargs = {}
    if config.use_sam2_image_attention:
        kwargs["sam2_encoder_hidden_states"] = img_embeds.to(dtype)
    if config.use_sam2_segmentation_attention:
        kwargs["sam2_segmentation_encoder_hidden_states"] = seg_logits_to_tokens(seg_logits).to(dtype)
    return kwargs
