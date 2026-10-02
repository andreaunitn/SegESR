import contextlib

import numpy as np
import torch

SAM2_MODELS = {
    "tiny": "facebook/sam2.1-hiera-tiny",
    "small": "facebook/sam2.1-hiera-small",
    "base_plus": "facebook/sam2.1-hiera-base-plus",
    "large": "facebook/sam2.1-hiera-large",
}

# Shape of the mask decoder low-res logits SegESR uses as segmentation embeddings.
SEG_LOGITS_SIZE = 256

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
    """(N, 1, 256, 256) per-mask logits -> (1, N, 256, 256) batch expected by the models."""

    return seg_logits.squeeze(1).unsqueeze(0)
