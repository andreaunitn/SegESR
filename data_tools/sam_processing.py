import argparse
import glob
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

import sam2.build_sam as build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

SAM_CONFIG = {
    "tiny": {
        "config": "sam2_hiera_t.yaml",
        "checkpoint": "preset/models/sam2_hiera_tiny.pt",
    },
    "small": {
        "config": "sam2_hiera_s.yaml",
        "checkpoint": "preset/models/sam2_hiera_small.pt",
    },
    "base_plus": {
        "config": "sam2_hiera_b+.yaml",
        "checkpoint": "preset/models/sam2_hiera_base_plus.pt",
    },
    "large": {
        "config": "sam2_hiera_l.yaml",
        "checkpoint": "preset/models/sam2_hiera_large.pt",
    },
}

def load(apply_postprocessing=False, stability_score_thresh=0.9, model_size="large", checkpoint_path=None, config_file=None, device=None, **kwargs):
    """
    Initializes SAM 2 and returns an automatic mask generator whose
    .predictor.model.image_encoder provides the vision embeddings used by SegESR.

    Args:
        apply_postprocessing (bool): Whether to apply postprocessing to masks.
        stability_score_thresh (float): Threshold for mask stability filtering.
        model_size (str): One of ['tiny', 'small', 'base_plus', 'large'].
        checkpoint_path (str, optional): Custom path to .pt weights file.
        config_file (str, optional): Custom config YAML filename.
        device (str, optional): Computation device ('cuda' or 'cpu').
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_entry = SAM_CONFIG.get(model_size, SAM_CONFIG["large"])
    resolved_config = config_file or cfg_entry["config"]
    resolved_ckpt = checkpoint_path or cfg_entry["checkpoint"]

    if not os.path.isfile(resolved_ckpt):
        # Fallback to local filename check
        alt_ckpt = os.path.join("preset", "models", os.path.basename(resolved_ckpt))
        if os.path.isfile(alt_ckpt):
            resolved_ckpt = alt_ckpt

    sam_model = build_sam2(
        config_file=resolved_config,
        ckpt_path=resolved_ckpt if os.path.isfile(resolved_ckpt) else None,
        device=device,
        **kwargs,
    )

    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam_model,
        apply_postprocessing=apply_postprocessing,
        stability_score_thresh=stability_score_thresh,
        output_mode="binary_mask",
    )

    return mask_generator

def extract_features_and_masks(mask_generator, image_rgb, device="cuda"):
    """
    Extracts:
    1. sam_img_embeds: Vision encoder embeddings from the Hiera image encoder.
    2. sam_seg_embeds: Multi-region layout mask tensor.
    """

    image_np = np.array(image_rgb)
    orig_h, orig_w = image_np.shape[:2]

    # 1. Extract image features via the internal predictor
    predictor = mask_generator.predictor
    predictor.set_image(image_np)

    # Fetch high-level vision features from predictor state
    if hasattr(predictor, "_features") and predictor._features is not None:
        img_embeds = predictor._features["image_embeds"].cpu()
    else:
        from torchvision import transforms
        norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        t_img = norm(image_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            img_embeds = predictor.model.image_encoder(t_img)["vision_features"].cpu()

    # 2. Extract and compile masks
    masks = mask_generator.generate(image_np)
    if len(masks) == 0:
        # Fallback empty mask representation (1, 64, 64)
        seg_embeds = torch.zeros((1, 64, 64), dtype=torch.float32)
    else:
        # Sort masks by area descending
        sorted_masks = sorted(masks, key=lambda m: m["area"], reverse=True)
        binary_masks = [torch.from_numpy(m["segmentation"]).float() for m in sorted_masks[:16]]
        stacked_masks = torch.stack(binary_masks, dim=0).unsqueeze(0) # (1, N, H, W)

        # Downsample to latent spatial resolution (64x64)
        seg_embeds = F.interpolate(stacked_masks, size=(64,64), mode="nearest").squeeze(0)

    return img_embeds, seg_embeds

def process_dataset(input_dir, save_dir, model_size="large", extension="png"):
    """
    Extracts and stores SAM 2 image embeddings and segmentation masks
    for all images inside input_dir.
    """

    os.makedirs(os.path.join(save_dir, "sam_img_embeds"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "sam_seg_embeds"), exist_ok=True)

    mask_generator = load(model_size=model_size, apply_postprocessing=False, stability_score_thresh=0.9)
    image_paths = sorted(glob.glob(os.path.join(input_dir, f"*.{extension}")))

    print(f"Processing {len(image_paths)} images from '{input_dir}' with SAM 2 ({model_size})...")

    for img_path in tqdm(image_paths, desc="Generating SAM 2 Embeddings"):
        stem = Path(img_path).stem
        img = Image.open(img_path).convert("RGB")

        img_embeds, seg_embeds = extract_features_and_masks(mask_generator, img)
        torch.save(img_embeds, os.path.join(save_dir, "sam_img_embeds", f"{stem}.pt"))
        torch.save(seg_embeds, os.path.join(save_dir, "sam_seg_embeds", f"{stem}.pt"))

    print(f"Finished processing! Embeddings saved to '{save_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract SAM 2 embeddings and segmentations for SegESR")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing RGB images")
    parser.add_argument("--save_dir", type=str, required=True, help="Output directory to save embeddings")
    parser.add_argument("--model_size", type=str, default="large", choices=["tiny", "small", "base_plus", "large"])
    parser.add_argument("--extention", type=str, default="png", help="Image file extension")

    args = parser.parse_args()
    process_dataset(args.input_dir, args.save_dir, model_size=args.model_size, extension=args.extention)