import argparse
import glob
import os
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import numpy as np
import torch

# Make `segesr` and the vendored packages importable without installation
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "third_party"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from segesr.utils.sam_utils import compute_sam2_conditions, load_sam2, sam2_autocast

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

def process_dataset(args):
    """
    Precomputes the SAM 2 data used by SegESR for all images inside `--image_dir`:

    - `--embed_dir`: Hiera image embeddings, (1, 256, 64, 64) per image  -> training folder 'sam_embeds/'
    - `--logit_dir`: mask decoder logits, (N, 1, 256, 256) per image   -> training folder 'seg_embeds/'
    - `--mask_dir`:  binary masks, (N, H, W) per image                  -> training folder 'gt_seg/' (optional)

    Embeddings and logits are computed on the bicubic-upsampled LR images ('sr_bicubic/'),
    binary masks on the GT images ('gt/').
    """

    output_dirs = [args.embed_dir, args.logit_dir, args.mask_dir]
    if not any(output_dirs):
        raise ValueError("Specify at least one output directory (--embed_dir, --logit_dir or --mask_dir).")

    for path in output_dirs:
        if path:
            os.makedirs(path, exist_ok=True)

    image_paths = sorted(
        p for p in glob.glob(os.path.join(args.image_dir, "*")) if p.lower().endswith(IMAGE_EXTENSIONS)
    )

    if args.skip_existing:
        def is_done(stem):
            return all(os.path.isfile(os.path.join(d, f"{stem}.pt")) for d in output_dirs if d)
        image_paths = [p for p in image_paths if not is_done(Path(p).stem)]

    sam_generator = load_sam2(
        model_size=args.model_size,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        stability_score_thresh=args.stability_score_thresh,
    )

    print(f"Processing {len(image_paths)} images from '{args.image_dir}' with SAM 2.1 ({args.model_size})...")

    for img_path in tqdm(image_paths, desc="Generating SAM 2 data"):
        stem = Path(img_path).stem
        image = Image.open(img_path).convert("RGB")

        if args.embed_dir or args.logit_dir:
            img_embeds, seg_logits = compute_sam2_conditions(image, sam_generator, max_masks=args.max_seg)

            if args.embed_dir:
                torch.save(img_embeds, os.path.join(args.embed_dir, f"{stem}.pt"))
            if args.logit_dir:
                torch.save(seg_logits, os.path.join(args.logit_dir, f"{stem}.pt"))

        if args.mask_dir:
            image_np = np.array(image)
            with torch.no_grad(), sam2_autocast(sam_generator):
                masks = sam_generator.generate(image_np)
            masks = sorted(masks, key=lambda m: m["area"], reverse=True)[:args.max_seg]

            if masks:
                binary_masks = torch.stack([torch.from_numpy(m["segmentation"]) for m in masks])
            else:
                binary_masks = torch.empty(0, *image_np.shape[:2], dtype=torch.bool)
            torch.save(binary_masks, os.path.join(args.mask_dir, f"{stem}.pt"))

    print("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract SAM 2 embeddings, mask logits and masks for SegESR.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing the RGB images.")
    parser.add_argument("--embed_dir", type=str, default=None, help="Output directory of the image embeddings ('sam_embeds').")
    parser.add_argument("--logit_dir", type=str, default=None, help="Output directory of the mask decoder logits ('seg_embeds').")
    parser.add_argument("--mask_dir", type=str, default=None, help="Output directory of the binary masks ('gt_seg').")
    parser.add_argument("--model_size", type=str, default="large", choices=["tiny", "small", "base_plus", "large"])
    parser.add_argument("--max_seg", type=int, default=150, help="Maximum number of masks kept per image, sorted by area.")
    parser.add_argument("--points_per_side", type=int, default=16, help="Points per side for mask generation grid.")
    parser.add_argument("--points_per_batch", type=int, default=128, help="Points processed in a batch for mask generation.")
    parser.add_argument("--stability_score_thresh", type=float, default=0.9, help="Stability score threshold for filtering masks.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip images whose outputs already exist.")

    process_dataset(parser.parse_args())
