import argparse
import glob
import os
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms

# Ensure project root and third_party directories are in the Python search path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("third_party"))

from third_party.ram.models.ram import ram
from third_party.ram import inference_ram as inference

def parse_args():
    parser = argparse.ArgumentParser(description="Generate RAM tags for dataset images.")

    parser.add_argument(
        "--root_path",
        type=str,
        default="preset/datasets/train_datasets/training_for_seesr",
        help="Root folder containing the 'gt/' subdirectory.",
    )
    parser.add_argument(
        "--start_gpu",
        type=int,
        default=0,
        help="GPU index to use (e.g., 0, 1, 2) when running parallel shards.",
    )
    parser.add_argument(
        "--all_gpu",
        type=int,
        default=1,
        help="Total number of GPU shards participating.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="preset/models/ram_swin_large_14m.pth",
        help="Path to pretrained RAM model checkpoint.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip images that already have a corresponding tag file.",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    gt_path = os.path.join(args.root_path, "gt")
    tag_path = os.path.join(args.root_path, "tag")
    os.makedirs(tag_path, exist_ok=True)

    # 1. Collect and SORT files so every parallel process sees the exact same index order
    image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG")
    img_list = []

    for ext in image_extensions:
        img_list.extend(glob.glob(os.path.join(gt_path, ext)))

    img_list = sorted(img_list)
    total_imgs = len(img_list)
    print(f"Found {total_imgs} total images in '{gt_path}'")

    if total_imgs == 0:
        print("No images found. Exiting.")
        return

    # 2. Assign the process to the specific requested GPU index
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.start_gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(f"Running shard on device: {device}")

    # 3. Compute slice boundaries for this GPU
    start_num = args.start_gpu * total_imgs // args.all_gpu
    end_num = (args.start_gpu + 1) * total_imgs // args.all_gpu
    current_shard = img_list[start_num:end_num]
    print(f"Processing shard [{start_num}:{end_num}] ({len(current_shard)} images)")

    # 4. Load the RAM model
    print(f"Loading RAM model from '{args.model_path}'...")
    model = ram(
        pretrained=args.model_path,
        image_size=384,
        vit="swin_l",
    )
    model = model.eval().to(device)

    ram_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 5. Tag extraction loop
    with torch.no_grad():
        for img_path in tqdm(current_shard, desc=f"GPU {args.start_gpu}"):
            basename = Path(img_path).stem
            tag_save_path = os.path.join(tag_path, f"{basename}.txt")

            if args.skip_existing and os.path.isfile(tag_save_path):
                continue

            