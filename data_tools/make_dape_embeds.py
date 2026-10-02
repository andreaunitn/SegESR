import argparse
import glob
import os
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms

# Make the vendored `ram` package importable without installation
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "third_party"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ram.models.ram_lora import ram

img_preproc = transforms.ToTensor()
dape_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

def parse_args():
    parser = argparse.ArgumentParser(description="Precompute the DAPE image embeddings used as 'ram_values' during training.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory of the bicubic-upsampled LR images ('sr_bicubic').")
    parser.add_argument("--embed_dir", type=str, required=True, help="Output directory of the embeddings ('dape_embeds').")
    parser.add_argument("--ram_path", type=str, default="preset/models/ram_swin_large_14m.pth", help="Path to the pretrained RAM model.")
    parser.add_argument("--ram_ft_path", type=str, required=True, help="Path to the DAPE weights.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip images whose embedding already exists.")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.embed_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dape = ram(pretrained=args.ram_path,
               pretrained_condition=args.ram_ft_path,
               image_size=384,
               vit="swin_l")
    dape.eval().to(device)

    image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.png")))
    print(f"Found {len(image_paths)} images in '{args.image_dir}'")

    with torch.no_grad():
        for image_path in tqdm(image_paths, desc="Generating DAPE embeddings"):
            save_path = os.path.join(args.embed_dir, f"{Path(image_path).stem}.pt")
            if args.skip_existing and os.path.isfile(save_path):
                continue

            img = img_preproc(Image.open(image_path).convert("RGB")).unsqueeze(0)

            dape_values = F.interpolate(img, size=(384, 384), mode="bicubic")
            dape_values = dape_normalize(dape_values.clamp(0.0, 1.0)).to(device)

            dape_image_embeddings = dape.generate_image_embeds(dape_values)
            torch.save(dape_image_embeddings.cpu(), save_path)

if __name__ == "__main__":
    main()
