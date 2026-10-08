"""
Downloads the pretrained models and the datasets used by SegESR into `preset/`:

    models   preset/models/stable-diffusion-2-base   SD 2 base, diffusers format (mirror linked by SeeSR)
             preset/models/seesr  + DAPE.pth         SeeSR baseline and DAPE (official SeeSR Hugging Face repo)
             preset/models/ram_swin_large_14m.pth    RAM
             preset/models/tiny_vae                  TAESD, used by the perceptual losses
    test     preset/datasets/test_datasets/{DIV2K,RealSR,DRealSR}/{test_LR,test_HR}   StableSR test sets
    train    preset/datasets/train_datasets/LSDIR/finetune_subset   15% of LSDIR + first 1.5K FFHQ images

LSDIR is gated on Hugging Face: accept its terms at https://huggingface.co/ofsoundof/LSDIR and provide a
read token (`HF_TOKEN` environment variable or ~/.cache/huggingface/token). RealLR200 is only distributed
on Google Drive and must be downloaded by hand (see the message printed at the end).

Every step skips what is already in place, so the script can be re-run after an interruption.

Usage: python data_tools/download_data.py [--what models test train] [--root preset]
"""

import argparse
import random
import shutil
import tarfile
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm

SD2_REPO = "Manojb/stable-diffusion-2-base"
SD2_FILES = [
    "model_index.json",
    "feature_extractor/*",
    "scheduler/*",
    "tokenizer/*",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]
SEESR_REPO = "CSWRY/SeeSR"
RAM_REPO = "xinyu1205/recognize_anything_model"
TINY_VAE_REPO = "madebyollin/taesd"

STABLESR_TEST_REPO = "Iceclear/StableSR-TestSets"
STABLESR_TEST_ZIP = "StableSR_testsets.zip"
# StableSR folder -> (SegESR dataset name, LR subfolder, HR subfolder)
STABLESR_TEST_SETS = {
    "StableSR_testsets/DIV2K_V2_val/": ("DIV2K", "lq", "gt"),
    "StableSR_testsets/RealSRVal_crop128/": ("RealSR", "test_LR", "test_HR"),
    "StableSR_testsets/DrealSRVal_crop128/": ("DRealSR", "test_LR", "test_HR"),
}

LSDIR_REPO = "ofsoundof/LSDIR"
LSDIR_SHARDS = [f"shard-{i:02d}.tar.gz" for i in range(17)]
FFHQ_REPO = "marcosv/ffhq-dataset"

REALLR200_URL = "https://drive.google.com/drive/folders/1L2VsQYQRKhWJxe6yWZU9FgBWSgBCk6mz"

def download_models(root):
    models = root / "models"

    print(f"--- SD 2 base ({SD2_REPO})")
    snapshot_download(SD2_REPO, allow_patterns=SD2_FILES, local_dir=models / "stable-diffusion-2-base")

    print(f"--- SeeSR + DAPE ({SEESR_REPO})")
    snapshot_download(SEESR_REPO, allow_patterns=["seesr/*", "DAPE.pth"], local_dir=models)

    print(f"--- RAM ({RAM_REPO})")
    hf_hub_download(RAM_REPO, "ram_swin_large_14m.pth", local_dir=models)

    print(f"--- Tiny VAE ({TINY_VAE_REPO})")
    snapshot_download(TINY_VAE_REPO, allow_patterns=["config.json", "diffusion_pytorch_model.safetensors"], local_dir=models / "tiny_vae")

def download_test_sets(root):
    test_root = root / "datasets" / "test_datasets"
    if all((test_root / name / "test_HR").is_dir() for name, _, _ in STABLESR_TEST_SETS.values()):
        print("--- StableSR test sets already in place")
        return

    print(f"--- StableSR test sets ({STABLESR_TEST_REPO})")
    download_dir = root / "downloads"
    zip_path = hf_hub_download(STABLESR_TEST_REPO, STABLESR_TEST_ZIP, repo_type="dataset", local_dir=download_dir)

    with zipfile.ZipFile(zip_path) as zf:
        for member in tqdm(zf.infolist(), desc="Extracting test sets"):
            for prefix, (name, lr_dir, hr_dir) in STABLESR_TEST_SETS.items():
                if member.is_dir() or not member.filename.startswith(prefix):
                    continue
                sub_dir, file_name = member.filename[len(prefix):].split("/", 1)
                target_dir = {lr_dir: "test_LR", hr_dir: "test_HR"}.get(sub_dir)
                if target_dir is None or "/" in file_name:
                    continue
                target = test_root / name / target_dir / file_name
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    Path(zip_path).unlink()
    for name, _, _ in STABLESR_TEST_SETS.values():
        for split in ("test_LR", "test_HR"):
            print(f"    {name}/{split}: {len(list((test_root / name / split).glob('*.png')))} images")

def download_lsdir(root, full_dir):
    """Downloads and extracts the 17 LSDIR HR shards (~155 GB), one at a time."""

    download_dir = root / "downloads"
    for shard in LSDIR_SHARDS:
        done_marker = full_dir / f".{shard}.done"
        if done_marker.exists():
            continue

        print(f"--- LSDIR {shard}")
        tar_path = hf_hub_download(LSDIR_REPO, shard, local_dir=download_dir)
        # The "data" filter refuses unsafe archive members (absolute paths, links outside the folder)
        extract_kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        with tarfile.open(tar_path) as tar:
            tar.extractall(full_dir, **extract_kwargs)
        Path(tar_path).unlink()
        done_marker.touch()

def build_train_subset(root, lsdir_fraction, ffhq_count, seed, delete_full_lsdir):
    subset_dir = root / "datasets" / "train_datasets" / "LSDIR" / "finetune_subset"
    done_marker = subset_dir / ".done"
    if done_marker.exists():
        print(f"--- Training subset already in place: {subset_dir}")
        return
    subset_dir.mkdir(parents=True, exist_ok=True)

    full_dir = root / "datasets" / "LSDIR_full"
    download_lsdir(root, full_dir)

    # Same selection as the original subset (random.sample with seed 42), on a sorted, reproducible list
    lsdir_images = sorted(full_dir.rglob("*.png"))
    selected = random.Random(seed).sample(lsdir_images, int(len(lsdir_images) * lsdir_fraction))
    for path in tqdm(selected, desc=f"Copying {lsdir_fraction:.0%} of LSDIR ({len(lsdir_images)} images)"):
        shutil.copy(path, subset_dir / path.name)

    print(f"--- First {ffhq_count} FFHQ images ({FFHQ_REPO})")
    ffhq_dir = root / "downloads" / "ffhq"
    ffhq_files = [f"Part1/{i:05d}.png" for i in range(ffhq_count)]
    snapshot_download(FFHQ_REPO, repo_type="dataset", allow_patterns=ffhq_files, local_dir=ffhq_dir)
    for file_name in tqdm(ffhq_files, desc="Copying FFHQ"):
        shutil.copy(ffhq_dir / file_name, subset_dir / Path(file_name).name)
    shutil.rmtree(ffhq_dir)

    if delete_full_lsdir:
        shutil.rmtree(full_dir)

    done_marker.touch()
    print(f"    {len(list(subset_dir.glob('*.png')))} training images in {subset_dir}")

def main():
    parser = argparse.ArgumentParser(description="Download the SegESR models and datasets into `preset/`.")
    parser.add_argument("--root", type=str, default="preset")
    parser.add_argument("--what", nargs="+", default=["models", "test", "train"], choices=["models", "test", "train"])
    parser.add_argument("--lsdir_fraction", type=float, default=0.15, help="Fraction of LSDIR randomly sampled for training.")
    parser.add_argument("--ffhq_count", type=int, default=1500,
                        help="Number of FFHQ images (the first ones) added for training. The default keeps the face share of SeeSR (10K FFHQ for 85K LSDIR).")
    parser.add_argument("--seed", type=int, default=42, help="Seed of the LSDIR sampling.")
    parser.add_argument("--delete_full_lsdir", action="store_true", help="Delete the full extracted LSDIR (~155 GB) once the subset is built.")
    args = parser.parse_args()

    root = Path(args.root)

    if "models" in args.what:
        download_models(root)
    if "test" in args.what:
        download_test_sets(root)
    if "train" in args.what:
        build_train_subset(root, args.lsdir_fraction, args.ffhq_count, args.seed, args.delete_full_lsdir)

    reallr200_dir = root / "datasets" / "test_datasets" / "RealLR200" / "test_LR"
    if "test" in args.what and not reallr200_dir.is_dir():
        print(
            f"\nRealLR200 must be downloaded by hand: open {REALLR200_URL} in a browser, download the 'RealLR200'"
            f" folder and put its images in '{reallr200_dir}' (it has no HR images)."
        )

    print("Done.")

if __name__ == "__main__":
    main()
