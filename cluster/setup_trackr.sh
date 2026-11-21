#!/bin/bash
#SBATCH --job-name=trackr_setup
#SBATCH --partition=department_only
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x_%j.log
#
# One-time setup of Track R, in the environment of cluster/setup_bibandsr.sh (run that first):
#   preset/external/FiDeSR             FiDeSR code at a pinned commit (GitHub archive, Apache-2.0)
#   preset/models/fidesr/fidesr.pkl    FiDeSR weights (Hugging Face jmjin2/FiDeSR, Apache-2.0)
#   preset/models/dinov2-base          DINOv2 ViT-B/14 (Hugging Face facebook/dinov2-base, Apache-2.0), safetensors only
# SD 2.1-base and RAM (preset/models/ram_swin_large_14m.pth) are already there.
#
# Usage (from the repository root): sbatch cluster/setup_trackr.sh
# Safe to re-run: what already exists is reused.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

FIDESR_COMMIT=8042f139fc3f6f91ac450c41f69a14c45f16ac4b

echo "--- 1/3: FiDeSR code (preset/external/FiDeSR)"
dir=preset/external/FiDeSR
if [ ! -f "$dir/.commit" ] || [ "$(cat "$dir/.commit")" != "$FIDESR_COMMIT" ]; then
    rm -rf "$dir" && mkdir -p "$dir"
    curl -sSfL "https://github.com/Ar0Kim/FiDeSR/archive/$FIDESR_COMMIT.tar.gz" | tar -xz -C "$dir" --strip-components=1
    echo "$FIDESR_COMMIT" > "$dir/.commit"
fi
echo "FiDeSR at $(cat "$dir/.commit")"

echo "--- 2/3: Weights (preset/models)"
in_bibandsr python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
hf_hub_download("jmjin2/FiDeSR", "fidesr.pkl", local_dir="preset/models/fidesr")
snapshot_download("facebook/dinov2-base", local_dir="preset/models/dinov2-base",
                  allow_patterns=["config.json", "model.safetensors", "preprocessor_config.json"])
PY
ls -l preset/models/fidesr preset/models/dinov2-base preset/models/ram_swin_large_14m.pth

echo "--- 3/3: Checks and tests"
in_bibandsr python - <<'PY'
import torch
from transformers import Dinov2Model
# FiDeSR loads its weights with torch.load and the default of torch 2.6 (weights_only=True)
sd = torch.load("preset/models/fidesr/fidesr.pkl", weights_only=True)
print(f"fidesr.pkl: {len(sd)} entries, loaded with weights_only=True, LoRA rank {sd.get('lora_rank_unet')}")
dino = Dinov2Model.from_pretrained("preset/models/dinov2-base")
print(f"dinov2-base: {sum(p.numel() for p in dino.parameters()) / 1e6:.1f} M parameters, patch {dino.config.patch_size}")
PY
in_bibandsr python -m pytest -q -p no:cacheprovider tests/test_feature_alignment.py

echo "Setup complete."
