#!/bin/bash
#SBATCH --job-name=r0_alignment
#SBATCH --partition=department_only
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x_%j.log
#
# Step R.0(b) of Track R: alignment of the FiDeSR U-Net features with DINOv2 of the HR, per block, at three
# degradation levels, and its per-image correlation with LPIPS (data_tools/feature_alignment.py). Inference only.
# Needs cluster/setup_trackr.sh first. Writes <OUTPUT_DIR>/alignment.json and per_image.csv.
#
# Usage (from the repository root):
#   sbatch cluster/feature_alignment.sh
#   NUM_IMAGES=50 OUTPUT_DIR=metrics/DIV2K/alignment_smoke sbatch cluster/feature_alignment.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

ROOT="$PWD"
HR_DIR="${HR_DIR:-preset/datasets/test_datasets/DIV2K/test_HR}"
OUTPUT_DIR="${OUTPUT_DIR:-metrics/DIV2K/alignment_fidesr}"

# FiDeSR with the official settings of its README (LF and HF strengths 0.2), empty prompt
in_bibandsr bash -c 'cd "$0" && exec python "$@"' "$ROOT/preset/external/FiDeSR" \
    "$ROOT/data_tools/feature_alignment.py" \
    --hr_dir "$ROOT/$HR_DIR" --output_dir "$ROOT/$OUTPUT_DIR" \
    --dino_path "$ROOT/preset/models/dinov2-base" \
    --pretrained_model_path "$ROOT/preset/models/stable-diffusion-2-1-base" \
    --pretrained_path "$ROOT/preset/models/fidesr/fidesr.pkl" \
    --num_images "${NUM_IMAGES:-500}" --hf_scale 0.2 --lf_scale 0.2
