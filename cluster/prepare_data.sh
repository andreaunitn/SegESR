#!/bin/bash
#SBATCH --job-name=segesr_data
#SBATCH --partition=department_only
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x_%j.log
#
# Builds the SegESR training folder (scripts/prepare_data.sh) on the cluster.
#
# Usage (from the repository root):
#   sbatch cluster/prepare_data.sh GT_DIR [SAVE_DIR]
# Environment variables of scripts/prepare_data.sh (RAM_FT_PATH, EPOCHS, WITH_GT_SAM, WITH_GT_SEG) are passed through.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

in_container bash scripts/prepare_data.sh "$@"
