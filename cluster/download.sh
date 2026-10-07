#!/bin/bash
#SBATCH --job-name=segesr_download
#SBATCH --partition=department_only
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/%x_%j.log
#
# Downloads the pretrained models and the datasets into preset/ (data_tools/download_data.py).
# No GPU is requested; add `#SBATCH --gpus=1` if the partition only accepts GPU jobs.
#
# LSDIR is gated on Hugging Face: accept its terms at https://huggingface.co/ofsoundof/LSDIR and save a
# read token in ~/.cache/huggingface/token before submitting.
#
# Usage (from the repository root):
#   sbatch cluster/download.sh                         # models, test sets and training subset
#   sbatch cluster/download.sh --what models test      # only some parts
# Re-submitting after an interruption continues where it stopped.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

in_container python data_tools/download_data.py "$@"
