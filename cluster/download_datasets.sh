#!/bin/bash
#SBATCH --job-name=segesr_download_datasets
#SBATCH --partition=department_only
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/%x_%j.log
#
# Downloads the datasets into preset/datasets: the StableSR test sets (DIV2K, RealSR, DRealSR) and the
# training subset (15% of LSDIR + first 1.5K FFHQ images). The full LSDIR (~155 GB) is downloaded first.
# No GPU is requested; add `#SBATCH --gpus=1` if the partition only accepts GPU jobs.
#
# LSDIR is gated on Hugging Face: accept its terms at https://huggingface.co/ofsoundof/LSDIR and save a
# read token in ~/.cache/huggingface/token before submitting.
#
# Usage (from the repository root):
#   sbatch cluster/download_datasets.sh                    # test sets and training subset
#   sbatch cluster/download_datasets.sh --what test        # only the test sets
#   sbatch cluster/download_datasets.sh --delete_full_lsdir  # delete the full LSDIR once the subset is built
# Re-submitting after an interruption continues where it stopped.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

# A `--what` given on the command line comes last, so it replaces this default
in_container python data_tools/download_data.py --what test train "$@"
