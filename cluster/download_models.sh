#!/bin/bash
#SBATCH --job-name=segesr_download_models
#SBATCH --partition=department_only
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/%x_%j.log
#
# Downloads the pretrained models into preset/models (~17 GB): SD 2 base, SeeSR + DAPE, RAM, tiny VAE.
# No GPU is requested; add `#SBATCH --gpus=1` if the partition only accepts GPU jobs.
#
# Usage (from the repository root): sbatch cluster/download_models.sh
# Re-submitting after an interruption continues where it stopped.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

in_container python data_tools/download_data.py --what models "$@"
