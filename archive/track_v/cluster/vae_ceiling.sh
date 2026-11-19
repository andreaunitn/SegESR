#!/bin/bash
#SBATCH --job-name=vae_ceiling
#SBATCH --partition=department_only
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x_%j.log
#
# VAE ceiling of latent SR models: reconstructs the GT images of each dataset with the SD 2 VAE
# (data_tools/vae_ceiling.py), then scores the reconstructions like an SR output (data_tools/metrics.py).
#
# Usage (from the repository root):
#   sbatch cluster/vae_ceiling.sh                                  # encoder + decoder, RealSR and DRealSR
#   MODE=optimize sbatch cluster/vae_ceiling.sh                    # best latent for the decoder (slower)
#   DATASETS="RealSR" MODE=optimize STEPS=300 sbatch cluster/vae_ceiling.sh
# Metrics: metrics/<dataset>/results_vae_sd2_<mode>.json (see: python3 data_tools/results_table.py vae_)
# Re-running the same command resumes: images already reconstructed are skipped.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

read -r -a datasets <<< "${DATASETS:-RealSR DRealSR}"
MODE="${MODE:-encode}"
STEPS="${STEPS:-200}"
NAME="vae_sd2_${MODE}"

for dataset in "${datasets[@]}"; do
    gt_dir="preset/datasets/test_datasets/${dataset}/test_HR"
    out_dir="preset/datasets/output/${NAME}/${dataset}"
    echo "## ${NAME}: ${dataset}"
    in_container python -W ignore data_tools/vae_ceiling.py --gt_dir "$gt_dir" --output_dir "$out_dir" \
        --mode "$MODE" --steps "$STEPS" --skip_existing
    in_container python -W ignore data_tools/metrics.py --sr_dir "$out_dir" --gt_dir "$gt_dir" \
        --dataset "$dataset" --name "$NAME"
done
