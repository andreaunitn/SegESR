#!/bin/bash
#SBATCH --job-name=segesr_train
#SBATCH --partition=department_only
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x_%j.log
#
# Trains SegESR (scripts/run_train.sh) on the cluster.
#
# Usage (from the repository root):
#   sbatch cluster/train.sh [CONFIG] [extra train.py flags...]
#   e.g. sbatch cluster/train.sh configs/train_segesr.yaml
#        sbatch cluster/train.sh configs/train_seesr.yaml --attention_fusion=parallel --output_dir=preset/train_output/pafb
#
# The configs resume from the latest checkpoint of `output_dir`, so if the job hits the time limit of
# the partition, submitting the same command again continues the training.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

in_container bash scripts/run_train.sh "$@"
