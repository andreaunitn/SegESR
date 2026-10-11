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
# Multi-GPU (experimental): request several GPUs at submission; the run then uses one process per GPU and
# 8 samples per update (MULTI_GPU_TOTAL_BATCH_SIZE to change it). Each process needs its own CPUs and RAM:
#   sbatch --gpus=2 --cpus-per-task=16 --mem=64G cluster/train.sh configs/train_segesr.yaml --output_dir=preset/train_output/segesr_2gpu
#
# The configs resume from the latest checkpoint of `output_dir`, so if the job hits the time limit of
# the partition, submitting the same command again continues the training.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

in_container bash scripts/run_train.sh "$@"
