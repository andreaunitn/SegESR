#!/bin/bash
#SBATCH --job-name=segesr_test
#SBATCH --partition=students
#SBATCH --qos=students_limit
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x_%j.log
#
# Runs inference + metrics on every checkpoint of a training run (scripts/run_test.sh) on the cluster.
#
# Usage (from the repository root):
#   sbatch cluster/test.sbatch [CHECKPOINT_BASE_DIR] [CONFIG]
#   e.g. DATASETS="RealSR DRealSR" sbatch cluster/test.sbatch preset/train_output/segesr_v2
# Metrics are written to metrics/<dataset>/results_<checkpoint>.json.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

in_container bash scripts/run_test.sh "$@"
