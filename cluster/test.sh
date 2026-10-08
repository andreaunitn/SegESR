#!/bin/bash
#SBATCH --job-name=segesr_test
#SBATCH --partition=department_only
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x_%j.log
#
# Runs inference + metrics on every checkpoint of a training run (scripts/run_test.sh) on the cluster.
#
# Usage (from the repository root):
#   sbatch cluster/test.sh [CHECKPOINT_BASE_DIR] [CONFIG]
#   e.g. sbatch cluster/test.sh preset/train_output/segesr          # every checkpoint of a run
#        sbatch cluster/test.sh preset/models/seesr                    # the original SeeSR (zero-shot baseline)
#        DATASETS="RealSR DRealSR" sbatch cluster/test.sh preset/train_output/segesr
# Metrics are written to metrics/<dataset>/results_<run>_<checkpoint>.json (results_<model>.json for a single model).

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

in_container bash scripts/run_test.sh "$@"
