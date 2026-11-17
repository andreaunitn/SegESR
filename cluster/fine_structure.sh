#!/bin/bash
#SBATCH --job-name=fine_structure
#SBATCH --partition=department_only
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x_%j.log
#
# Fine-structure metrics (data_tools/fine_structure.py: error per frequency band, edges) of every SR output folder
# in preset/datasets/output/ that has the dataset, against its GT. CPU only, no GPU needed.
#
# Usage (from the repository root):
#   sbatch cluster/fine_structure.sh                                  # every output, RealSR and DRealSR
#   NAMES="seesr vae_sd2_encode" DATASETS=RealSR sbatch cluster/fine_structure.sh
# Results: metrics/<dataset>/finestruct_<name>.json (see: python3 data_tools/results_table.py --prefix finestruct)

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source cluster/env.sh

read -r -a datasets <<< "${DATASETS:-RealSR DRealSR}"
OUTPUT_DIR="preset/datasets/output"
if [ -n "${NAMES:-}" ]; then
    read -r -a names <<< "$NAMES"
else
    mapfile -t names < <(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)
fi

for name in "${names[@]}"; do
    for dataset in "${datasets[@]}"; do
        sr_dir="$OUTPUT_DIR/$name/$dataset"
        gt_dir="preset/datasets/test_datasets/$dataset/test_HR"
        if [ ! -d "$sr_dir/sample00" ]; then
            echo "skip $name/$dataset: no outputs"
            continue
        fi
        in_container python -W ignore data_tools/fine_structure.py --sr_dir "$sr_dir" --gt_dir "$gt_dir" \
            --dataset "$dataset" --name "$name"
    done
done
