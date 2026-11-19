#!/bin/bash
#SBATCH --job-name=band_scale
#SBATCH --partition=department_only
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x_%j.log
#
# Controlled test of how much generated fine detail the metrics reward (data_tools/band_scale.py): the frequencies
# above the LR Nyquist limit of an SR model's outputs are scaled by each ALPHAS value (and by the per-image oracle
# that matches the GT energy), then scored with data_tools/metrics.py and data_tools/fine_structure.py.
#
# Usage (from the repository root):
#   sbatch cluster/band_scale.sh                                     # SeeSR outputs, RealSR and DRealSR
#   NAME=segesr_checkpoint-15000 ALPHAS="0 0.5 1" sbatch cluster/band_scale.sh
# Results: metrics/<dataset>/results_<NAME>_hf<alpha>.json and finestruct_<NAME>_hf<alpha>.json
#          (python3 data_tools/results_table.py <NAME>_hf; ... --prefix finestruct)

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

NAME="${NAME:-seesr}"
read -r -a alphas <<< "${ALPHAS:-0 0.25 0.5 1 1.5 2 oracle}"
read -r -a datasets <<< "${DATASETS:-RealSR DRealSR}"
OUTPUT_DIR="preset/datasets/output"

for dataset in "${datasets[@]}"; do
    sr_dir="$OUTPUT_DIR/$NAME/$dataset"
    gt_dir="preset/datasets/test_datasets/$dataset/test_HR"
    for alpha in "${alphas[@]}"; do
        variant="${NAME}_hf${alpha}"
        out_dir="$OUTPUT_DIR/$variant/$dataset"
        echo "## $variant on $dataset"
        if [ "$alpha" = oracle ]; then
            in_container python -W ignore data_tools/band_scale.py --sr_dir "$sr_dir" --output_dir "$out_dir" --oracle --gt_dir "$gt_dir"
        else
            in_container python -W ignore data_tools/band_scale.py --sr_dir "$sr_dir" --output_dir "$out_dir" --alpha "$alpha"
        fi
        in_container python -W ignore data_tools/metrics.py --sr_dir "$out_dir" --gt_dir "$gt_dir" --dataset "$dataset" --name "$variant"
        in_container python -W ignore data_tools/fine_structure.py --sr_dir "$out_dir" --gt_dir "$gt_dir" --dataset "$dataset" --name "$variant"
    done
done
