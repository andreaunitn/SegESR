#!/bin/bash
#SBATCH --job-name=regression_fusion
#SBATCH --partition=department_only
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x_%j.log
#
# Step T.0: can a model trained for fidelity reconstruct the band below the LR Nyquist limit better than SeeSR?
# For each regression model of data_tools/regression_sr.py (weights downloaded on first use):
#   1. its outputs (<model>), scored like an SR output;
#   2. the band below 1/16 or 1/8 of the SeeSR output taken from it (<NAME>_<model>low16, <NAME>_<model>low8).
# Then data_tools/metrics.py and data_tools/fine_structure.py on every new folder.
#
# Usage (from the repository root):
#   sbatch cluster/regression_fusion.sh
#   NAME=seesr MODELS=realesrnet DATASETS=RealSR sbatch cluster/regression_fusion.sh
# Results: python3 data_tools/results_table.py realesrnet bsrnet; ... --prefix finestruct

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

NAME="${NAME:-seesr}"
read -r -a models <<< "${MODELS:-realesrnet bsrnet}"
read -r -a datasets <<< "${DATASETS:-RealSR DRealSR}"
OUTPUT_DIR="preset/datasets/output"

score() {   # score <variant> <dataset> <gt_dir>
    in_container python -W ignore data_tools/metrics.py --sr_dir "$OUTPUT_DIR/$1/$2" --gt_dir "$3" --dataset "$2" --name "$1"
    in_container python -W ignore data_tools/fine_structure.py --sr_dir "$OUTPUT_DIR/$1/$2" --gt_dir "$3" --dataset "$2" --name "$1"
}

for dataset in "${datasets[@]}"; do
    data_dir="preset/datasets/test_datasets/$dataset"
    for model in "${models[@]}"; do
        echo "## $model on $dataset"
        in_container python -W ignore data_tools/regression_sr.py --model "$model" --lr_dir "$data_dir/test_LR" \
            --output_dir "$OUTPUT_DIR/$model/$dataset"
        score "$model" "$dataset" "$data_dir/test_HR"

        for cutoff in 16:0.0625 8:0.125; do
            variant="${NAME}_${model}low${cutoff%%:*}"
            echo "## $variant on $dataset"
            in_container python -W ignore data_tools/band_scale.py --sr_dir "$OUTPUT_DIR/$NAME/$dataset" \
                --output_dir "$OUTPUT_DIR/$variant/$dataset" --low_from "$OUTPUT_DIR/$model/$dataset/sample00" \
                --cutoff "${cutoff#*:}"
            score "$variant" "$dataset" "$data_dir/test_HR"
        done
    done
done
