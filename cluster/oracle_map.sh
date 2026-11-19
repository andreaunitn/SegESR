#!/bin/bash
#SBATCH --job-name=oracle_map
#SBATCH --partition=department_only
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x_%j.log
#
# Step M.0: per-tile oracle of a spatially varying lambda_sem (data_tools/oracle_map.py) from the outputs of
# cluster/onestep_baselines.sh with LAMBDA_SEM set, then scored like any output. For each dataset and tile size:
#   <NAME>_oracle<tile>    in each tile, the lambda_sem output closest to the GT
#   <NAME>_shuffle<tile>   the same choices placed at random (control; only for the first tile size)
# Then: python3 data_tools/curve_gap.py --curve <NAME>_ls<each value> --points <NAME>_oracle<tile> ...
#
# Usage (from the repository root):
#   NAME=freqortho LAMBDA_SEM="0 0.5 1 1.5" sbatch cluster/oracle_map.sh
#   NAME=freqortho LAMBDA_SEM="0 0.5 1 1.5" TILES="64 128" DATASETS=RealSR sbatch cluster/oracle_map.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

NAME="${NAME:?set NAME, e.g. freqortho}"
read -r -a lambdas <<< "${LAMBDA_SEM:?set LAMBDA_SEM, e.g. \"0 0.5 1 1.5\"}"
read -r -a tiles <<< "${TILES:-64 128}"
read -r -a datasets <<< "${DATASETS:-RealSR DRealSR DIV2K}"
OUTPUT_DIR="preset/datasets/output"

score() {   # score <name> <dataset>
    local gt_dir="preset/datasets/test_datasets/$2/test_HR"
    in_container python -W ignore data_tools/metrics.py --sr_dir "$OUTPUT_DIR/$1/$2" --gt_dir "$gt_dir" \
        --dataset "$2" --name "$1"
    in_container python -W ignore data_tools/fine_structure.py --sr_dir "$OUTPUT_DIR/$1/$2" --gt_dir "$gt_dir" \
        --dataset "$2" --name "$1"
}

for dataset in "${datasets[@]}"; do
    gt_dir="preset/datasets/test_datasets/$dataset/test_HR"
    candidates=()
    for value in "${lambdas[@]}"; do
        candidates+=("$OUTPUT_DIR/${NAME}_ls$value/$dataset")
    done
    for i in "${!tiles[@]}"; do
        tile="${tiles[$i]}"
        variants=(oracle)
        [ "$i" -eq 0 ] && variants+=(shuffle)
        for variant in "${variants[@]}"; do
            name="${NAME}_${variant}${tile}"
            control=()
            [ "$variant" = shuffle ] && control=(--control shuffle)
            echo "## $name on $dataset"
            in_container python data_tools/oracle_map.py --candidates "${candidates[@]}" --gt_dir "$gt_dir" \
                --output_dir "$OUTPUT_DIR/$name/$dataset" --tile "$tile" "${control[@]}"
            score "$name" "$dataset"
        done
    done
done
