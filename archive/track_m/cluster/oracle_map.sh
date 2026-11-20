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
# Step M.0b, with BUDGETS set: the oracle with a realism budget, only at the first tile size, one output per budget
# (the mean lambda_sem over the tiles of the dataset):
#   <NAME>_budget<tile>_b<budget>     the oracle at the price that reaches the budget
#   <NAME>_bshuffle<tile>_b<budget>   the same choices placed at random (control)
# CONTROLS picks which of the two (default both, "oracle shuffle"), so they can run as two jobs.
# Then the budget curve against the default (a negative gap: the curve is above the default):
#   python3 data_tools/curve_gap.py --curve <NAME>_budget<tile>_b<each budget> <NAME>_ls<largest> --points <NAME>
#
# FINE_STRUCTURE=0 skips data_tools/fine_structure.py.
#
# Usage (from the repository root):
#   NAME=freqortho LAMBDA_SEM="0 0.5 1 1.5" sbatch cluster/oracle_map.sh
#   NAME=freqortho LAMBDA_SEM="0 0.5 1 1.5" TILES="64 128" DATASETS=RealSR sbatch cluster/oracle_map.sh
#   NAME=freqortho LAMBDA_SEM="0 0.5 1 1.5" BUDGETS="0.75 1 1.25" CONTROLS=oracle FINE_STRUCTURE=0 \
#       sbatch cluster/oracle_map.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

NAME="${NAME:?set NAME, e.g. freqortho}"
read -r -a lambdas <<< "${LAMBDA_SEM:?set LAMBDA_SEM, e.g. \"0 0.5 1 1.5\"}"
read -r -a tiles <<< "${TILES:-64 128}"
read -r -a datasets <<< "${DATASETS:-RealSR DRealSR DIV2K}"
read -r -a budgets <<< "${BUDGETS:-}"
read -r -a controls <<< "${CONTROLS:-oracle shuffle}"
FINE_STRUCTURE="${FINE_STRUCTURE:-1}"
OUTPUT_DIR="preset/datasets/output"

score() {   # score <name> <dataset>
    local gt_dir="preset/datasets/test_datasets/$2/test_HR"
    in_container python -W ignore data_tools/metrics.py --sr_dir "$OUTPUT_DIR/$1/$2" --gt_dir "$gt_dir" \
        --dataset "$2" --name "$1"
    [ "$FINE_STRUCTURE" = 0 ] && return
    in_container python -W ignore data_tools/fine_structure.py --sr_dir "$OUTPUT_DIR/$1/$2" --gt_dir "$gt_dir" \
        --dataset "$2" --name "$1"
}

for dataset in "${datasets[@]}"; do
    gt_dir="preset/datasets/test_datasets/$dataset/test_HR"
    candidates=()
    for value in "${lambdas[@]}"; do
        candidates+=("$OUTPUT_DIR/${NAME}_ls$value/$dataset")
    done
    if [ "${#budgets[@]}" -gt 0 ]; then
        tile="${tiles[0]}"
        for variant in "${controls[@]}"; do
            prefix="${NAME}_budget${tile}"
            control=()
            if [ "$variant" = shuffle ]; then
                prefix="${NAME}_bshuffle${tile}"
                control=(--control shuffle)
            fi
            echo "## $prefix (budgets ${budgets[*]}) on $dataset"
            in_container python data_tools/oracle_map.py --candidates "${candidates[@]}" --gt_dir "$gt_dir" \
                --values "${lambdas[@]}" --budgets "${budgets[@]}" --output_dir "$OUTPUT_DIR/${prefix}_b{budget}/$dataset" \
                --tile "$tile" "${control[@]}"
            for budget in "${budgets[@]}"; do
                name="${prefix}_b$(printf %g "$budget")"
                grep -h '"price"\|"mean_value"' "$OUTPUT_DIR/$name/$dataset/choice.json"
                score "$name" "$dataset"
            done
        done
        continue
    fi
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
