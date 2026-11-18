#!/bin/bash
#SBATCH --job-name=onestep_baselines
#SBATCH --partition=department_only
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x_%j.log
#
# Step O.0: the one-step baselines PiSA-SR and OSEDiff with their official weights and test scripts (default
# settings), scored with our protocol. Needs cluster/setup_bibandsr.sh first.
#   1. the official test script (BiBandSR environment) writes to preset/datasets/external_raw/<model>/<dataset>;
#   2. data_tools/collect_sr.py brings the outputs to preset/datasets/output/<model>/<dataset>/sample00, at 4x the
#      LR size (the scripts round it down to a multiple of 8);
#   3. data_tools/metrics.py and data_tools/fine_structure.py.
# A model and dataset whose outputs are all collected already is not generated again.
#
# Usage (from the repository root):
#   sbatch cluster/onestep_baselines.sh
#   MODELS=pisasr DATASETS=RealSR sbatch cluster/onestep_baselines.sh
# Results: python3 data_tools/results_table.py pisasr osediff; ... --prefix finestruct

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

read -r -a models <<< "${MODELS:-pisasr osediff}"
read -r -a datasets <<< "${DATASETS:-RealSR DRealSR DIV2K}"
ROOT="$PWD"
MODELS_DIR="$ROOT/preset/models"
OUTPUT_DIR="preset/datasets/output"
RAW_DIR="preset/datasets/external_raw"

generate() {   # generate <model> <LR dir> <raw output dir>, all paths absolute
    case "$1" in
        pisasr)
            in_bibandsr bash -c 'cd "$0" && exec python test_pisasr.py "$@"' "$ROOT/preset/external/PiSA-SR" \
                --pretrained_model_path "$MODELS_DIR/stable-diffusion-2-1-base" \
                --pretrained_path "$MODELS_DIR/pisasr/pisa_sr.pkl" \
                --process_size 512 --upscale 4 --default \
                --input_image "$2" --output_dir "$3" ;;
        osediff)
            in_bibandsr bash -c 'cd "$0" && exec python test_osediff.py "$@"' "$ROOT/preset/external/OSEDiff" \
                --pretrained_model_name_or_path "$MODELS_DIR/stable-diffusion-2-1-base" \
                --osediff_path "$MODELS_DIR/osediff/osediff.pkl" \
                --ram_path "$MODELS_DIR/ram_swin_large_14m.pth" --ram_ft_path "$MODELS_DIR/DAPE.pth" \
                --process_size 512 --upscale 4 \
                -i "$2" -o "$3" ;;
        *) echo "Unknown model: $1"; exit 1 ;;
    esac
}

for dataset in "${datasets[@]}"; do
    data_dir="preset/datasets/test_datasets/$dataset"
    for model in "${models[@]}"; do
        out_dir="$OUTPUT_DIR/$model/$dataset"
        n_lr=$(find "$data_dir/test_LR" -maxdepth 1 -type f | wc -l)
        n_out=$(find "$out_dir/sample00" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
        if [ "$n_out" -eq "$n_lr" ]; then
            echo "## $model on $dataset: outputs already collected"
        else
            echo "## $model on $dataset: generating"
            mkdir -p "$RAW_DIR/$model/$dataset"
            generate "$model" "$ROOT/$data_dir/test_LR" "$ROOT/$RAW_DIR/$model/$dataset"
            in_container python data_tools/collect_sr.py --src_dir "$RAW_DIR/$model/$dataset" \
                --lr_dir "$data_dir/test_LR" --output_dir "$out_dir"
        fi
        in_container python -W ignore data_tools/metrics.py --sr_dir "$out_dir" --gt_dir "$data_dir/test_HR" \
            --dataset "$dataset" --name "$model"
        in_container python -W ignore data_tools/fine_structure.py --sr_dir "$out_dir" --gt_dir "$data_dir/test_HR" \
            --dataset "$dataset" --name "$model"
    done
done
