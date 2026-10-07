#!/bin/bash
# Run SegESR inference + metrics on every checkpoint of a training run and every test dataset.
#
# Usage: ./scripts/run_test.sh [CHECKPOINT_BASE_DIR] [CONFIG]
#   CHECKPOINT_BASE_DIR  folder containing 'checkpoint-*' subfolders (default: preset/train_output/segesr)
#   CONFIG               test config (default: configs/test_default.yaml)
#
# Environment variables:
#   DATASETS          space-separated dataset names (default: "DIV2K DRealSR RealLR200 RealSR")
#   SKIP_GENERATION   "true" to only (re)compute metrics on existing outputs (default: false)
#   CUDA_VISIBLE_DEVICES  GPU to use (default: 0)

set -u

cd "$(dirname "$0")/.." || exit 1

CHECKPOINT_BASE_DIR="${1:-preset/train_output/segesr}"
CONFIG="${2:-configs/test_default.yaml}"
read -r -a datasets <<< "${DATASETS:-DIV2K DRealSR RealLR200 RealSR}"
SKIP_GENERATION="${SKIP_GENERATION:-false}"

BASE_TEST_DATA_DIR="preset/datasets/test_datasets"
BASE_OUTPUT_DIR="preset/datasets/output"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config file '$CONFIG' not found."
    exit 1
fi

mapfile -t checkpoints < <(find "$CHECKPOINT_BASE_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V)
if [ ${#checkpoints[@]} -eq 0 ]; then
    echo "ERROR: no 'checkpoint-*' folder found in '$CHECKPOINT_BASE_DIR'."
    exit 1
fi

mkdir -p "$BASE_OUTPUT_DIR"

for checkpoint_path in "${checkpoints[@]}"; do
    MODEL_NAME=$(basename "$checkpoint_path")

    echo "======================================================================"
    echo "## PROCESSING CHECKPOINT: $MODEL_NAME"
    echo "======================================================================"

    for dataset_name in "${datasets[@]}"; do
        echo "######################################################################"
        echo "## PROCESSING DATASET: $dataset_name for CHECKPOINT: $MODEL_NAME"
        echo "######################################################################"

        LR_IMAGE_PATH="${BASE_TEST_DATA_DIR}/${dataset_name}/test_LR"
        GT_PATH="${BASE_TEST_DATA_DIR}/${dataset_name}/test_HR"
        SR_OUTPUT_PATH="${BASE_OUTPUT_DIR}/${MODEL_NAME}/${dataset_name}"

        if [ "$SKIP_GENERATION" = false ]; then
            echo "--- Step 1: Generating images for ${dataset_name} ---"

            if [ ! -d "$LR_IMAGE_PATH" ]; then
                echo "ERROR: LR image directory not found at '$LR_IMAGE_PATH'. Skipping dataset."
                continue
            fi

            mkdir -p "$SR_OUTPUT_PATH"

            if ! python -W ignore test.py \
                --config "$CONFIG" \
                --finetuned_model_path "$checkpoint_path" \
                --image_path "$LR_IMAGE_PATH" \
                --output_dir "$SR_OUTPUT_PATH"; then
                echo "ERROR: generation failed for ${dataset_name}. Skipping metrics."
                continue
            fi
        else
            echo "--- Step 1: SKIPPING image generation for ${dataset_name} ---"
        fi

        echo "--- Step 2: Calculating metrics for ${dataset_name} ---"

        if [ ! -d "$SR_OUTPUT_PATH" ]; then
            echo "ERROR: Super-Resolution directory not found at '$SR_OUTPUT_PATH'. Run generation first."
            continue
        fi

        metric_cmd_args=(
            --sr_dir "$SR_OUTPUT_PATH"
            --dataset "$dataset_name"
            --name "$MODEL_NAME"
        )

        if [ -d "$GT_PATH" ]; then
            echo "Found Ground Truth directory: $GT_PATH"
            metric_cmd_args+=(--gt_dir "$GT_PATH")
        else
            echo "No Ground Truth directory found for $dataset_name. Calculating No-Reference metrics only."
        fi

        python -W ignore data_tools/metrics.py "${metric_cmd_args[@]}"
        echo
    done
done

echo "######################################################################"
echo "## All checkpoints and datasets processed."
echo "######################################################################"
