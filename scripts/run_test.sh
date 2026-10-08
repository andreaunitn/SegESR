#!/bin/bash
# Run SegESR inference + metrics on every checkpoint of a training run and every test dataset.
#
# Usage: ./scripts/run_test.sh [CHECKPOINT_BASE_DIR] [CONFIG]
#   CHECKPOINT_BASE_DIR  folder containing 'checkpoint-*' subfolders (default: preset/train_output/segesr),
#                        or a single model folder with 'unet/' and 'controlnet/' (e.g. preset/models/seesr)
#   CONFIG               test config (default: configs/test_default.yaml)
#
# Environment variables:
#   DATASETS          space-separated dataset names (default: "DIV2K DRealSR RealLR200 RealSR")
#   SKIP_GENERATION   "true" to only (re)compute metrics on existing outputs (default: false)
#   TEST_ARGS         extra test.py flags, e.g. "--sam_refresh_timesteps" to disable the SAM 2 refresh (default: none)
#   TAG               suffix of the output / metric names, to keep test variants apart, e.g. "norefresh" (default: none)
#
# Images already generated are skipped, so re-running the same command resumes an interrupted run.
#   CUDA_VISIBLE_DEVICES  GPU to use (default: 0)

set -u

cd "$(dirname "$0")/.." || exit 1

CHECKPOINT_BASE_DIR="${1:-preset/train_output/segesr}"
CONFIG="${2:-configs/test_default.yaml}"
read -r -a datasets <<< "${DATASETS:-DIV2K DRealSR RealLR200 RealSR}"
SKIP_GENERATION="${SKIP_GENERATION:-false}"
read -r -a test_args <<< "${TEST_ARGS:-}"
TAG="${TAG:-}"

BASE_TEST_DATA_DIR="preset/datasets/test_datasets"
BASE_OUTPUT_DIR="preset/datasets/output"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config file '$CONFIG' not found."
    exit 1
fi

if [ -d "$CHECKPOINT_BASE_DIR/unet" ] && [ -d "$CHECKPOINT_BASE_DIR/controlnet" ]; then
    checkpoints=("$CHECKPOINT_BASE_DIR")   # a single model, e.g. the original SeeSR
else
    mapfile -t checkpoints < <(find "$CHECKPOINT_BASE_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V)
fi
if [ ${#checkpoints[@]} -eq 0 ]; then
    echo "ERROR: no 'checkpoint-*' folder (or 'unet/' + 'controlnet/') found in '$CHECKPOINT_BASE_DIR'."
    exit 1
fi

mkdir -p "$BASE_OUTPUT_DIR"

for checkpoint_path in "${checkpoints[@]}"; do
    # Outputs and metrics are named <run>_<checkpoint> (or <model> for a single model), so runs never overwrite each other
    if [ "$checkpoint_path" = "$CHECKPOINT_BASE_DIR" ]; then
        MODEL_NAME=$(basename "$checkpoint_path")
    else
        MODEL_NAME="$(basename "$CHECKPOINT_BASE_DIR")_$(basename "$checkpoint_path")"
    fi
    MODEL_NAME="${MODEL_NAME}${TAG:+_$TAG}"

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
                --output_dir "$SR_OUTPUT_PATH" \
                --skip_existing \
                ${test_args[@]+"${test_args[@]}"}; then
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
