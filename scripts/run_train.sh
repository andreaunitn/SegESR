#!/bin/bash
# Train SegESR.
#
# Usage: ./scripts/run_train.sh [CONFIG] [extra train.py flags...]
#   CONFIG defaults to configs/train_segesr.yaml (full model); configs/train_seesr.yaml is the baseline.
#   Extra flags override the config, e.g.
#   ./scripts/run_train.sh configs/train_seesr.yaml --attention_fusion=parallel --output_dir=preset/train_output/pafb
#
# Environment variables:
#   CUDA_VISIBLE_DEVICES  GPUs to use (default: 0)
#   MAX_RESTARTS          automatic restarts from the latest checkpoint after a crash (default: 10)

set -u

cd "$(dirname "$0")/.." || exit 1

CONFIG="${1:-configs/train_segesr.yaml}"
shift $(( $# > 0 ? 1 : 0 ))

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONWARNINGS="ignore"
MAX_RESTARTS="${MAX_RESTARTS:-10}"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config file '$CONFIG' not found."
    exit 1
fi

echo "Starting training with config: $CONFIG"

attempt=0
while true; do
    # The configs resume from the latest checkpoint, so a restart continues where the crash happened
    accelerate launch train.py --config "$CONFIG" "$@"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "-----------------------------------"
        echo "Training completed successfully."
        echo "-----------------------------------"
        break
    fi

    attempt=$((attempt + 1))
    if [ $attempt -gt "$MAX_RESTARTS" ]; then
        echo "Training failed with exit code $EXIT_CODE after $MAX_RESTARTS restarts. Giving up."
        exit $EXIT_CODE
    fi

    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "  Training failed with exit code $EXIT_CODE (restart $attempt/$MAX_RESTARTS)."
    echo "  Restarting from the latest checkpoint in 15 seconds."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    sleep 15 # Avoid rapid-fire crashes
done
