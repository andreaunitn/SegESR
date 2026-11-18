#!/bin/bash
#SBATCH --job-name=bibandsr_train
#SBATCH --partition=department_only
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/%x_%j.log
#
# BiBandSR step O.1: fine-tuning of PiSA-SR from its official weights (bibandsr/train.py), BiBandSR environment.
# Settings (environment variables):
#   RUN         run name (required): outputs in preset/train_output/bibandsr/$RUN
#   LOSS        pisasr (the official losses, control) or split (losses split at the LR Nyquist limit) (required)
#   TRAIN_ARGS  extra flags of bibandsr/train.py, e.g. "--max_steps 100 --val_every 50"
#   STOP_AFTER  minutes after which the run saves its state and stops (default 1380, 1 h before the 1-day limit)
#   EVAL        1 (default): once finished, submit cluster/onestep_baselines.sh for the last checkpoint
#   MAX_SEGMENTS  most jobs in a chain (default 6)
# A run stopped by STOP_AFTER resubmits this script with the same settings and resumes from its state. It refuses to
# resume with a different code revision than the one it started with (code_rev in the run folder).
#
# Usage (from the repository root, or through cluster/hpc submit):
#   RUN=o1_split LOSS=split sbatch cluster/bibandsr_train.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

: "${RUN:?set RUN}" "${LOSS:?set LOSS}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
SEGMENT="${SEGMENT:-1}"
MAX_SEGMENTS="${MAX_SEGMENTS:-6}"
OUT="preset/train_output/bibandsr/$RUN"
mkdir -p "$OUT"

rev="$(git rev-parse --short HEAD)"
if [ -f "$OUT/code_rev" ] && [ "$(cat "$OUT/code_rev")" != "$rev" ]; then
    echo "The run started with code $(cat "$OUT/code_rev"), the repository is now at $rev: not resuming."
    echo "Restore that revision, or remove $OUT/code_rev to resume with the new code on purpose."
    exit 1
fi
echo "$rev" > "$OUT/code_rev"
echo "Run $RUN, loss $LOSS, segment $SEGMENT/$MAX_SEGMENTS, code $rev, node $(hostname)"

# shellcheck disable=SC2086  # TRAIN_ARGS holds several flags
in_bibandsr python -m bibandsr.train --loss "$LOSS" --output_dir "$OUT" \
    --stop_after_minutes "${STOP_AFTER:-1380}" $TRAIN_ARGS

if [ -f "$OUT/FINISHED" ]; then
    last="$(ls -v "$OUT"/checkpoints/model_*.pkl | tail -n 1)"
    echo "Finished; last checkpoint $last"
    if [ "${EVAL:-1}" = 1 ]; then
        sbatch --comment="code:$rev" --export=ALL,MODELS="$RUN=$last" cluster/onestep_baselines.sh
    fi
elif [ "$SEGMENT" -lt "$MAX_SEGMENTS" ]; then
    sbatch --comment="code:$rev" --job-name="$SLURM_JOB_NAME" \
        --export=ALL,RUN="$RUN",LOSS="$LOSS",SEGMENT="$((SEGMENT + 1))" cluster/bibandsr_train.sh
else
    echo "Not finished after $MAX_SEGMENTS jobs: not resubmitting."
    exit 1
fi
