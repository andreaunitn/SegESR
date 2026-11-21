#!/bin/bash
#SBATCH --job-name=onestep_baselines
#SBATCH --partition=department_only
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x_%j.log
#
# The one-step models PiSA-SR, OSEDiff (step O.0), FreqOrtho-SR (step M.0) and FiDeSR (step R.0) with their official
# weights and test scripts, scored with our protocol. Needs cluster/setup_bibandsr.sh first, and
# cluster/setup_trackr.sh for FiDeSR.
# FiDeSR runs through data_tools/fidesr_test.py (the official script with a choice of prompt): `fidesr` with the
# official empty prompt, `fidesr_ramlr` with RAM tags of the LR, `fidesr_ramhr` with RAM tags of the GT (an oracle).
#   1. the official test script (BiBandSR environment) writes to preset/datasets/external_raw/<model>/<dataset>;
#   2. data_tools/collect_sr.py brings the outputs to preset/datasets/output/<model>/<dataset>/sample00, at 4x the
#      LR size (the scripts round it down to a multiple of 8);
#   3. data_tools/metrics.py and data_tools/fine_structure.py.
# A model and dataset whose outputs are all collected already is not generated again.
# A MODELS entry <name>=<weights.pkl> runs PiSA-SR with other weights (e.g. a checkpoint of the archived
# bibandsr/train.py) and scores it as <name>.
# PiSA-SR and FreqOrtho-SR run in their default one-step mode. With LAMBDA_SEM set, they run instead in the adjustable
# mode, once per value of lambda_sem (lambda_pix = LAMBDA_PIX, default 1), scored as <name>_ls<value>.
#
# Usage (from the repository root):
#   sbatch cluster/onestep_baselines.sh
#   MODELS=pisasr DATASETS=RealSR sbatch cluster/onestep_baselines.sh
#   MODELS="o1_split=preset/archive/bibandsr/train_output/bibandsr/o1_split/checkpoints/model_2000.pkl" sbatch cluster/onestep_baselines.sh
#   MODELS=freqortho LAMBDA_SEM="0 0.5 1 1.5" sbatch cluster/onestep_baselines.sh
#   MODELS="fidesr fidesr_ramlr fidesr_ramhr" sbatch cluster/onestep_baselines.sh
# Results: python3 data_tools/results_table.py pisasr osediff; ... --prefix finestruct

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

read -r -a models <<< "${MODELS:-pisasr osediff}"
read -r -a datasets <<< "${DATASETS:-RealSR DRealSR DIV2K}"
read -r -a lambdas <<< "${LAMBDA_SEM:-}"
ROOT="$PWD"
MODELS_DIR="$ROOT/preset/models"
OUTPUT_DIR="preset/datasets/output"
RAW_DIR="preset/datasets/external_raw"

generate() {   # generate <kind> <weights> <LR dir> <raw output dir> <mode flags...>, all paths absolute
    local kind="$1" weights="$2" lr="$3" out="$4"
    shift 4
    case "$kind" in
        pisasr|freqortho)   # the same test script (FreqOrtho-SR is built on the PiSA-SR code)
            local code="$ROOT/preset/external/PiSA-SR"
            [ "$kind" = freqortho ] && code="$ROOT/preset/external/FreqOrtho-SR"
            in_bibandsr bash -c 'cd "$0" && exec python test_pisasr.py "$@"' "$code" \
                --pretrained_model_path "$MODELS_DIR/stable-diffusion-2-1-base" \
                --pretrained_path "$weights" \
                --process_size 512 --upscale 4 "$@" \
                --input_image "$lr" --output_dir "$out" ;;
        osediff)
            in_bibandsr bash -c 'cd "$0" && exec python test_osediff.py "$@"' "$ROOT/preset/external/OSEDiff" \
                --pretrained_model_name_or_path "$MODELS_DIR/stable-diffusion-2-1-base" \
                --osediff_path "$MODELS_DIR/osediff/osediff.pkl" \
                --ram_path "$MODELS_DIR/ram_swin_large_14m.pth" --ram_ft_path "$MODELS_DIR/DAPE.pth" \
                --process_size 512 --upscale 4 \
                -i "$lr" -o "$out" ;;
        fidesr|fidesr_ramlr|fidesr_ramhr)
            local prompt=empty
            [ "$kind" = fidesr_ramlr ] && prompt=ram_lr
            [ "$kind" = fidesr_ramhr ] && prompt=ram_hr
            # Official settings of the FiDeSR README (LF and HF strengths 0.2); the GTs are next to the LRs
            in_bibandsr bash -c 'cd "$0" && exec python "$@"' "$ROOT/preset/external/FiDeSR" \
                "$ROOT/data_tools/fidesr_test.py" \
                --pretrained_model_path "$MODELS_DIR/stable-diffusion-2-1-base" \
                --pretrained_path "$MODELS_DIR/fidesr/fidesr.pkl" \
                --ram_path "$MODELS_DIR/ram_swin_large_14m.pth" --hr_dir "$(dirname "$lr")/test_HR" \
                --prompt "$prompt" --process_size 512 --upscale 4 --hf_scale 0.2 --lf_scale 0.2 \
                -i "$lr" -o "$out" ;;
        *) echo "Unknown model: $kind"; exit 1 ;;
    esac
}

# Runs: "<name> <kind> <weights> <mode flags...>"
runs=()
for spec in "${models[@]}"; do
    name="${spec%%=*}"
    kind="$name"
    weights="$MODELS_DIR/pisasr/pisa_sr.pkl"
    [ "$name" = freqortho ] && weights="$MODELS_DIR/freqortho/freqortho_sr.pkl"
    if [ "$spec" != "$name" ]; then
        kind=pisasr
        weights="$ROOT/${spec#*=}"
    fi
    if [ "$kind" = osediff ] || [ "${kind#fidesr}" != "$kind" ] || [ ${#lambdas[@]} -eq 0 ]; then
        runs+=("$name $kind $weights --default")
    else
        for value in "${lambdas[@]}"; do
            runs+=("${name}_ls$value $kind $weights --lambda_pix ${LAMBDA_PIX:-1} --lambda_sem $value")
        done
    fi
done

for dataset in "${datasets[@]}"; do
    data_dir="preset/datasets/test_datasets/$dataset"
    for run in "${runs[@]}"; do
        read -r model kind weights mode <<< "$run"
        read -r -a mode <<< "$mode"
        [ "$kind" = osediff ] || [ "${kind#fidesr}" != "$kind" ] && mode=()
        out_dir="$OUTPUT_DIR/$model/$dataset"
        n_lr=$(find "$data_dir/test_LR" -maxdepth 1 -type f | wc -l)
        n_out=0     # find fails on a missing folder, which would end the script (set -e with pipefail)
        if [ -d "$out_dir/sample00" ]; then
            n_out=$(find "$out_dir/sample00" -maxdepth 1 -name "*.png" | wc -l)
        fi
        if [ "$n_out" -eq "$n_lr" ]; then
            echo "## $model on $dataset: outputs already collected"
        else
            echo "## $model on $dataset: generating"
            mkdir -p "$RAW_DIR/$model/$dataset"
            generate "$kind" "$weights" "$ROOT/$data_dir/test_LR" "$ROOT/$RAW_DIR/$model/$dataset" "${mode[@]}"
            in_container python data_tools/collect_sr.py --src_dir "$RAW_DIR/$model/$dataset" \
                --lr_dir "$data_dir/test_LR" --output_dir "$out_dir"
        fi
        in_container python -W ignore data_tools/metrics.py --sr_dir "$out_dir" --gt_dir "$data_dir/test_HR" \
            --dataset "$dataset" --name "$model"
        in_container python -W ignore data_tools/fine_structure.py --sr_dir "$out_dir" --gt_dir "$data_dir/test_HR" \
            --dataset "$dataset" --name "$model"
    done
done
