#!/bin/bash
#SBATCH --job-name=low_band_proxy
#SBATCH --partition=department_only
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x_%j.log
#
# Step A.0b, run after cluster/low_band.sh:
#   1. what a frequency-split guidance could give, without changing the sampler: the band below 1/16 or 1/8 of the
#      SeeSR output taken from the output without guidance (<NAME>_g1low16, <NAME>_g1low8);
#   2. how much of the GT the LR itself holds: bicubic upsampling of the LR (bicubic).
# Then data_tools/metrics.py (GPU, pyiqa) and data_tools/fine_structure.py on the new folders.
#
# Usage (from the repository root):
#   sbatch cluster/low_band_proxy.sh
#   NAME=seesr DATASETS=RealSR sbatch cluster/low_band_proxy.sh
# Results: python3 data_tools/results_table.py <NAME> bicubic; ... --prefix finestruct

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

NAME="${NAME:-seesr}"
read -r -a datasets <<< "${DATASETS:-RealSR DRealSR}"
OUTPUT_DIR="preset/datasets/output"

for dataset in "${datasets[@]}"; do
    data_dir="preset/datasets/test_datasets/$dataset"
    gt_dir="$data_dir/test_HR"

    for cutoff in 16:0.0625 8:0.125; do
        echo "## band below 1/${cutoff%%:*} from ${NAME}_g1, on $dataset"
        in_container python -W ignore data_tools/band_scale.py --sr_dir "$OUTPUT_DIR/$NAME/$dataset" \
            --output_dir "$OUTPUT_DIR/${NAME}_g1low${cutoff%%:*}/$dataset" \
            --low_from "$OUTPUT_DIR/${NAME}_g1/$dataset/sample00" --cutoff "${cutoff#*:}"
    done

    echo "## bicubic upsampling of the LR, on $dataset"
    in_container python -W ignore data_tools/bicubic.py --lr_dir "$data_dir/test_LR" --output_dir "$OUTPUT_DIR/bicubic/$dataset"

    for variant in "${NAME}_g1low16" "${NAME}_g1low8" bicubic; do
        in_container python -W ignore data_tools/metrics.py --sr_dir "$OUTPUT_DIR/$variant/$dataset" --gt_dir "$gt_dir" \
            --dataset "$dataset" --name "$variant"
        in_container python -W ignore data_tools/fine_structure.py --sr_dir "$OUTPUT_DIR/$variant/$dataset" \
            --gt_dir "$gt_dir" --dataset "$dataset" --name "$variant"
    done
done
