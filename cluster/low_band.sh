#!/bin/bash
#SBATCH --job-name=low_band
#SBATCH --partition=department_only
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x_%j.log
#
# Step A.0: where the error of SeeSR in the LR-supported band (below 1/8 cycle per HR pixel for x4) comes from.
# Run it after the test variants of SeeSR (cluster/test.sh with TAG=g1, g3, wavelet, nofix, s3):
#   1. seed part of the error per band (data_tools/seed_spread.py on <NAME>_s3, 3 samples), and the mean of the
#      samples saved as <NAME>_s3mean;
#   2. oracle: the band below 1/16 and below 1/8 taken from the GT (<NAME>_lowgt16, <NAME>_lowgt8);
#   3. realistic fusion: the band below 1/8 of sample00 taken from the mean of the 3 samples (<NAME>_s3lowmean8);
#   4. data_tools/metrics.py on the new folders, data_tools/fine_structure.py on every variant.
# The GPU is only used by metrics.py (pyiqa).
#
# Usage (from the repository root):
#   sbatch cluster/low_band.sh
#   NAME=seesr DATASETS=RealSR sbatch cluster/low_band.sh
# Results: python3 data_tools/results_table.py <NAME>; ... --prefix finestruct; ... --prefix seedspread

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

NAME="${NAME:-seesr}"
read -r -a datasets <<< "${DATASETS:-RealSR DRealSR}"
OUTPUT_DIR="preset/datasets/output"

score() {   # score <variant> <dataset> <gt_dir>
    in_container python -W ignore data_tools/metrics.py --sr_dir "$OUTPUT_DIR/$1/$2" --gt_dir "$3" --dataset "$2" --name "$1"
}
fine() {    # fine <variant> <dataset> <gt_dir>
    in_container python -W ignore data_tools/fine_structure.py --sr_dir "$OUTPUT_DIR/$1/$2" --gt_dir "$3" --dataset "$2" --name "$1"
}

for dataset in "${datasets[@]}"; do
    gt_dir="preset/datasets/test_datasets/$dataset/test_HR"
    seeds="$OUTPUT_DIR/${NAME}_s3/$dataset"

    echo "## seed part of the error: ${NAME}_s3 on $dataset"
    in_container python -W ignore data_tools/seed_spread.py --sr_dir "$seeds" --gt_dir "$gt_dir" --dataset "$dataset" \
        --name "${NAME}_s3" --mean_dir "$OUTPUT_DIR/${NAME}_s3mean/$dataset"

    for cutoff in 16:0.0625 8:0.125; do
        echo "## oracle: band below 1/${cutoff%%:*} from the GT, on $dataset"
        in_container python -W ignore data_tools/band_scale.py --sr_dir "$OUTPUT_DIR/$NAME/$dataset" \
            --output_dir "$OUTPUT_DIR/${NAME}_lowgt${cutoff%%:*}/$dataset" --low_from "$gt_dir" --cutoff "${cutoff#*:}"
    done

    echo "## fusion: band below 1/8 from the mean of the seeds, on $dataset"
    in_container python -W ignore data_tools/band_scale.py --sr_dir "$seeds" \
        --output_dir "$OUTPUT_DIR/${NAME}_s3lowmean8/$dataset" --low_from "$OUTPUT_DIR/${NAME}_s3mean/$dataset/sample00"

    for variant in "${NAME}_s3mean" "${NAME}_lowgt16" "${NAME}_lowgt8" "${NAME}_s3lowmean8"; do
        score "$variant" "$dataset" "$gt_dir"
    done
    for variant in "${NAME}_g1" "${NAME}_g3" "${NAME}_wavelet" "${NAME}_nofix" "${NAME}_s3" \
                   "${NAME}_s3mean" "${NAME}_lowgt16" "${NAME}_lowgt8" "${NAME}_s3lowmean8"; do
        if [ -d "$OUTPUT_DIR/$variant/$dataset/sample00" ]; then
            fine "$variant" "$dataset" "$gt_dir"
        else
            echo "skip $variant/$dataset: no outputs"
        fi
    done
done
