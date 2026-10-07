#!/bin/bash
# Build a SegESR training folder from a directory of HR images.
#
# Usage: ./scripts/prepare_data.sh GT_DIR [SAVE_DIR]
#   GT_DIR    folder(s) of HR images (quote and space-separate multiple folders)
#   SAVE_DIR  output training folder (default: preset/datasets/train_datasets/LSDIR)
#
# Environment variables:
#   RAM_FT_PATH  DAPE weights (default: preset/models/DAPE.pth)
#   EPOCHS       degradation epochs for make_paired_data.py (default: 1)
#   WITH_GT_SAM  "false" to skip the SAM 2 conditions of the GT images, needed by `clean_sam_prob` (default: true)
#   WITH_GT_SEG  "true" to also save SAM 2 binary masks of the GT images in 'gt_seg/' (default: false)
#
# Resulting layout (read by segesr/dataloaders/paired_dataset.py):
#   SAVE_DIR/gt  sr_bicubic  lr  tag  dape_embeds  sam_embeds  seg_embeds  [sam_embeds_gt  seg_embeds_gt]  [gt_seg]

set -euo pipefail

cd "$(dirname "$0")/.."

if [ $# -lt 1 ]; then
    echo "Usage: $0 GT_DIR [SAVE_DIR]"
    exit 1
fi

read -r -a GT_DIRS <<< "$1"
SAVE_DIR="${2:-preset/datasets/train_datasets/LSDIR}"
RAM_FT_PATH="${RAM_FT_PATH:-preset/models/DAPE.pth}"
EPOCHS="${EPOCHS:-1}"
WITH_GT_SAM="${WITH_GT_SAM:-true}"
WITH_GT_SEG="${WITH_GT_SEG:-false}"

export PYTHONWARNINGS="ignore"

echo "--- Step 1/4: Degraded LR / GT pairs ---"
python data_tools/make_paired_data.py --gt_path "${GT_DIRS[@]}" --save_dir "$SAVE_DIR" --epoch "$EPOCHS"

echo "--- Step 2/4: RAM tags ---"
python data_tools/make_tags.py --root_path "$SAVE_DIR" --skip_existing

echo "--- Step 3/4: DAPE embeddings ---"
python data_tools/make_dape_embeds.py \
    --image_dir "$SAVE_DIR/sr_bicubic" \
    --embed_dir "$SAVE_DIR/dape_embeds" \
    --ram_ft_path "$RAM_FT_PATH" \
    --skip_existing

echo "--- Step 4/4: SAM 2 embeddings and mask logits ---"
python data_tools/sam_processing.py \
    --image_dir "$SAVE_DIR/sr_bicubic" \
    --embed_dir "$SAVE_DIR/sam_embeds" \
    --logit_dir "$SAVE_DIR/seg_embeds" \
    --skip_existing

if [ "$WITH_GT_SAM" = true ]; then
    echo "--- Extra: SAM 2 embeddings and mask logits of the GT images ---"
    python data_tools/sam_processing.py \
        --image_dir "$SAVE_DIR/gt" \
        --embed_dir "$SAVE_DIR/sam_embeds_gt" \
        --logit_dir "$SAVE_DIR/seg_embeds_gt" \
        --skip_existing
fi

if [ "$WITH_GT_SEG" = true ]; then
    echo "--- Extra: SAM 2 masks of the GT images ---"
    python data_tools/sam_processing.py \
        --image_dir "$SAVE_DIR/gt" \
        --mask_dir "$SAVE_DIR/gt_seg" \
        --skip_existing
fi

echo "Done. Training data written to '$SAVE_DIR'."
