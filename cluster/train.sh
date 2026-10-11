#!/bin/bash
#SBATCH --job-name=segesr_train
#SBATCH --partition=department_only
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x_%j.log
#
# Trains SegESR (scripts/run_train.sh) on the cluster.
#
# Usage (from the repository root):
#   sbatch cluster/train.sh [CONFIG] [extra train.py flags...]
#   e.g. sbatch cluster/train.sh configs/train_segesr.yaml
#        sbatch cluster/train.sh configs/train_seesr.yaml --attention_fusion=parallel --output_dir=preset/train_output/pafb
#
# Multi-GPU (experimental): request several GPUs at submission; the run then uses one process per GPU and
# 8 samples per update (MULTI_GPU_TOTAL_BATCH_SIZE to change it). Each process needs its own CPUs and RAM:
#   sbatch --gpus=2 --cpus-per-task=16 --mem=64G cluster/train.sh configs/train_segesr.yaml --output_dir=preset/train_output/segesr_2gpu
# 4 GPUs = 2 nodes x 2 GPUs (no node has more): one copy of the launcher runs on each node.
#   sbatch --nodes=2 --ntasks-per-node=1 --gpus=4 --cpus-per-task=16 --mem=64G cluster/train.sh configs/train_segesr.yaml \
#          --output_dir=preset/train_output/segesr_4gpu
# Multi-node runs do not restart automatically after a crash (resubmit the same command to resume).
#
# The configs resume from the latest checkpoint of `output_dir`, so if the job hits the time limit of
# the partition, submitting the same command again continues the training.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

if [ "${SLURM_NNODES:-1}" -gt 1 ]; then
    # The nodes meet at the IP address of the first one (node names do not resolve on every node of this cluster)
    first_node="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)"
    MASTER_ADDR="$(srun --nodes=1 --ntasks=1 -w "$first_node" hostname -I | awk '{print $1}')"
    export MASTER_ADDR MASTER_PORT="${MASTER_PORT:-29500}" NUM_MACHINES="$SLURM_NNODES"
    # An automatic restart on one node only would leave the other nodes waiting forever
    export MAX_RESTARTS=0
    echo "Multi-node run on $SLURM_NNODES nodes ($SLURM_JOB_NODELIST); main process at $MASTER_ADDR:$MASTER_PORT"

    # One task per node; each starts one training process per GPU of its node. No GPU option here: a step gets
    # all the job's GPUs on its node by default (and this cluster rejects a per-node GPU request inside a --gpus job)
    srun --nodes="$SLURM_NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
        --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
        bash -c 'source cluster/env.sh && MACHINE_RANK="$SLURM_NODEID" in_container bash scripts/run_train.sh "$@"' _ "$@"
else
    in_container bash scripts/run_train.sh "$@"
fi
