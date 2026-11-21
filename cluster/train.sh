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
    # The nodes meet at the IP address of the first one. This script runs on the first node of the job, so its
    # own address is used (node names do not resolve on every node of this cluster)
    MASTER_ADDR="$(hostname -I | awk '{print $1}')"
    export MASTER_ADDR MASTER_PORT="${MASTER_PORT:-29500}" NUM_MACHINES="$SLURM_NNODES"
    # An automatic restart on one node only would leave the other nodes waiting forever
    export MAX_RESTARTS=0
    GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-2}"
    echo "Multi-node run on $SLURM_NNODES nodes ($SLURM_JOB_NODELIST), $GPUS_PER_NODE GPUs each; main process at $MASTER_ADDR:$MASTER_PORT"

    # NCCL (GPU communication) picks its own network interface, which may not be reachable from the other node:
    # use the interface of MASTER_ADDR, which the nodes already reach each other through (override by exporting
    # NCCL_SOCKET_IFNAME before sbatch). NCCL_IB_DISABLE=1 (exported before sbatch) forces TCP instead of InfiniBand.
    if [ -z "${NCCL_SOCKET_IFNAME:-}" ]; then
        NCCL_SOCKET_IFNAME="$(ip -o -4 addr show | awk -v ip="$MASTER_ADDR" '{split($4, a, "/")} a[1] == ip {print $2; exit}')"
    fi
    export NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"
    echo "NCCL network interface: ${NCCL_SOCKET_IFNAME:-<auto>}, NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}"
    echo "GPU-related SLURM settings of the job:"
    env | grep -E "^SLURM_(GPUS|TRES|GRES|JOB_GPUS|STEP_GPUS)" | sed 's/^/    /' || true

    # One task per node; each starts one training process per GPU of its node. The job's whole-job GPU request
    # (--gpus=4, inherited by srun through SLURM_GPUS) would be invalid for a per-node step, so it is dropped
    # and the step asks for the GPUs of each node explicitly.
    env -u SLURM_GPUS -u SLURM_GPUS_PER_NODE -u SLURM_GPUS_PER_TASK -u SLURM_TRES_PER_TASK \
        srun --nodes="$SLURM_NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
        --gres="gpu:${GPUS_PER_NODE}" --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
        bash -c 'source cluster/env.sh && MACHINE_RANK="$SLURM_NODEID" in_container bash scripts/run_train.sh "$@"' _ "$@"
else
    in_container bash scripts/run_train.sh "$@"
fi
