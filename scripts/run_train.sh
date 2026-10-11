#!/bin/bash
# Train SegESR.
#
# Usage: ./scripts/run_train.sh [CONFIG] [extra train.py flags...]
#   CONFIG defaults to configs/train_segesr.yaml (full model); configs/train_seesr.yaml is the baseline.
#   Extra flags override the config, e.g.
#   ./scripts/run_train.sh configs/train_seesr.yaml --attention_fusion=parallel --output_dir=preset/train_output/pafb
#
# Environment variables:
#   CUDA_VISIBLE_DEVICES        GPUs to use (default: all GPUs given to the job)
#   MAX_RESTARTS                automatic restarts from the latest checkpoint after a crash (default: 10)
#   MULTI_GPU_TOTAL_BATCH_SIZE  samples per update in multi-GPU runs (default: 8)
#
# Multi-GPU: only if the job was given several GPUs (e.g. `sbatch --gpus=2 cluster/train.sh ...`): one training
# process runs per GPU and each update uses MULTI_GPU_TOTAL_BATCH_SIZE samples (the accumulation steps are derived
# from it). With one GPU nothing changes. Give multi-GPU runs their own --output_dir: their batch size differs.

set -u

cd "$(dirname "$0")/.." || exit 1

CONFIG="${1:-configs/train_segesr.yaml}"
shift $(( $# > 0 ? 1 : 0 ))

# Number of GPUs given to this job: SLURM usually lists them in CUDA_VISIBLE_DEVICES (e.g. "0,1");
# otherwise the GPUs visible to nvidia-smi are counted
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=${#visible_gpus[@]}
else
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU" || true)
    [ "$NUM_GPUS" -ge 1 ] 2>/dev/null || NUM_GPUS=1
fi

NUM_MACHINES="${NUM_MACHINES:-1}"

if [ "$NUM_MACHINES" -gt 1 ] || [ "$NUM_GPUS" -gt 1 ]; then
    # The two GPUs of a node sit on different CPU sockets (nvidia-smi topo: SYS), where direct GPU-to-GPU (P2P)
    # transfers hang: GPUs of the same node exchange data through shared CPU memory instead (export
    # NCCL_P2P_DISABLE=0 before sbatch to try P2P anyway)
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
fi

if [ "$NUM_MACHINES" -gt 1 ]; then
    # Multi-node (set up by cluster/train.sh): this node runs one process per GPU, all nodes form one run
    LAUNCH=(accelerate launch --multi_gpu --num_machines "$NUM_MACHINES" --machine_rank "$MACHINE_RANK"
            --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT"
            --num_processes $(( NUM_GPUS * NUM_MACHINES )) --dynamo_backend no
            train.py --config "$CONFIG" --total_batch_size="${MULTI_GPU_TOTAL_BATCH_SIZE:-8}")
    echo "Node $MACHINE_RANK of $NUM_MACHINES: $NUM_GPUS GPUs ($(( NUM_GPUS * NUM_MACHINES )) in total, ${MULTI_GPU_TOTAL_BATCH_SIZE:-8} samples per update)"
elif [ "$NUM_GPUS" -gt 1 ]; then
    # --total_batch_size comes before the user's flags, so a --total_batch_size given on the command line wins
    LAUNCH=(accelerate launch --multi_gpu --num_processes "$NUM_GPUS" --num_machines 1 --dynamo_backend no
            train.py --config "$CONFIG" --total_batch_size="${MULTI_GPU_TOTAL_BATCH_SIZE:-8}")
    echo "Multi-GPU run on $NUM_GPUS GPUs (${MULTI_GPU_TOTAL_BATCH_SIZE:-8} samples per update)"
else
    LAUNCH=(accelerate launch --num_processes 1 --num_machines 1 --dynamo_backend no train.py --config "$CONFIG")
fi
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
    "${LAUNCH[@]}" "$@"
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
