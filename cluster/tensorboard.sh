#!/bin/bash
#SBATCH --job-name=segesr_tensorboard
#SBATCH --partition=department_only
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x_%j.log
#
# Starts TensorBoard on tensorboard/ (training and test logs of every run).
#
# Usage (from the repository root):
#   sbatch cluster/tensorboard.sh [PORT]   # as a job on a compute node; its log says how to connect
#   bash cluster/tensorboard.sh [PORT]     # directly on the submit node (needs Singularity there)
# PORT defaults to 6006; pick another one if it is already in use.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source cluster/env.sh

PORT="${1:-6006}"
HOST="$(hostname)"
# The submit node may not resolve compute node names, so the tunnel uses the node's IP address
HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
HOST_IP="${HOST_IP:-$HOST}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
    TUNNEL="ssh -N -L ${PORT}:${HOST_IP}:${PORT} segesr"
    BIND=(--bind_all)            # reachable from the submit node, through which the tunnel goes
else
    TUNNEL="ssh -N -L ${PORT}:localhost:${PORT} segesr"
    BIND=(--host localhost)
fi

echo "TensorBoard on ${HOST}:${PORT}. On your Mac, run:"
echo "    ${TUNNEL}"
echo "then open http://localhost:${PORT} in the browser (Ctrl+C in that terminal closes the tunnel)."

in_container tensorboard --logdir tensorboard --port "$PORT" "${BIND[@]}"
