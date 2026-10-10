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

# TensorBoard runs from its own small environment: the training environment has TensorBoard 2.14, whose
# viewer breaks with recent protobuf versions (the training side only writes logs, which works fine)
TB_VENV="${TB_VENV:-$HOME/venvs/tensorboard}"
if [ ! -x "$TB_VENV/bin/tensorboard" ]; then
    echo "Creating the TensorBoard environment in $TB_VENV (first run only)..."
    singularity exec "$SEGESR_SIF" python -m venv "$TB_VENV"
    singularity exec "$SEGESR_SIF" "$TB_VENV/bin/pip" install --quiet "tensorboard==2.18.0"
fi
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

# --load_fast=false: the classic event reader (the fast one can fail silently on network filesystems such as /home)
SEGESR_VENV="$TB_VENV" in_container tensorboard --logdir tensorboard --port "$PORT" "${BIND[@]}" --load_fast=false --reload_interval 30
