# Shared settings of the SegESR cluster jobs, sourced by the other cluster/*.sh files.
# Not a job: do not submit it with sbatch.
# Every path can be overridden from the environment, e.g. `SEGESR_VENV=~/venvs/other sbatch cluster/train.sh`.

CONTAINER_IMAGE="${CONTAINER_IMAGE:-docker://pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel}"
SEGESR_SIF="${SEGESR_SIF:-$HOME/containers/pytorch_2.6.0-cuda12.6-cudnn9-devel.sif}"
SEGESR_VENV="${SEGESR_VENV:-$HOME/venvs/segesr}"

# Packages in ~/.local must not shadow the ones of the virtual environment
export PYTHONNOUSERSITE=1

# Write Python output to the job log immediately (it is buffered by default when not printed to a terminal)
export PYTHONUNBUFFERED=1

# Runs a command inside the container (with GPU support), with the SegESR virtual environment active.
in_container() {
    singularity exec --nv "$SEGESR_SIF" bash -c 'source "$0/bin/activate" && exec "$@"' "$SEGESR_VENV" "$@"
}
