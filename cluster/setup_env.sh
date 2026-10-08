#!/bin/bash
#SBATCH --job-name=segesr_setup
#SBATCH --partition=department_only
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x_%j.log
#
# One-time setup of the SegESR environment on the cluster: pulls the PyTorch container, creates a
# virtual environment on top of it (reusing the container's torch), installs the dependencies,
# downloads the SAM 2.1 weights and runs the test suite.
#
# Usage (from the repository root): sbatch cluster/setup_env.sh
# Safe to re-run: existing container image and virtual environment are reused.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

echo "--- 1/5: Container image -> $SEGESR_SIF"
mkdir -p "$(dirname "$SEGESR_SIF")"
if [ ! -f "$SEGESR_SIF" ]; then
    singularity pull "$SEGESR_SIF" "$CONTAINER_IMAGE"
fi

echo "--- 2/5: Virtual environment -> $SEGESR_VENV"
if [ ! -d "$SEGESR_VENV" ]; then
    singularity exec "$SEGESR_SIF" python -m venv --system-site-packages "$SEGESR_VENV"
fi

echo "--- 3/5: Python packages"
in_container bash -c '
    set -euo pipefail
    pip install --upgrade pip

    # Pin the container torch / torchvision, so that no package can replace them
    python -c "import torch, torchvision; print(f\"torch=={torch.__version__}\ntorchvision=={torchvision.__version__}\")" \
        > "$VIRTUAL_ENV/constraints.txt"

    pip install -r requirements-cluster.txt -c "$VIRTUAL_ENV/constraints.txt"

    # facexlib (a pyiqa dependency) pulls the GUI build of OpenCV, which needs system libraries missing in
    # the container (libgthread, libGL). Both builds share the cv2 folder, so the headless one is reinstalled.
    pip uninstall -y opencv-python
    pip install --force-reinstall --no-deps "$(grep -E "^opencv-python-headless" requirements-cluster.txt)"

    # Without build isolation, the SAM 2 extension is compiled against the container torch.
    # Installed from the GitHub archive: the container has no git.
    pip install --no-build-isolation "https://github.com/facebookresearch/sam2/archive/refs/heads/main.zip"

    # xformers is not used: its wheels do not match the container CUDA, and the PyTorch attention is used instead
    pip uninstall -y xformers
'

echo "--- 4/5: Check"
in_container python - <<'EOF'
import torch, diffusers, transformers, accelerate, sam2, pyiqa

print(f"torch {torch.__version__} | diffusers {diffusers.__version__} | transformers {transformers.__version__} | accelerate {accelerate.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    print(f"GPU compute capability {major}.{minor}: " + ("bf16 supported" if major >= 8 else "use mixed_precision fp16"))
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
EOF

echo "--- 5/5: SAM 2.1 weights and tests"
in_container python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/sam2.1-hiera-large')"
in_container python -m pytest -q -p no:cacheprovider

echo "Setup complete."
