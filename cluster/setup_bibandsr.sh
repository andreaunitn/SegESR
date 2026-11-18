#!/bin/bash
#SBATCH --job-name=bibandsr_setup
#SBATCH --partition=department_only
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x_%j.log
#
# One-time setup of BiBandSR: a second virtual environment on the same container as SegESR (~/venvs/bibandsr,
# requirements-bibandsr.txt), the official code of the one-step baselines and their weights:
#   preset/external/PiSA-SR, preset/external/OSEDiff   code at a pinned commit (GitHub archives)
#   preset/models/stable-diffusion-2-1-base             SD 2.1-base, safetensors only (*)
#   preset/models/pisasr/pisa_sr.pkl                    PiSA-SR weights (Google Drive folder of the authors)
#   preset/models/osediff/osediff.pkl                   OSEDiff weights (in the OSEDiff repository)
# (*) The official stabilityai repository is no longer public: the community mirror sd2-community is used. Only
#     safetensors and configs are downloaded (no pickle), and the VAE and text encoder are compared with the SD 2
#     base copy in preset/models/stable-diffusion-2-base, which should be identical.
#
# Usage (from the repository root): sbatch cluster/setup_bibandsr.sh
# Safe to re-run: what already exists is reused.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source cluster/env.sh

PISASR_COMMIT=9c76e12e261d1980148738a0cff41e5e27d18c9f
OSEDIFF_COMMIT=cdd3b7b04850fa9bb7faf8f2187edd279cc96507
PISASR_WEIGHTS_FOLDER="https://drive.google.com/drive/folders/1oLetijWNd59xwJE5oU-eXylQBifxWdss"
SD21_MIRROR=sd2-community/stable-diffusion-2-1-base

echo "--- 1/5: Virtual environment -> $BIBANDSR_VENV"
if [ ! -d "$BIBANDSR_VENV" ]; then
    singularity exec "$SEGESR_SIF" python -m venv --system-site-packages "$BIBANDSR_VENV"
fi
in_bibandsr bash -c '
    set -euo pipefail
    pip install --upgrade pip
    # Pin the container torch / torchvision, so that no package can replace them
    python -c "import torch, torchvision; print(f\"torch=={torch.__version__}\ntorchvision=={torchvision.__version__}\")" \
        > "$VIRTUAL_ENV/constraints.txt"
    pip install -r requirements-bibandsr.txt -c "$VIRTUAL_ENV/constraints.txt"
'

echo "--- 2/5: Official code of the baselines (preset/external)"
mkdir -p preset/external
fetch_repo() {   # fetch_repo <name> <github owner/repo> <commit>
    local dir="preset/external/$1"
    if [ ! -f "$dir/.commit" ] || [ "$(cat "$dir/.commit")" != "$3" ]; then
        rm -rf "$dir" && mkdir -p "$dir"
        curl -sSfL "https://github.com/$2/archive/$3.tar.gz" | tar -xz -C "$dir" --strip-components=1
        echo "$3" > "$dir/.commit"
    fi
    echo "$1 at $(cat "$dir/.commit")"
}
fetch_repo PiSA-SR csslc/PiSA-SR "$PISASR_COMMIT"
fetch_repo OSEDiff cswry/OSEDiff "$OSEDIFF_COMMIT"
# Their RAM code loads the BERT tokenizer from a path of the authors' machine; use the Hub id, as third_party/ram does
for repo in PiSA-SR OSEDiff; do
    sed -i "s|BertTokenizer.from_pretrained('/home/notebook/data/group/LowLevelLLM/LLM/bert-base-uncased', local_files_only=True)|BertTokenizer.from_pretrained('bert-base-uncased')|" \
        "preset/external/$repo/ram/models/utils.py"
done
grep -n "BertTokenizer.from_pretrained(" preset/external/*/ram/models/utils.py

echo "--- 3/5: Weights (preset/models)"
mkdir -p preset/models/osediff preset/models/pisasr
cp -n preset/external/OSEDiff/preset/models/osediff.pkl preset/models/osediff/
if [ ! -f preset/models/pisasr/pisa_sr.pkl ]; then
    in_bibandsr gdown --folder "$PISASR_WEIGHTS_FOLDER" -O preset/models/pisasr
fi
ls -l preset/models/pisasr preset/models/osediff
in_bibandsr python - "$SD21_MIRROR" <<'EOF'
import sys
from huggingface_hub import snapshot_download
patterns = ["model_index.json", "scheduler/*", "tokenizer/*", "feature_extractor/*",
            "text_encoder/config.json", "text_encoder/model.safetensors",
            "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
            "vae/config.json", "vae/diffusion_pytorch_model.safetensors"]
snapshot_download(sys.argv[1], local_dir="preset/models/stable-diffusion-2-1-base", allow_patterns=patterns)
EOF

echo "--- 4/5: Checks"
in_bibandsr python - <<'EOF'
import glob, torch, diffusers, peft, transformers
from diffusers import AutoencoderKL
from transformers import CLIPTextModel

print(f"torch {torch.__version__} | diffusers {diffusers.__version__} | peft {peft.__version__} | transformers {transformers.__version__}")

# The SD 2.1-base mirror: VAE and text encoder should equal those of SD 2 base (only the UNet was fine-tuned)
for name, cls in (("vae", AutoencoderKL), ("text_encoder", CLIPTextModel)):
    a = cls.from_pretrained("preset/models/stable-diffusion-2-1-base", subfolder=name).state_dict()
    b = cls.from_pretrained("preset/models/stable-diffusion-2-base", subfolder=name).state_dict()
    same = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    print(f"SD 2.1-base mirror, {name}: {'identical to SD 2 base' if same else 'DIFFERENT from SD 2 base'}")

# The weights must load with the safe loader of torch 2.6 (weights_only=True, the default), as the test scripts do
for path in ["preset/models/osediff/osediff.pkl"] + glob.glob("preset/models/pisasr/*.pkl"):
    sd = torch.load(path, weights_only=True)
    print(f"{path}: {len(sd)} entries, loaded with weights_only=True")
EOF

echo "--- 5/5: Tests"
in_bibandsr python -m pytest -q -p no:cacheprovider tests/test_collect_sr.py

echo "Setup complete."
