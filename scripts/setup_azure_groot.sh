#!/bin/bash
# One-time setup for the Azure A10 VM to run the Isaac-GR00T policy server.
#
# Run from the repo root:
#   scp -i ~/azure_files/a10-1.5-inference-fabricio_key.pem \
#       scripts/setup_azure_groot.sh \
#       fabricio@a10-pi05-embodyx.southcentralus.cloudapp.azure.com:~/
#   ssh -i ~/azure_files/a10-1.5-inference-fabricio_key.pem \
#       fabricio@a10-pi05-embodyx.southcentralus.cloudapp.azure.com \
#       'bash ~/setup_azure_groot.sh'

set -euo pipefail

echo "=== [1/4] System dependencies ==="
sudo apt-get update -q
sudo apt-get install -y git git-lfs ffmpeg curl

git lfs install

echo "=== [2/4] Clone Isaac-GR00T ==="
if [ -d ~/Isaac-GR00T ]; then
    echo "  ~/Isaac-GR00T already exists — skipping clone"
else
    git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T ~/Isaac-GR00T
fi
cd ~/Isaac-GR00T

echo "=== [3/4] Install uv ==="
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# Make uv available in this script session
export PATH="$HOME/.local/bin:$PATH"

echo "=== [4/4] Install Isaac-GR00T dependencies (Python 3.10) ==="
# This installs torch 2.7.1+cu128, flash-attn, tensorrt, and all other deps.
# Takes several minutes on first run due to flash-attn wheel download.
uv sync --python 3.10

echo ""
echo "=== Setup complete ==="
echo "Verify: uv run python -c \"import gr00t; print('GR00T OK')\""
echo ""
echo "Next: download checkpoints and launch the server."
echo "  bash scripts/launch_groot_server.sh [9000step|18000step|30000step]"
