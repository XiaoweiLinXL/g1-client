#!/bin/bash
# Download a GR00T checkpoint (if not already present) and start the policy server.
#
# Usage (on the Azure VM):
#   bash launch_groot_server.sh <hf-repo-id> [port]
#
# Examples:
#   bash launch_groot_server.sh XiaoweiLinXL/unitree-GR00T-load-bottle-water-30000step
#   bash launch_groot_server.sh EmbodyX/UnitreeG1-GR00T-putaway-30000step 5556
#   bash launch_groot_server.sh yigao7117/sort_tool-groot-Nstep
#
# The local checkpoint directory is derived from the repo name:
#   ~/Isaac-GR00T/checkpoints/<repo-name>
#
# Port defaults to 5555. Port must be open in the Azure NSG for the robot's source IP.

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <hf-repo-id> [port]" >&2
    exit 1
fi

HF_REPO="$1"
PORT="${2:-5555}"
REPO_NAME="${HF_REPO##*/}"          # last component of repo id
LOCAL_DIR="$HOME/Isaac-GR00T/checkpoints/${REPO_NAME}"

cd ~/Isaac-GR00T

echo "=== GR00T policy server ==="
echo "  repo       : ${HF_REPO}"
echo "  local path : ${LOCAL_DIR}"
echo "  port       : ${PORT}"
echo ""

if [ -d "$LOCAL_DIR" ] && [ -f "$LOCAL_DIR/config.json" ]; then
    echo "Checkpoint already downloaded — skipping."
else
    echo "Downloading ${HF_REPO} → ${LOCAL_DIR}"
    uv run huggingface-cli download "$HF_REPO" --local-dir "$LOCAL_DIR"
fi

echo ""
echo "Starting server on port ${PORT}..."
exec uv run python gr00t/eval/run_gr00t_server.py \
    --model-path "$LOCAL_DIR" \
    --embodiment-tag new_embodiment \
    --device cuda:0 \
    --host 0.0.0.0 \
    --port "$PORT"
