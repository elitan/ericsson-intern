#!/bin/bash
set -e

echo "=== mmWave Beam Prediction — Cloud Run ==="
echo "Started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

cd /workspace/mmwave-beam-prediction 2>/dev/null || cd "$(dirname "$0")"

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv sync

echo ""
echo "=== Scenario 1: boston5g_28 ==="
uv run python run_simulation.py --source deepmimo --scenario boston5g_28 --seeds 42,123,456

echo ""
echo "=== Scenario 2: O1_28 ==="
uv run python run_simulation.py --source deepmimo --scenario O1_28 --seeds 42,123,456

echo ""
echo "=== Results ==="
ls -la figures/*.png 2>/dev/null || echo "No figures"
echo ""
echo "Finished: $(date)"
