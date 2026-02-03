#!/bin/bash
set -e

echo "=== mmWave Beam Prediction — Cloud Run ==="
echo "Started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

cd /workspace/conformal-beam-prediction

# Install uv if not present
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Try Sionna install (best-effort, non-fatal)
pip install tensorflow sionna 2>/dev/null || echo "Sionna unavailable, skipping"

uv sync

# === Static pipeline: DeepMIMO scenarios ===
mkdir -p figures/boston5g_28 figures/O1_28

echo ""
echo "=== Static pipeline: boston5g_28 ==="
uv run python run_simulation.py --source deepmimo --scenario boston5g_28
mv figures/*.png figures/boston5g_28/ 2>/dev/null || true

echo ""
echo "=== Static pipeline: O1_28 ==="
uv run python run_simulation.py --source deepmimo --scenario O1_28
mv figures/*.png figures/O1_28/ 2>/dev/null || true

# === Temporal pipeline ===
echo ""
echo "=== Generating temporal channel data ==="
uv run python -m beampred.sionna_channel --all-speeds --n-ues 500 --duration 10

echo ""
echo "=== Running temporal pipeline ==="
uv run python run_temporal.py --n-ues 500 --duration 10

# === Results ===
echo ""
echo "=== Results ==="
ls -la figures/*.png 2>/dev/null || echo "No static figures"
ls -la figures/temporal/ 2>/dev/null || echo "No temporal figures"
echo ""
echo "Finished: $(date)"
