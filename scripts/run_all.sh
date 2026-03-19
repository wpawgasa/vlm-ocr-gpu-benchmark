#!/usr/bin/env bash
# Full experiment orchestrator: validate → lock clocks → run phases → reset → analyze
set -euo pipefail

CONFIG="${1:-configs/experiments/full_benchmark.yaml}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo " vlm-ocr-bench — Full Experiment"
echo " Config: $CONFIG"
echo "============================================"

# 1. Validate config
echo "[1/5] Validating config..."
vlm-ocr-bench validate "$CONFIG"

# 2. Lock GPU clocks (optional, needs root)
if [[ "${LOCK_CLOCKS:-0}" == "1" ]]; then
    echo "[2/5] Locking GPU clocks..."
    bash "$SCRIPT_DIR/lock_clocks.sh"
else
    echo "[2/5] Skipping clock lock (set LOCK_CLOCKS=1 to enable)"
fi

# 3. Run all phases
echo "[3/5] Running benchmark..."
vlm-ocr-bench run "$CONFIG"

# 4. Reset GPU (optional)
if [[ "${LOCK_CLOCKS:-0}" == "1" ]]; then
    echo "[4/5] Resetting GPU clocks..."
    bash "$SCRIPT_DIR/reset_gpu.sh"
else
    echo "[4/5] Skipping GPU reset"
fi

# 5. Analyze results
echo "[5/5] Generating report..."
RESULTS_DIR="$(grep -oP 'output_dir:\s*\K\S+' "$CONFIG" 2>/dev/null || echo "results")"
vlm-ocr-bench analyze "$RESULTS_DIR"

echo "============================================"
echo " Experiment complete!"
echo "============================================"
