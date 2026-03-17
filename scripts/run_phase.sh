#!/usr/bin/env bash
# Single-phase runner
set -euo pipefail

PHASE="${1:?Usage: run_phase.sh <phase> <config>}"
CONFIG="${2:?Usage: run_phase.sh <phase> <config>}"
echo "Running phase '$PHASE' with config: $CONFIG"
vlm-ocr-bench run "$CONFIG" --phase "$PHASE"
