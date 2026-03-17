#!/usr/bin/env bash
# Full experiment orchestrator
set -euo pipefail

CONFIG="${1:-configs/experiments/full_benchmark.yaml}"
echo "Running full benchmark with config: $CONFIG"
vlm-ocr-bench run "$CONFIG"
