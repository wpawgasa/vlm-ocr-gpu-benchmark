#!/usr/bin/env bash
# Environment setup
set -euo pipefail

echo "Installing vlm-ocr-bench with dev dependencies..."
pip install -e ".[dev]"
echo "Setup complete."
