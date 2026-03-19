#!/usr/bin/env bash
# Environment setup using uv (preferred) or pip
set -euo pipefail

echo "Setting up vlm-ocr-bench environment..."

if command -v uv &>/dev/null; then
    echo "Using uv..."
    uv sync --extra dev
    echo "Downloading NLTK data..."
    uv run python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('wordnet', quiet=True)"
else
    echo "uv not found, falling back to pip..."
    pip install -e ".[dev]"
    python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('wordnet', quiet=True)"
fi

echo "Setup complete."
echo "Run 'vlm-ocr-bench info' to verify installation."
