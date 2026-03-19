# CLI & Orchestration Module

**Date**: 2026-03-19
**Branch**: `feature/cli-orchestration`

## Overview

Implements the Typer-based CLI entry point with all commands and shell scripts for experiment orchestration.

## Changes

### Modified Files
- `src/vlm_ocr_bench/cli.py` — Full CLI implementation with 6 commands
- `scripts/run_all.sh` — Enhanced full experiment orchestrator (validate → lock clocks → run → reset → analyze)
- `scripts/setup_env.sh` — Updated to use uv (preferred) with pip fallback

### New Files
- `tests/unit/test_cli.py` — 18 unit tests

## CLI Commands

| Command | Description |
|---------|-------------|
| `vlm-ocr-bench info` | Print GPU and environment information |
| `vlm-ocr-bench list-models` | List all registered OCR models in table format |
| `vlm-ocr-bench validate <config>` | Validate experiment config YAML |
| `vlm-ocr-bench run <config>` | Run benchmark experiment (supports --phase, --model, --dry-run, --output-dir) |
| `vlm-ocr-bench analyze <results_dir>` | Generate Markdown report from results |
| `vlm-ocr-bench download [target]` | Pre-download models/data/all |

## Testing

- 18 unit tests in `tests/unit/test_cli.py`
- All 464 unit tests pass
- ruff clean, mypy clean
