# 11 — CLI & Orchestration

## Overview

Typer-based CLI entry point and shell scripts for experiment orchestration.

## Checklist

### `cli.py`

- [ ] Create Typer app: `app = typer.Typer(name="vlm-ocr-bench")`
- [ ] Implement `run` command:
  - Args: config (Path), phase, model, gpu, precision, batch_size, dry_run, resume, output_dir
  - Load and validate config
  - Auto-detect GPU if not specified
  - Dispatch to phase runner(s)
  - Save config snapshot to results dir
- [ ] Implement `analyze` command:
  - Args: results_dir (Path), report_format (markdown|html|pdf)
  - Load raw results, run aggregation + analysis, generate report
- [ ] Implement `info` command:
  - Print detected GPU info, environment versions, available models
- [ ] Implement `validate` command:
  - Parse and validate config YAML, print errors/warnings
- [ ] Implement `list-models` command:
  - Print all registered model configs in table format
- [ ] Implement `download` command:
  - Args: target (models|data|all)
  - Pre-download model weights and/or datasets

### Shell Scripts

- [ ] `scripts/run_all.sh` — full experiment orchestrator (validate → lock clocks → phases → reset → analyze)
- [ ] `scripts/run_phase.sh` — single phase runner
- [ ] `scripts/setup_env.sh` — conda/pip environment setup
- [ ] `scripts/download_models.sh` — pre-download all model weights
- [ ] `scripts/download_data.sh` — pre-download benchmark datasets
- [ ] `scripts/lock_clocks.sh` — lock GPU clocks for stable benchmarking
- [ ] `scripts/reset_gpu.sh` — reset clocks + persistence mode
- [ ] `scripts/export_results.sh` — package results for sharing

## Key Rules

- CLI uses `typer` with type hints for auto-generated help
- Use `rich` for formatted console output (tables, progress bars)
- All shell scripts use `set -euo pipefail`
- `--dry-run` validates config and prints plan without executing
- `--resume` loads checkpoint from previous run and continues
- Entry point registered in pyproject.toml: `vlm-ocr-bench = "vlm_ocr_bench.cli:app"`
