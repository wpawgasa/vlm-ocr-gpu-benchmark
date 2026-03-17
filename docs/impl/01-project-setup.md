# 01 — Project Setup

## Overview

Bootstrap the repository structure, `pyproject.toml`, Makefile, and directory scaffold.

## Checklist

- [ ] Create `pyproject.toml` with all dependencies from spec section 14.1
  - Core ML: torch, transformers, peft, accelerate, deepspeed
  - Inference: vllm
  - Data: datasets, Pillow, pdf2image
  - Evaluation: nltk, pythainlp, editdistance, sacrebleu
  - Profiling: pynvml
  - Analysis: numpy, pandas, scipy, matplotlib, plotly, pyarrow
  - Config/CLI: pydantic, pyyaml, typer, rich
  - Logging: structlog
  - Dev extras: pytest, pytest-asyncio, pytest-cov, ruff, mypy, pre-commit
- [ ] Set entry point: `vlm-ocr-bench = "vlm_ocr_bench.cli:app"`
- [ ] Create `src/vlm_ocr_bench/__init__.py` with version
- [ ] Create all subpackage `__init__.py` files:
  - `config/`, `hardware/`, `models/`, `models/adapters/`, `data/`
  - `inference/`, `training/`, `evaluation/`, `evaluation/metrics/`, `evaluation/benchmarks/`
  - `profiling/`, `analysis/`, `utils/`
- [ ] Create `Makefile` with targets: `setup`, `lint`, `typecheck`, `test`, `test-unit`, `test-gpu`, `run`, `clean`
- [ ] Create `configs/` directory tree with placeholder YAMLs
- [ ] Create `scripts/` directory with shell script stubs
- [ ] Create `results/` directory with `.gitkeep` and `README.md`
- [ ] Add `.gitignore` (results/raw/, results/aggregated/, results/plots/, *.parquet, __pycache__, .venv, *.egg-info, .mypy_cache)
- [ ] Create `notebooks/` directory with placeholder `.ipynb` files
- [ ] Verify `pip install -e ".[dev]"` succeeds

## Key Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package metadata, deps, entry points |
| `Makefile` | Developer convenience commands |
| `src/vlm_ocr_bench/__init__.py` | Package root |

## Notes

- Use `requires-python = ">=3.11"` — needed for modern type hints
- FlashAttention is NOT in pyproject.toml — installed per-GPU in Docker
- Keep `[project.optional-dependencies]` for `dev` extras only
