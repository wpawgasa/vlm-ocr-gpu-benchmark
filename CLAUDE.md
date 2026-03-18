# CLAUDE.md — vlm-ocr-bench

## Project Overview

Reproducible benchmarking framework comparing NVIDIA B300 vs H100 GPUs for document OCR Vision-Language Models. Measures inference latency/throughput, training throughput, power efficiency, and OCR quality for 6 sub-10B VLMs.

**Full spec:** [docs/CODEBASE_SPEC.md](docs/CODEBASE_SPEC.md)

## Quick Reference

- **Language:** Python 3.11+
- **Package:** `vlm_ocr_bench` (under `src/vlm_ocr_bench/`)
- **CLI:** `vlm-ocr-bench` via Typer
- **Config:** YAML + Pydantic validation
- **Logging:** structlog (JSON)
- **Tests:** pytest (unit / integration / smoke)
- **Lint:** ruff + mypy
- **Entry point:** `src/vlm_ocr_bench/cli.py`

## Build & Run

```bash
pip install -e ".[dev]"          # Install with dev dependencies
pytest tests/unit/ -v            # Run unit tests (no GPU)
pytest tests/ -v --timeout=600   # Run all tests (needs GPU)
ruff check src/ tests/           # Lint
mypy src/vlm_ocr_bench/          # Type check
vlm-ocr-bench info               # Print GPU/environment info
vlm-ocr-bench run configs/experiments/quick_smoke.yaml  # Smoke test
```

## Architecture

```
CLI (Typer) → Config Loader → Phase Dispatcher
                                 ├── Phase A: Inference (vLLM)
                                 ├── Phase B: Training (HF Trainer + PEFT)
                                 └── Phase C: Quality Evaluation
All phases use: config/, models/, data/, profiling/, hardware/, utils/
Results flow to: analysis/ → reports + plots
```

## Code Conventions

- All config is YAML-driven, validated by Pydantic models in `config/schema.py`
- Hardware-specific logic is isolated in `hardware/` — core modules must be GPU-agnostic
- Each model has an adapter in `models/adapters/` implementing `OCRModelAdapter`
- Profiling runs in a separate process to avoid benchmark interference
- Results are stored as Parquet (primary) + CSV (portable)
- Use `structlog` for all logging — no `print()` statements
- Use `torch.cuda.Event` for precise GPU timing, not `time.time()`
- Seeds are set deterministically via `utils/seeds.py`
- All experiment results include full config snapshots and environment info

## Implementation Guides

Each module has a dedicated implementation guide with checklist:

| Module | Guide | Status |
|--------|-------|--------|
| Project setup | [docs/impl/01-project-setup.md](docs/impl/01-project-setup.md) | DONE |
| Configuration system | [docs/impl/02-config.md](docs/impl/02-config.md) | DONE |
| Hardware abstraction | [docs/impl/03-hardware.md](docs/impl/03-hardware.md) | DONE |
| Model adapters | [docs/impl/04-models.md](docs/impl/04-models.md) | DONE |
| Data pipeline | [docs/impl/05-data.md](docs/impl/05-data.md) | DONE |
| Inference benchmark (Phase A) | [docs/impl/06-inference.md](docs/impl/06-inference.md) | DONE |
| Training benchmark (Phase B) | [docs/impl/07-training.md](docs/impl/07-training.md) | TODO |
| Quality evaluation (Phase C) | [docs/impl/08-evaluation.md](docs/impl/08-evaluation.md) | TODO |
| Profiling & monitoring | [docs/impl/09-profiling.md](docs/impl/09-profiling.md) | TODO |
| Analysis & reporting | [docs/impl/10-analysis.md](docs/impl/10-analysis.md) | TODO |
| CLI & orchestration | [docs/impl/11-cli.md](docs/impl/11-cli.md) | TODO |
| Containerization | [docs/impl/12-containers.md](docs/impl/12-containers.md) | TODO |
| Testing | [docs/impl/13-testing.md](docs/impl/13-testing.md) | TODO |

## Implementation Order

Recommended build sequence (dependencies flow downward):

```
1. Project setup (pyproject.toml, directory structure)
2. Configuration system (schema.py, loader.py, resolvers.py)
3. Hardware abstraction (detector.py, profiles.py, power.py, clock.py)
4. Utilities (logging, timing, seeds, storage, environment)
5. Model adapters (base.py, then each adapter)
6. Data pipeline (datasets.py, loaders, image_utils, tokenization)
7. Profiling & monitoring (monitor.py, dcgm.py, nvidia_smi.py)
8. Inference benchmark — Phase A (engine.py, runner.py, workload.py, metrics.py)
9. Training benchmark — Phase B (lora.py, trainer.py, data_collator.py, callbacks.py)
10. Quality evaluation — Phase C (runner.py, parsers.py, all metrics)
11. Analysis & reporting (aggregator, statistics, roofline, pareto, tco, plots, report)
12. CLI & orchestration (cli.py, shell scripts)
13. Containerization (Dockerfiles, docker-compose, monitoring stack)
14. Testing (unit → integration → smoke)
```

## Key Design Rules

1. **Reproducibility:** Every run is defined by YAML config + seed. Results include full config snapshots.
2. **GPU-agnostic core:** No GPU-specific logic outside `hardware/`. Use adapters and config.
3. **Composable phases:** A/B/C run independently. Each produces self-contained artifacts.
4. **Fail-safe profiling:** Monitor runs in separate process; benchmark continues if profiling fails.
5. **Idempotent runs:** Same config + seed = same results (modulo GPU non-determinism, which is logged).
