# B300/H100 GPU Benchmark for VLM in OCR tasks

Reproducible benchmarking framework comparing NVIDIA B300 vs H100 GPUs for document OCR Vision-Language Models. Measures inference latency/throughput, training throughput, power efficiency, and OCR quality for 6 sub-10B VLMs.

## Models

| Model | Params | Tier | HuggingFace ID |
|-------|--------|------|----------------|
| PaddleOCR-VL | 0.9B | Ultra Compact | `PaddlePaddle/PaddleOCR-VL-0.9B` |
| DOTS OCR 1.5 | 3.0B | Compact | `rednote-hilab/dots.ocr-1.5` |
| Nanonets OCR2 | 3.0B | Compact | `nanonets/Nanonets-OCR-s` |
| DeepSeek OCR | 3.0B | Compact | `deepseek-ai/DeepSeek-OCR-3B` |
| olmOCR2 | 7.0B | Midsize | `allenai/olmOCR2-7B-0225-preview` |
| Chandra OCR | 9.0B | Midsize | `amaai-lab/Chandra-OCR-9B` |

## Benchmarks

| Dataset | Samples | Purpose |
|---------|---------|---------|
| OmniDocBench v1.5 | 1,355 | Per-document-type quality (9 types) |
| olmOCR-Bench | — | General OCR quality |
| Real5-OmniDocBench | 6,775 | Real-world capture scenarios (5 types) |
| Thai Docs 500 | 500 | Thai financial/legal documents |

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Verify installation
vlm-ocr-bench info

# Run smoke test (requires GPU)
vlm-ocr-bench run configs/experiments/quick_smoke.yaml

# Run full benchmark
vlm-ocr-bench run configs/experiments/full_benchmark.yaml
```

## CLI Commands

```
vlm-ocr-bench info                    # GPU and environment info
vlm-ocr-bench list-models             # Registered model table
vlm-ocr-bench validate <config>       # Validate config YAML
vlm-ocr-bench run <config>            # Run benchmark experiment
vlm-ocr-bench run <config> --phase inference   # Single phase
vlm-ocr-bench run <config> --dry-run  # Print plan without executing
vlm-ocr-bench analyze <results_dir>   # Generate report from results
vlm-ocr-bench download [models|data|all]  # Pre-download guidance
```

## Architecture

```
CLI (Typer) → Config Loader → Phase Dispatcher
                                 ├── Phase A: Inference (vLLM)
                                 ├── Phase B: Training (HF Trainer + PEFT/LoRA)
                                 └── Phase C: Quality Evaluation
All phases use: config/, models/, data/, profiling/, hardware/
Results flow to: analysis/ → reports + plots
```

### Project Structure

```
src/vlm_ocr_bench/
├── config/        # YAML config loading + Pydantic validation
├── hardware/      # GPU detection, profiles, power monitoring, clocks
├── models/        # 6 model adapters implementing OCRModelAdapter
├── data/          # Dataset loaders (HuggingFace + local)
├── inference/     # Phase A: vLLM engine, workload generation, metrics
├── training/      # Phase B: LoRA fine-tuning, callbacks, data collation
├── evaluation/    # Phase C: edit distance, BLEU, METEOR, structural metrics
├── profiling/     # Background GPU monitoring (separate process)
├── analysis/      # Aggregation, statistics, roofline, Pareto, TCO, plots
├── utils/         # Logging, timing, seeds
└── cli.py         # Typer CLI entry point
```

## Benchmark Phases

### Phase A: Inference

Measures latency (p50/p95/p99), throughput (pages/s, tokens/s), power efficiency, and memory usage across batch sizes, resolutions, and precision modes (BF16, FP8, FP4).

### Phase B: Training

Measures LoRA fine-tuning throughput (samples/s), finds maximum batch sizes, tracks convergence, and estimates training cost per epoch.

### Phase C: Quality Evaluation

Computes OCR quality metrics with language-aware tokenization:
- **Edit distance** (normalized, block-level with Hungarian matching)
- **BLEU** (Thai via pythainlp, Chinese character-level, English via NLTK)
- **METEOR** (language-aware)
- **Structural accuracy** (table cell matching, formula token F1)
- **Precision degradation** analysis (BF16 baseline vs FP8/FP4)

## Configuration

All experiments are defined by YAML configs with hierarchical merging:

```
configs/
├── defaults.yaml              # Global defaults
├── hardware/
│   ├── h100_sxm.yaml          # H100 GPU settings
│   └── b300_sxm.yaml          # B300 GPU settings
├── models/
│   └── <model_name>.yaml      # Per-model overrides
├── phases/
│   ├── inference.yaml          # Phase A settings
│   ├── training.yaml           # Phase B settings
│   └── quality.yaml            # Phase C settings
└── experiments/
    ├── full_benchmark.yaml     # Full benchmark suite
    ├── quick_smoke.yaml        # CI smoke test (~2 min)
    └── single_model_debug.yaml # Single model debugging
```

Merge order: `defaults → hardware → models → phases → experiment → CLI overrides`

## Analysis & Reporting

After benchmarking, generate analysis reports:

```bash
vlm-ocr-bench analyze results/
```

Produces a Markdown report with:
- Executive summary with headline speedups
- Throughput comparison tables and charts
- Latency CDF plots
- Power efficiency analysis
- TCO projections (GPU requirements, monthly/annual costs)
- Roofline analysis (compute vs memory bottleneck)
- Pareto frontier (quality vs throughput tradeoffs)
- Model x GPU pairing recommendations

## Docker

GPU-specific Docker images with monitoring stack:

```bash
# Build for H100
cp docker/.env.template docker/.env
GPU_TYPE=h100 docker compose -f docker/docker-compose.yaml build

# Run with monitoring
GPU_TYPE=h100 docker compose -f docker/docker-compose.yaml up

# Grafana dashboard at http://localhost:3000
```

Services: `benchmark`, `dcgm-exporter` (GPU metrics), `prometheus`, `grafana`

## Testing

```bash
# Unit tests (no GPU, runs in CI)
pytest tests/unit/ -v --timeout=120

# Integration tests (GPU required)
pytest tests/integration/ -v --timeout=600

# Smoke tests (GPU required, end-to-end)
pytest tests/smoke/ -v --timeout=600

# Lint and type check
ruff check src/ tests/
mypy src/vlm_ocr_bench/
```

| Tier | GPU | CI | Runtime |
|------|-----|----|---------|
| Unit | No | Yes | <30s |
| Integration | Yes | GPU CI | <5min |
| Smoke | Yes | GPU CI | <10min |

## Requirements

- Python >= 3.11
- CUDA GPU (H100 or B300 for full benchmarks)
- Key dependencies: PyTorch, Transformers, PEFT, vLLM, structlog, Typer, Rich

## License

See [LICENSE](LICENSE) for details.
