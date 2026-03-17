# VLM-OCR GPU Benchmark — Codebase Specification

**Project:** `vlm-ocr-bench` — NVIDIA B300 vs H100 Benchmark for Document OCR VLMs
**Version:** 1.0
**Date:** March 2026
**Status:** Proposed

---

## Table of Contents

1. [Overview](#1-overview)
2. [Repository Structure](#2-repository-structure)
3. [Architecture](#3-architecture)
4. [Configuration System](#4-configuration-system)
5. [Core Modules](#5-core-modules)
6. [Data Pipeline](#6-data-pipeline)
7. [Inference Benchmark (Phase A)](#7-inference-benchmark-phase-a)
8. [Training Benchmark (Phase B)](#8-training-benchmark-phase-b)
9. [Quality Evaluation (Phase C)](#9-quality-evaluation-phase-c)
10. [Profiling & Monitoring](#10-profiling--monitoring)
11. [Analysis & Reporting](#11-analysis--reporting)
12. [CLI & Orchestration](#12-cli--orchestration)
13. [Containerization](#13-containerization)
14. [Dependencies](#14-dependencies)
15. [Testing Strategy](#15-testing-strategy)
16. [ADRs (Architecture Decision Records)](#16-adrs)

---

## 1. Overview

### 1.1 Purpose

A reproducible benchmarking framework that measures training throughput, inference latency/throughput, power efficiency, and OCR quality for sub-10B Vision-Language Models on NVIDIA B300 (Blackwell Ultra) vs H100 (Hopper) GPUs, focused on document-to-Markdown/HTML parsing tasks.

### 1.2 Scope

| In Scope | Out of Scope |
|----------|-------------|
| Single-GPU benchmarks (TP=1) | Multi-node distributed training |
| 6 OCR VLMs (0.9B–9B) | Models > 10B parameters |
| vLLM inference serving | TensorRT-LLM (future phase) |
| LoRA fine-tuning | Full-parameter training |
| H100 SXM + B300 SXM | PCIe variants, AMD GPUs |
| FlashAttention 3 & 4 | Custom CUDA kernel development |

### 1.3 Design Principles

- **Reproducibility First:** Every experiment is defined by a YAML config file. Results include full config snapshots, random seeds, and software versions.
- **GPU-Agnostic Core:** Business logic is GPU-unaware. Hardware-specific behavior is isolated in `hardware/` adapters and Docker images.
- **Composable Phases:** Phases A/B/C run independently or together. Each phase produces self-contained artifacts.
- **Fail-Safe Profiling:** GPU monitoring runs in a separate process; benchmark continues even if profiling fails.
- **Idempotent Runs:** Re-running with the same config + seed produces identical results (modulo GPU non-determinism, which is logged).

---

## 2. Repository Structure

```
vlm-ocr-bench/
├── README.md                          # Quick start, usage, results summary
├── LICENSE                            # Apache 2.0
├── pyproject.toml                     # Project metadata, dependencies, entry points
├── Makefile                           # Common commands (setup, lint, test, run)
│
├── configs/                           # All experiment configurations (YAML)
│   ├── defaults.yaml                  # Global defaults inherited by all configs
│   ├── models/                        # Per-model configurations
│   │   ├── paddleocr_vl_0.9b.yaml
│   │   ├── dots_ocr_1.5_3b.yaml
│   │   ├── nanonets_ocr2_3b.yaml
│   │   ├── deepseek_ocr_3b.yaml
│   │   ├── olmocr2_7b.yaml
│   │   └── chandra_ocr_9b.yaml
│   ├── hardware/                      # Per-GPU hardware profiles
│   │   ├── h100_sxm.yaml
│   │   └── b300_sxm.yaml
│   ├── phases/                        # Per-phase sweep definitions
│   │   ├── inference.yaml
│   │   ├── training.yaml
│   │   └── quality.yaml
│   └── experiments/                   # Composed experiment manifests
│       ├── full_benchmark.yaml        # All phases, all models, all GPUs
│       ├── quick_smoke.yaml           # 1 model, 1 batch size — CI validation
│       └── single_model_debug.yaml    # Debug config for development
│
├── src/                               # Main Python package: vlm_ocr_bench
│   └── vlm_ocr_bench/
│       ├── __init__.py
│       ├── cli.py                     # Typer CLI entry point
│       │
│       ├── config/                    # Configuration loading & validation
│       │   ├── __init__.py
│       │   ├── schema.py              # Pydantic models for all config types
│       │   ├── loader.py              # YAML loading with inheritance & overrides
│       │   └── resolvers.py           # Dynamic resolution (GPU detection, paths)
│       │
│       ├── hardware/                  # GPU abstraction layer
│       │   ├── __init__.py
│       │   ├── detector.py            # Auto-detect GPU type, VRAM, compute cap
│       │   ├── profiles.py            # Hardware profile dataclasses
│       │   ├── power.py               # Power reading (nvidia-smi / DCGM)
│       │   └── clock.py               # Clock management (lock, reset, query)
│       │
│       ├── models/                    # Model loading & management
│       │   ├── __init__.py
│       │   ├── registry.py            # Model registry (name → config mapping)
│       │   ├── loader.py              # HuggingFace model/tokenizer loading
│       │   ├── quantization.py        # FP8 / FP4 / NVFP4 quantization wrappers
│       │   └── adapters/              # Model-specific prompt/postprocess adapters
│       │       ├── __init__.py
│       │       ├── base.py            # Abstract adapter interface
│       │       ├── paddleocr_vl.py
│       │       ├── dots_ocr.py
│       │       ├── nanonets_ocr.py
│       │       ├── deepseek_ocr.py
│       │       ├── olmocr.py
│       │       └── chandra_ocr.py
│       │
│       ├── data/                      # Data loading & preprocessing
│       │   ├── __init__.py
│       │   ├── datasets.py            # Dataset registry & loaders
│       │   ├── omnidocbench.py        # OmniDocBench v1.5 loader & preprocessor
│       │   ├── olmocr_bench.py        # olmOCR-Bench loader
│       │   ├── real5_omnidoc.py       # Real5-OmniDocBench loader
│       │   ├── thai_docs.py           # Custom Thai document set loader
│       │   ├── docmatix.py            # Docmatix training data sampler
│       │   ├── image_utils.py         # Resize, normalize, format conversion
│       │   └── tokenization.py        # Prompt template construction per model
│       │
│       ├── inference/                 # Phase A: Inference benchmarking
│       │   ├── __init__.py
│       │   ├── engine.py              # vLLM engine wrapper (init, generate, shutdown)
│       │   ├── runner.py              # Benchmark loop: warmup → measure → collect
│       │   ├── workload.py            # Workload generator (batch, resolution, tokens)
│       │   ├── metrics.py             # InferenceMetrics dataclass & collectors
│       │   └── vllm_config.py         # vLLM-specific config builder per GPU
│       │
│       ├── training/                  # Phase B: Training benchmarking
│       │   ├── __init__.py
│       │   ├── lora.py                # LoRA configuration & PEFT integration
│       │   ├── trainer.py             # Training loop with metric hooks
│       │   ├── data_collator.py       # VLM-aware data collator (image + text)
│       │   ├── callbacks.py           # Throughput, memory, power logging callbacks
│       │   ├── metrics.py             # TrainingMetrics dataclass & collectors
│       │   └── deepspeed_config.py    # DeepSpeed ZeRO Stage 2 config builder
│       │
│       ├── evaluation/                # Phase C: Quality evaluation
│       │   ├── __init__.py
│       │   ├── runner.py              # Eval orchestrator: infer → parse → score
│       │   ├── parsers.py             # Output parsers (Markdown structure extraction)
│       │   ├── metrics/
│       │   │   ├── __init__.py
│       │   │   ├── edit_distance.py   # Normalized edit distance
│       │   │   ├── bleu.py            # BLEU score (with Thai segmentation support)
│       │   │   ├── meteor.py          # METEOR score
│       │   │   ├── cdm.py             # CDM (formula evaluation) wrapper
│       │   │   └── structural.py      # Table/formula structural equivalence
│       │   └── benchmarks/
│       │       ├── __init__.py
│       │       ├── omnidocbench.py     # OmniDocBench evaluation pipeline
│       │       ├── olmocr_bench.py     # olmOCR-Bench evaluation pipeline
│       │       └── real5.py            # Real5-OmniDocBench evaluation pipeline
│       │
│       ├── profiling/                 # GPU profiling & monitoring
│       │   ├── __init__.py
│       │   ├── monitor.py             # Background GPU monitor (async process)
│       │   ├── dcgm.py                # DCGM field collection
│       │   ├── nvidia_smi.py          # nvidia-smi polling fallback
│       │   ├── power.py               # Power measurement & energy integration
│       │   └── events.py              # Thermal throttle event detection
│       │
│       ├── analysis/                  # Post-experiment analysis
│       │   ├── __init__.py
│       │   ├── aggregator.py          # Result aggregation across runs
│       │   ├── statistics.py          # Welch's t-test, CI, Cohen's d
│       │   ├── roofline.py            # Roofline model computation & plotting
│       │   ├── pareto.py              # Pareto frontier (quality vs throughput)
│       │   ├── tco.py                 # TCO projection calculator
│       │   ├── plots.py               # Matplotlib/Plotly visualization generators
│       │   └── report.py              # Markdown report generator
│       │
│       └── utils/                     # Shared utilities
│           ├── __init__.py
│           ├── logging.py             # Structured JSON logging (structlog)
│           ├── timing.py              # Timer context manager, CUDA sync timing
│           ├── seeds.py               # Deterministic seed management
│           ├── checkpoints.py         # Experiment state save/resume
│           ├── storage.py             # Result file I/O (CSV, JSON, Parquet)
│           └── environment.py         # System info snapshot (versions, env vars)
│
├── scripts/                           # Shell scripts & utilities
│   ├── run_all.sh                     # Full experiment orchestrator
│   ├── run_phase.sh                   # Single-phase runner
│   ├── setup_env.sh                   # Environment setup (conda/pip)
│   ├── download_models.sh             # Pre-download all model weights
│   ├── download_data.sh               # Pre-download all benchmark datasets
│   ├── lock_clocks.sh                 # Lock GPU clocks for stable benchmarking
│   ├── reset_gpu.sh                   # GPU reset & persistence mode
│   └── export_results.sh             # Package results for sharing
│
├── docker/                            # Container definitions
│   ├── Dockerfile.base                # Common base (Ubuntu 24.04, CUDA 12.6, Python)
│   ├── Dockerfile.h100               # H100-specific: FA-3, cuDNN for Hopper
│   ├── Dockerfile.b300               # B300-specific: FA-4, cuDNN for Blackwell
│   ├── docker-compose.yaml            # Multi-service: bench + monitor + storage
│   └── .env.template                  # Environment variable template
│
├── monitoring/                        # Observability stack configs
│   ├── prometheus/
│   │   └── prometheus.yml             # Scrape config for DCGM exporter
│   ├── grafana/
│   │   ├── provisioning/
│   │   │   └── dashboards.yaml
│   │   └── dashboards/
│   │       └── gpu_benchmark.json     # Pre-built Grafana dashboard
│   └── dcgm/
│       └── dcgm-exporter-config.csv   # DCGM field selection
│
├── tests/                             # Test suite
│   ├── conftest.py                    # Shared fixtures (mock GPU, sample data)
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_metrics.py
│   │   ├── test_workload.py
│   │   ├── test_adapters.py
│   │   ├── test_statistics.py
│   │   └── test_tco.py
│   ├── integration/
│   │   ├── test_inference_engine.py   # Requires GPU
│   │   ├── test_training_loop.py      # Requires GPU
│   │   └── test_eval_pipeline.py      # Requires GPU + data
│   └── smoke/
│       └── test_smoke.py              # Quick end-to-end on smallest model
│
├── notebooks/                         # Exploration & presentation
│   ├── 01_explore_results.ipynb
│   ├── 02_roofline_analysis.ipynb
│   └── 03_pareto_frontier.ipynb
│
└── results/                           # Output directory (git-ignored, except README)
    ├── README.md                      # Results directory structure documentation
    ├── raw/                           # Raw per-run outputs
    │   ├── inference/
    │   ├── training/
    │   └── quality/
    ├── aggregated/                    # Aggregated cross-run results
    ├── plots/                         # Generated figures
    └── reports/                       # Final markdown/PDF reports
```

---

## 3. Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              CLI Layer                                   │
│   cli.py (Typer)  →  experiment manifest  →  phase dispatcher           │
├─────────────────────────────────────────────────────────────────────────┤
│                         Orchestration Layer                              │
│   inference/runner.py  │  training/trainer.py  │  evaluation/runner.py   │
│         Phase A        │       Phase B         │        Phase C          │
├─────────────────────────────────────────────────────────────────────────┤
│                          Core Services Layer                             │
│   config/  │  models/  │  data/  │  profiling/  │  analysis/  │ utils/ │
├─────────────────────────────────────────────────────────────────────────┤
│                        Hardware Abstraction Layer                        │
│   hardware/detector.py  │  hardware/profiles.py  │  hardware/power.py   │
├─────────────────────────────────────────────────────────────────────────┤
│                        External Dependencies                             │
│   vLLM  │  HuggingFace  │  PEFT  │  DeepSpeed  │  FlashAttention       │
│   DCGM  │  nvidia-smi   │  PyTorch             │  OmniDocBench eval    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow

```
                    ┌──────────────┐
                    │  YAML Config │
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │ Config Loader│──→ Validates via Pydantic schemas
                    └──────┬───────┘
                           ▼
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Phase A  │ │ Phase B  │ │ Phase C  │
        │Inference │ │Training  │ │ Quality  │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │            │            │
             ▼            ▼            ▼
        ┌──────────────────────────────────┐
        │       Profiling Monitor          │  (background process)
        │   DCGM / nvidia-smi / power      │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │      results/raw/{phase}/        │  JSON + CSV + Parquet
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │     Analysis & Aggregation       │
        │  statistics / roofline / pareto  │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │     results/reports/report.md    │  + plots/
        └──────────────────────────────────┘
```

### 3.3 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Inference engine | vLLM | Widest VLM support, native FA-3/4, FP8/FP4 KV cache, PagedAttention |
| Fine-tuning framework | HF Trainer + PEFT | LoRA support, DeepSpeed integration, callback system |
| Config format | YAML + Pydantic | Human-readable, composable inheritance, strict validation |
| Profiling | DCGM (primary) + nvidia-smi (fallback) | DCGM gives per-SM metrics at 100ms; nvidia-smi works everywhere |
| Result format | Parquet (primary) + CSV (portable) | Parquet for efficient analysis; CSV for compatibility |
| CLI framework | Typer | Type hints → auto CLI, minimal boilerplate |
| Logging | structlog (JSON) | Machine-parseable, structured, filterable |

---

## 4. Configuration System

### 4.1 Config Hierarchy & Inheritance

```
defaults.yaml          ← Global defaults (seeds, logging, output paths)
    ↓ merged with
hardware/{gpu}.yaml    ← GPU-specific (precision, FA version, clock settings)
    ↓ merged with
models/{model}.yaml    ← Model-specific (HF ID, prompts, resolution, tokens)
    ↓ merged with
phases/{phase}.yaml    ← Phase-specific (batch sizes, epochs, metrics)
    ↓ merged with
experiments/{exp}.yaml ← Experiment manifest (selects models × GPUs × phases)
    ↓ overridden by
CLI flags              ← Runtime overrides (--batch-size 4 --precision fp8)
```

Merge strategy: deep merge with later values overriding earlier ones. Lists are replaced, not appended.

### 4.2 Schema Definitions (`config/schema.py`)

```python
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional
from pathlib import Path


class PrecisionMode(str, Enum):
    BF16 = "bf16"
    FP16 = "fp16"
    FP8 = "fp8"
    FP4 = "fp4"
    NVFP4 = "nvfp4"


class FlashAttnVersion(str, Enum):
    FA2 = "flash_attn_2"
    FA3 = "flash_attn_3"
    FA4 = "flash_attn_4"
    AUTO = "auto"          # detect from GPU


class OutputFormat(str, Enum):
    MARKDOWN = "markdown"
    HTML = "html"


class GPUType(str, Enum):
    H100_SXM = "h100_sxm"
    B300_SXM = "b300_sxm"


# ─── Hardware Config ───

class GPUConfig(BaseModel):
    gpu_type: GPUType
    precision_modes: list[PrecisionMode]
    flash_attn: FlashAttnVersion = FlashAttnVersion.AUTO
    kv_cache_dtype: PrecisionMode = PrecisionMode.FP8
    tensor_parallel: int = Field(default=1, ge=1, le=8)
    lock_clocks: bool = True
    max_clock_mhz: Optional[int] = None     # None = max stable
    power_limit_watts: Optional[int] = None  # None = default TDP
    cuda_version: str = "12.6"


# ─── Model Config ───

class ModelConfig(BaseModel):
    name: str                                # e.g. "paddleocr_vl_0.9b"
    hf_model_id: str                         # e.g. "PaddlePaddle/PaddleOCR-VL"
    params_billion: float                    # e.g. 0.9
    tier: str                                # "ultra_compact" | "compact" | "midsize"
    adapter: str                             # adapter class name in models/adapters/
    supported_resolutions: list[int]         # [1024, 1536, 2048]
    max_output_tokens: int = 4096
    default_prompt_key: str = "document_parse_md"
    supports_flash_attn: list[FlashAttnVersion]
    vision_token_estimate: str = "dynamic"   # or integer for fixed
    requires_padding: bool = False


# ─── Inference Phase Config ───

class InferenceConfig(BaseModel):
    batch_sizes: list[int] = [1, 2, 4, 8, 16, 32]
    resolutions: list[int] = [1024, 1536, 2048]
    output_formats: list[OutputFormat] = [OutputFormat.MARKDOWN]
    max_output_tokens_sweep: list[int] = [2048, 4096, 8192]
    precision_modes: list[PrecisionMode] = [PrecisionMode.BF16, PrecisionMode.FP8]
    warmup_requests: int = 50
    measurement_requests: int = 200
    runs_per_config: int = 3
    cooldown_seconds: int = 60
    vllm_args: dict = {}                     # extra vLLM engine args


# ─── Training Phase Config ───

class LoRAConfig(BaseModel):
    rank: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: list[str] = ["q_proj", "v_proj", "k_proj", "o_proj"]
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class TrainingConfig(BaseModel):
    lora: LoRAConfig = LoRAConfig()
    optimizer: str = "adamw_torch"
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    epochs: int = 3
    effective_batch_size: int = 32
    micro_batch_sizes: list[int] = [1, 2, 4, 8, 16]
    precision_modes: list[PrecisionMode] = [PrecisionMode.BF16, PrecisionMode.FP8]
    gradient_checkpointing: bool = True      # auto-enable for 7B+
    deepspeed_stage: Optional[int] = None    # 2 for 7B+
    max_image_resolution: int = 2048
    dataset: str = "docmatix_50k"
    domain_dataset: Optional[str] = "thai_fin_legal_5k"
    warmup_steps: int = 500
    measurement_steps: int = 2000
    runs_per_config: int = 3
    early_stop_patience: int = 3


# ─── Quality Evaluation Config ───

class QualityConfig(BaseModel):
    benchmarks: list[str] = [
        "omnidocbench_v1.5",
        "olmocr_bench",
        "real5_omnidocbench",
        "thai_docs_500",
    ]
    metrics: list[str] = [
        "edit_distance",
        "bleu",
        "meteor",
        "cdm",
        "structural_table",
        "structural_formula",
    ]
    precision_modes: list[PrecisionMode] = [
        PrecisionMode.BF16,
        PrecisionMode.FP8,
        PrecisionMode.FP4,
    ]
    per_document_type: bool = True           # breakdown by doc type
    max_samples: Optional[int] = None        # None = full benchmark


# ─── Profiling Config ───

class ProfilingConfig(BaseModel):
    enabled: bool = True
    backend: str = "dcgm"                    # "dcgm" | "nvidia_smi"
    sample_interval_ms: int = 100
    fields: list[str] = [
        "gpu_temp", "gpu_power", "sm_active", "sm_occupancy",
        "tensor_active", "dram_active", "pcie_tx_bytes", "pcie_rx_bytes",
        "gpu_mem_used", "gpu_mem_total", "throttle_reasons",
    ]
    export_format: str = "parquet"


# ─── Top-Level Experiment Config ───

class ExperimentConfig(BaseModel):
    name: str                                # experiment identifier
    description: str = ""
    seed: int = 42
    output_dir: Path = Path("results")
    models: list[str]                        # model config names to run
    gpus: list[GPUType]                      # GPU types to benchmark
    phases: list[str] = ["inference", "training", "quality"]
    inference: InferenceConfig = InferenceConfig()
    training: TrainingConfig = TrainingConfig()
    quality: QualityConfig = QualityConfig()
    profiling: ProfilingConfig = ProfilingConfig()
    gpu_configs: dict[str, GPUConfig] = {}   # overrides per GPU
    model_configs: dict[str, ModelConfig] = {}
```

### 4.3 Example Config Files

**`configs/defaults.yaml`**
```yaml
seed: 42
output_dir: results
profiling:
  enabled: true
  backend: dcgm
  sample_interval_ms: 100
```

**`configs/hardware/h100_sxm.yaml`**
```yaml
gpu_type: h100_sxm
precision_modes: [bf16, fp8]
flash_attn: flash_attn_3
kv_cache_dtype: fp8
lock_clocks: true
cuda_version: "12.6"
```

**`configs/hardware/b300_sxm.yaml`**
```yaml
gpu_type: b300_sxm
precision_modes: [bf16, fp8, fp4, nvfp4]
flash_attn: flash_attn_4
kv_cache_dtype: fp8   # sweep fp4 in experiments
lock_clocks: true
cuda_version: "12.6"
```

**`configs/models/dots_ocr_1.5_3b.yaml`**
```yaml
name: dots_ocr_1.5_3b
hf_model_id: rednote-hilab/dots.ocr-1.5
params_billion: 3.0
tier: compact
adapter: DotsOcrAdapter
supported_resolutions: [1024, 1536, 2048]
max_output_tokens: 8192
default_prompt_key: document_parse_md
supports_flash_attn: [flash_attn_2, flash_attn_3, flash_attn_4]
vision_token_estimate: dynamic
```

**`configs/experiments/full_benchmark.yaml`**
```yaml
name: full_b300_vs_h100
description: "Complete B300 vs H100 benchmark — all models, all phases"
models:
  - paddleocr_vl_0.9b
  - dots_ocr_1.5_3b
  - nanonets_ocr2_3b
  - deepseek_ocr_3b
  - olmocr2_7b
  - chandra_ocr_9b
gpus:
  - h100_sxm
  - b300_sxm
phases:
  - inference
  - training
  - quality

inference:
  batch_sizes: [1, 2, 4, 8, 16, 32]
  resolutions: [1024, 1536, 2048]
  max_output_tokens_sweep: [2048, 4096, 8192]
  warmup_requests: 50
  measurement_requests: 200
  runs_per_config: 3
  cooldown_seconds: 60

training:
  epochs: 3
  micro_batch_sizes: [1, 2, 4, 8, 16]
  dataset: docmatix_50k
  domain_dataset: thai_fin_legal_5k
  runs_per_config: 3
```

---

## 5. Core Modules

### 5.1 Hardware Abstraction (`hardware/`)

#### `detector.py`

```python
@dataclass
class GPUInfo:
    name: str               # "NVIDIA H100 SXM" | "NVIDIA B300 SXM"
    gpu_type: GPUType       # enum
    compute_capability: tuple[int, int]  # (9, 0) | (10, 0)
    vram_gb: float          # 80.0 | 288.0
    driver_version: str
    cuda_version: str
    flash_attn_version: str # "3.x.x" | "4.x.x"
    vllm_version: str
    pytorch_version: str
    num_sms: int
    clock_mhz: int          # current
    power_limit_w: int

def detect_gpu() -> GPUInfo:
    """Auto-detect GPU type and capabilities via pynvml."""

def validate_gpu_for_config(gpu: GPUInfo, config: GPUConfig) -> list[str]:
    """Return list of warnings/errors if GPU doesn't match config."""
```

#### `power.py`

```python
class PowerReader:
    """Read GPU power consumption. Uses DCGM if available, nvidia-smi fallback."""

    def __init__(self, backend: str = "dcgm", interval_ms: int = 100):
        ...

    def start(self) -> None:
        """Begin background power sampling."""

    def stop(self) -> PowerSummary:
        """Stop sampling, return summary statistics."""

    def get_current_watts(self) -> float:
        """Instantaneous power reading."""

@dataclass
class PowerSummary:
    mean_watts: float
    peak_watts: float
    min_watts: float
    total_energy_joules: float
    total_energy_wh: float
    duration_seconds: float
    samples: int
    throttle_events: int
```

#### `clock.py`

```python
def lock_gpu_clocks(gpu_id: int = 0, max_mhz: Optional[int] = None) -> None:
    """Lock GPU SM and memory clocks for deterministic benchmarking."""

def reset_gpu_clocks(gpu_id: int = 0) -> None:
    """Reset clocks to default."""

def get_clock_info(gpu_id: int = 0) -> dict:
    """Return current SM clock, memory clock, throttle reasons."""
```

### 5.2 Model Adapters (`models/adapters/`)

Each OCR VLM has a different prompt format, image preprocessing, and output parsing. Adapters normalize these differences.

#### `base.py` — Abstract Interface

```python
from abc import ABC, abstractmethod
from PIL import Image


class OCRModelAdapter(ABC):
    """Adapter interface normalizing prompt/postprocess across OCR VLMs."""

    @abstractmethod
    def get_model_id(self) -> str:
        """HuggingFace model ID."""

    @abstractmethod
    def build_prompt(
        self,
        image: Image.Image,
        output_format: OutputFormat = OutputFormat.MARKDOWN,
        task: str = "full_page_parse",
    ) -> dict:
        """
        Build model-specific prompt.

        Returns:
            {
                "messages": [...],  # for chat-style models
                "images": [...],    # PIL images
                "prompt": str,      # for completion-style models
            }
        """

    @abstractmethod
    def parse_output(self, raw_output: str) -> ParsedDocument:
        """
        Parse raw model output into structured ParsedDocument.

        Handles model-specific output quirks (e.g., dots.ocr uses
        special tokens for layout regions).
        """

    @abstractmethod
    def get_vllm_kwargs(self, gpu_config: GPUConfig) -> dict:
        """
        Return model-specific vLLM engine kwargs.

        E.g., trust_remote_code, tokenizer settings, vision config.
        """

    def get_supported_resolutions(self) -> list[int]:
        """Resolutions this model handles well."""
        return [1024, 1536, 2048]

    def get_max_output_tokens(self) -> int:
        """Default max tokens for full-page parsing."""
        return 4096

    def estimate_vision_tokens(self, width: int, height: int) -> int:
        """Estimate number of vision tokens for given image dimensions."""
        # Default: assume patch-based encoding
        patch_size = 14
        return (width // patch_size) * (height // patch_size)


@dataclass
class ParsedDocument:
    """Normalized document parse output."""
    raw_text: str              # full raw output
    markdown: str              # cleaned Markdown
    html: Optional[str]        # cleaned HTML (if requested)
    elements: list[DocElement] # structured elements
    parse_time_ms: float
    token_count: int


@dataclass
class DocElement:
    """A detected document element."""
    type: str       # "text" | "table" | "formula" | "title" | "figure" | ...
    content: str    # extracted content (Markdown for tables, LaTeX for formulas)
    bbox: Optional[tuple[float, float, float, float]]  # normalized x1,y1,x2,y2
    confidence: Optional[float]
```

#### Example: `dots_ocr.py`

```python
class DotsOcrAdapter(OCRModelAdapter):
    """Adapter for dots.ocr-1.5 (3B)."""

    def get_model_id(self) -> str:
        return "rednote-hilab/dots.ocr-1.5"

    def build_prompt(self, image, output_format, task):
        # dots.ocr uses specific prompt keys
        prompt_map = {
            "full_page_parse": "<|im_start|>user\n<image>\nParse this document to markdown.<|im_end|>\n<|im_start|>assistant\n",
            "layout_only": "<|im_start|>user\n<image>\nprompt_layout_only_en<|im_end|>\n<|im_start|>assistant\n",
        }
        return {
            "prompt": prompt_map.get(task, prompt_map["full_page_parse"]),
            "images": [image],
        }

    def parse_output(self, raw_output: str) -> ParsedDocument:
        # dots.ocr outputs Markdown directly; strip special tokens
        cleaned = self._strip_special_tokens(raw_output)
        elements = self._extract_elements(cleaned)
        return ParsedDocument(
            raw_text=raw_output,
            markdown=cleaned,
            html=None,
            elements=elements,
            parse_time_ms=0,  # filled by caller
            token_count=len(raw_output.split()),
        )

    def get_vllm_kwargs(self, gpu_config):
        return {
            "trust_remote_code": True,
            "dtype": "bfloat16" if gpu_config.kv_cache_dtype == PrecisionMode.BF16 else "auto",
            "max_model_len": 16384,
        }
```

### 5.3 Data Module (`data/`)

#### `datasets.py` — Dataset Registry

```python
DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "omnidocbench_v1.5": DatasetSpec(
        name="omnidocbench_v1.5",
        hf_id="opendatalab/OmniDocBench",
        version="v1.5",
        split="test",
        num_samples=1355,
        doc_types=["academic", "textbook", "slide", "financial", "newspaper",
                    "exam", "note", "magazine", "colorful_textbook"],
        loader_cls="OmniDocBenchLoader",
    ),
    "olmocr_bench": DatasetSpec(
        name="olmocr_bench",
        source="github:allenai/olmocr",
        loader_cls="OlmOCRBenchLoader",
    ),
    "real5_omnidocbench": DatasetSpec(
        name="real5_omnidocbench",
        hf_id="PaddlePaddle/Real5-OmniDocBench",
        scenarios=["scanning", "warping", "screen_photo", "illumination", "skew"],
        num_samples=6775,  # 1355 × 5
        loader_cls="Real5OmniDocLoader",
    ),
    "thai_docs_500": DatasetSpec(
        name="thai_docs_500",
        source="local",
        path="data/thai_fin_legal/",
        num_samples=500,
        loader_cls="ThaiDocsLoader",
    ),
    "docmatix_50k": DatasetSpec(
        name="docmatix_50k",
        hf_id="HuggingFaceM4/Docmatix",
        split="train",
        sample_size=50000,
        stratify_by="doc_type",
        loader_cls="DocmatixLoader",
    ),
}
```

#### `omnidocbench.py` — OmniDocBench Loader

```python
class OmniDocBenchLoader:
    """Load and preprocess OmniDocBench v1.5 for evaluation."""

    def __init__(self, config: DatasetSpec):
        self.config = config
        self._data: Optional[list[DocSample]] = None

    def load(self) -> list[DocSample]:
        """Download from HuggingFace and parse annotation JSON."""

    def get_samples(
        self,
        doc_types: Optional[list[str]] = None,
        max_samples: Optional[int] = None,
    ) -> list[DocSample]:
        """Filter samples by document type."""

    def get_ground_truth(self, sample_id: str) -> GroundTruth:
        """Return structured ground truth for a sample."""


@dataclass
class DocSample:
    sample_id: str
    image: Image.Image        # PIL Image (loaded lazily)
    image_path: Path
    doc_type: str             # e.g., "academic", "financial"
    language: str             # "en", "zh", "mixed"
    layout_type: str          # "single_column", "double_column", etc.
    has_tables: bool
    has_formulas: bool
    has_figures: bool
    attributes: dict          # page-level attributes (blurred, watermark, etc.)


@dataclass
class GroundTruth:
    markdown: str             # full-page ground truth Markdown
    elements: list[GTElement] # element-level annotations
    reading_order: list[int]  # ordered element indices
    bboxes: list[BBox]        # layout bounding boxes


@dataclass
class GTElement:
    element_id: int
    type: str                 # "title", "text", "table", "formula", etc.
    content: str              # text / LaTeX / HTML
    bbox: BBox
    attributes: dict          # element-level attributes
```

---

## 6. Data Pipeline

### 6.1 Image Preprocessing

```python
# data/image_utils.py

def preprocess_image(
    image: Image.Image,
    target_resolution: int,
    model_name: str,
) -> Image.Image:
    """
    Resize document image to target resolution while preserving aspect ratio.

    Strategy:
    - Resize longest side to target_resolution
    - Pad to square if model requires (flag in ModelConfig)
    - Maintain original aspect ratio (critical for document layout fidelity)
    - Convert to RGB (drop alpha channel)
    - No augmentation for benchmarking (deterministic)
    """

def get_image_stats(image: Image.Image) -> dict:
    """Return width, height, channels, file_size_bytes, DPI estimate."""
```

### 6.2 Prompt Construction

```python
# data/tokenization.py

PROMPT_TEMPLATES = {
    "document_parse_md": "Convert this document image to Markdown format. "
                         "Preserve the reading order, table structure, "
                         "and mathematical formulas (in LaTeX).",
    "document_parse_html": "Convert this document image to HTML format. "
                           "Use <table> for tables and inline LaTeX for formulas.",
    "ocr_text_only": "Extract all text from this document image, preserving "
                     "reading order. Output as plain text.",
}

def build_inference_input(
    adapter: OCRModelAdapter,
    image: Image.Image,
    output_format: OutputFormat,
    resolution: int,
) -> dict:
    """
    Build complete inference input for a model.

    Returns dict compatible with vLLM's generate() interface:
    {
        "prompt": str | list[dict],
        "multi_modal_data": {"image": Image},
    }
    """
```

---

## 7. Inference Benchmark (Phase A)

### 7.1 Engine Wrapper (`inference/engine.py`)

```python
class VLLMEngine:
    """Wrapper around vLLM's LLM/AsyncLLMEngine for benchmarking."""

    def __init__(
        self,
        model_config: ModelConfig,
        gpu_config: GPUConfig,
        inference_config: InferenceConfig,
    ):
        self.model_config = model_config
        self.gpu_config = gpu_config
        self.adapter = get_adapter(model_config.adapter)
        self._engine: Optional[LLM] = None

    def initialize(self) -> EngineInfo:
        """
        Load model into vLLM engine with GPU-specific settings.

        Returns EngineInfo with:
        - model_load_time_s
        - gpu_memory_allocated_gb
        - flash_attn_version_used
        - actual_precision
        """

    def generate_batch(
        self,
        inputs: list[dict],     # from build_inference_input()
        sampling_params: SamplingParams,
    ) -> list[GenerationResult]:
        """
        Run batch inference, return results with timing.

        Each GenerationResult contains:
        - output_text: str
        - num_tokens: int
        - ttft_ms: float         (time to first token)
        - generation_time_ms: float
        - total_time_ms: float
        """

    def shutdown(self) -> None:
        """Release GPU memory, destroy engine."""

    def get_memory_stats(self) -> MemoryStats:
        """Current GPU memory allocation breakdown."""


@dataclass
class EngineInfo:
    model_load_time_s: float
    gpu_memory_allocated_gb: float
    gpu_memory_reserved_gb: float
    flash_attn_version: str
    actual_precision: str
    kv_cache_dtype: str
    max_batch_size: int
    max_model_len: int


@dataclass
class GenerationResult:
    output_text: str
    num_output_tokens: int
    num_input_tokens: int         # prompt + vision tokens
    ttft_ms: float
    generation_time_ms: float
    total_time_ms: float
    tokens_per_second: float
```

### 7.2 Benchmark Runner (`inference/runner.py`)

```python
class InferenceBenchmarkRunner:
    """Orchestrates Phase A: inference benchmarking."""

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        model_config: ModelConfig,
        gpu_config: GPUConfig,
    ):
        ...

    def run(self) -> InferenceBenchmarkResult:
        """
        Execute full inference benchmark.

        Algorithm:
        1. Initialize vLLM engine
        2. For each (resolution, batch_size, precision, max_tokens):
            a. Generate workload batch
            b. Warm-up phase (discard results)
            c. For each run in runs_per_config:
                i.   Start profiling monitor
                ii.  Execute measurement_requests
                iii. Stop profiling, collect GPU metrics
                iv.  Compute inference metrics
            d. Cooldown
        3. Aggregate results across runs
        4. Save raw + aggregated results
        """

    def _run_single_config(
        self,
        resolution: int,
        batch_size: int,
        precision: PrecisionMode,
        max_tokens: int,
        run_id: int,
    ) -> SingleConfigResult:
        """Execute one configuration and return metrics."""

    def _warmup(self, engine: VLLMEngine, n: int) -> None:
        """Run n warmup requests, discarding results."""


@dataclass
class InferenceBenchmarkResult:
    model_name: str
    gpu_type: GPUType
    engine_info: EngineInfo
    configs: list[SingleConfigResult]
    environment: EnvironmentSnapshot
    total_wall_time_s: float


@dataclass
class SingleConfigResult:
    resolution: int
    batch_size: int
    precision: PrecisionMode
    max_output_tokens: int
    run_id: int
    # Throughput
    pages_per_second: float
    tokens_per_second: float
    # Latency (ms)
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float
    e2e_latency_p50: float
    e2e_latency_p95: float
    e2e_latency_p99: float
    # Memory
    peak_gpu_memory_gb: float
    kv_cache_memory_gb: float
    # Power
    mean_power_watts: float
    peak_power_watts: float
    energy_per_page_wh: float
    pages_per_watt_hour: float
    tokens_per_watt_hour: float
    # GPU utilization
    mean_sm_occupancy: float
    mean_tensor_util: float
    mean_memory_util: float
    # Profiling
    profiling_data: Optional[Path]   # path to Parquet file
    throttle_events: int
    # Raw data
    per_request_latencies: list[float]
    num_requests: int
    wall_time_s: float
```

### 7.3 vLLM Config Builder (`inference/vllm_config.py`)

```python
def build_vllm_engine_args(
    model_config: ModelConfig,
    gpu_config: GPUConfig,
    precision: PrecisionMode,
    adapter: OCRModelAdapter,
) -> dict:
    """
    Build vLLM EngineArgs dict tailored to GPU + model + precision.

    Key decisions:
    - H100: enforce_eager=False, enable FA-3 FP8
    - B300: enforce_eager=False, enable FA-4, NVFP4 KV cache if FP4
    - Quantization: auto-detect AWQ/GPTQ/FP8 based on precision mode
    - Memory: gpu_memory_utilization=0.90 (leave room for profiling)
    """

    base_args = {
        "model": model_config.hf_model_id,
        "tensor_parallel_size": gpu_config.tensor_parallel,
        "gpu_memory_utilization": 0.90,
        "max_model_len": adapter.get_max_output_tokens() + 2048,
        "trust_remote_code": True,
        **adapter.get_vllm_kwargs(gpu_config),
    }

    # Precision-specific
    if precision == PrecisionMode.FP8:
        base_args["quantization"] = "fp8"
        base_args["kv_cache_dtype"] = "fp8"
    elif precision in (PrecisionMode.FP4, PrecisionMode.NVFP4):
        if gpu_config.gpu_type != GPUType.B300_SXM:
            raise ValueError("FP4/NVFP4 requires B300")
        base_args["quantization"] = "fp4"
        base_args["kv_cache_dtype"] = "fp4"
    else:
        base_args["dtype"] = "bfloat16"

    return base_args
```

---

## 8. Training Benchmark (Phase B)

### 8.1 Trainer (`training/trainer.py`)

```python
class TrainingBenchmarkRunner:
    """Orchestrates Phase B: LoRA fine-tuning benchmarking."""

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        model_config: ModelConfig,
        gpu_config: GPUConfig,
    ):
        ...

    def run(self) -> TrainingBenchmarkResult:
        """
        Execute full training benchmark.

        Algorithm:
        1. Load base model with PEFT/LoRA config
        2. Load training dataset (Docmatix 50K or Thai domain set)
        3. For each (micro_batch_size, precision):
            a. Configure Trainer with callbacks
            b. Warm-up steps (discard)
            c. For each run in runs_per_config:
                i.   Reset model to initial LoRA weights
                ii.  Start profiling monitor
                iii. Train for measurement_steps
                iv.  Record throughput, loss curve, power
                v.   Stop profiling
            d. Determine max feasible micro_batch (OOM boundary)
        4. Full convergence run: best config × 3 epochs
        5. Save results
        """

    def _find_max_batch_size(
        self,
        model,
        precision: PrecisionMode,
    ) -> int:
        """Binary search for max micro_batch_size before OOM."""

    def _run_throughput_measurement(
        self,
        model,
        dataloader,
        micro_batch_size: int,
        precision: PrecisionMode,
        steps: int,
    ) -> ThroughputResult:
        """Run N steps, measure samples/s and tokens/s."""


@dataclass
class TrainingBenchmarkResult:
    model_name: str
    gpu_type: GPUType
    configs: list[TrainingConfigResult]
    convergence_run: Optional[ConvergenceResult]
    environment: EnvironmentSnapshot


@dataclass
class TrainingConfigResult:
    micro_batch_size: int
    effective_batch_size: int
    gradient_accumulation_steps: int
    precision: PrecisionMode
    run_id: int
    # Throughput
    samples_per_second: float
    tokens_per_second: float
    # Memory
    peak_gpu_memory_gb: float
    model_memory_gb: float
    optimizer_memory_gb: float
    activation_memory_gb: float
    # Power
    mean_power_watts: float
    energy_per_1k_samples_wh: float
    # Cost
    estimated_cost_per_epoch: float    # based on cloud pricing
    estimated_time_per_epoch_hours: float
    # Stability
    loss_values: list[float]           # per-step loss
    gradient_norm_values: list[float]
    oom_occurred: bool
    steps_completed: int


@dataclass
class ConvergenceResult:
    """Full training run to measure time-to-convergence."""
    precision: PrecisionMode
    micro_batch_size: int
    epochs_completed: int
    final_train_loss: float
    final_val_loss: float
    best_val_loss: float
    best_val_epoch: int
    total_training_time_hours: float
    total_energy_kwh: float
    total_cost_estimate: float
    loss_curve: list[tuple[int, float]]   # (step, loss)
    val_metrics: list[tuple[int, dict]]   # (step, {metric: value})
```

### 8.2 LoRA Configuration (`training/lora.py`)

```python
def build_peft_config(
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> PeftConfig:
    """
    Build LoRA config for PEFT library.

    Handles model-specific target module detection:
    - Qwen2-VL based (OlmOCR, Chandra): q_proj, k_proj, v_proj, o_proj
    - ERNIE based (PaddleOCR-VL): attention layers auto-detected
    - Custom VLMs: fallback to all Linear layers
    """

def get_trainable_param_summary(model) -> dict:
    """
    Return:
    {
        "total_params": int,
        "trainable_params": int,
        "trainable_pct": float,
        "lora_params_by_module": dict[str, int],
    }
    """
```

### 8.3 Data Collator (`training/data_collator.py`)

```python
class VLMDataCollator:
    """
    Data collator for VLM fine-tuning that handles:
    - Variable-size images (padded to batch max)
    - Image + text interleaved sequences
    - Dynamic vision token counts
    - Label masking (only supervise text output, not vision tokens)
    """

    def __init__(
        self,
        tokenizer,
        adapter: OCRModelAdapter,
        max_image_resolution: int = 2048,
    ):
        ...

    def __call__(self, features: list[dict]) -> dict:
        """
        Collate a batch of (image, text) pairs.

        Returns dict compatible with HF Trainer:
        {
            "input_ids": Tensor,
            "attention_mask": Tensor,
            "labels": Tensor,          # -100 for vision tokens
            "pixel_values": Tensor,    # batched images
            "image_sizes": Tensor,     # original sizes for dynamic res
        }
        """
```

---

## 9. Quality Evaluation (Phase C)

### 9.1 Evaluation Runner (`evaluation/runner.py`)

```python
class QualityEvalRunner:
    """Orchestrates Phase C: quality evaluation across benchmarks."""

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        model_config: ModelConfig,
        gpu_config: GPUConfig,
    ):
        ...

    def run(self) -> QualityEvalResult:
        """
        Execute quality evaluation.

        Algorithm:
        1. For each benchmark in quality_config.benchmarks:
            a. Load dataset
            b. For each precision in quality_config.precision_modes:
                i.   Initialize engine at this precision
                ii.  Run inference on all benchmark samples
                iii. Parse outputs via adapter
                iv.  Compute all quality metrics
                v.   Breakdown by document type (if per_document_type)
            c. Compare across precisions (degradation analysis)
        2. Generate quality comparison tables
        """


@dataclass
class QualityEvalResult:
    model_name: str
    gpu_type: GPUType
    benchmark_results: dict[str, BenchmarkQuality]
    precision_comparison: PrecisionComparison


@dataclass
class BenchmarkQuality:
    benchmark_name: str
    precision: PrecisionMode
    overall_scores: dict[str, float]         # metric_name → score
    per_doc_type_scores: dict[str, dict[str, float]]  # doc_type → {metric → score}
    per_element_scores: dict[str, dict[str, float]]   # element_type → {metric → score}
    num_samples: int
    num_failed: int                          # samples where model output was empty/invalid


@dataclass
class PrecisionComparison:
    """Compare quality across precision modes for same model+GPU."""
    baseline_precision: PrecisionMode        # typically BF16
    comparisons: list[PrecisionDelta]


@dataclass
class PrecisionDelta:
    precision: PrecisionMode
    benchmark: str
    metric: str
    baseline_score: float
    this_score: float
    absolute_delta: float
    relative_delta_pct: float
```

### 9.2 Metrics (`evaluation/metrics/`)

#### `edit_distance.py`

```python
def normalized_edit_distance(prediction: str, reference: str) -> float:
    """
    Compute normalized Levenshtein edit distance.

    Returns 1.0 - (edit_distance / max(len(pred), len(ref)))
    so that 1.0 = perfect match, 0.0 = completely wrong.

    Handles unicode normalization (NFKC) for fair comparison.
    """

def block_level_edit_distance(
    pred_blocks: list[str],
    ref_blocks: list[str],
    match_strategy: str = "hungarian",
) -> float:
    """
    OmniDocBench-style block-level evaluation.

    Uses Hungarian matching to align predicted blocks
    to reference blocks, then averages edit distance.
    """
```

#### `bleu.py`

```python
def compute_bleu(
    prediction: str,
    reference: str,
    language: str = "en",
    n_gram: int = 4,
) -> float:
    """
    BLEU score with language-aware tokenization.

    For Thai: uses pythainlp word segmentation.
    For Chinese: character-level tokenization.
    For English/others: nltk word_tokenize.
    """
```

#### `structural.py`

```python
def table_structural_accuracy(
    pred_table: str,
    ref_table: str,
    format: str = "markdown",
) -> dict:
    """
    Evaluate table structure preservation.

    Returns:
    {
        "row_count_match": bool,
        "col_count_match": bool,
        "cell_content_accuracy": float,  # avg edit distance per cell
        "structure_f1": float,           # F1 of correct cell positions
    }
    """

def formula_structural_accuracy(
    pred_latex: str,
    ref_latex: str,
) -> dict:
    """
    Evaluate LaTeX formula accuracy.

    Returns:
    {
        "exact_match": bool,
        "normalized_edit_distance": float,
        "cdm_score": Optional[float],    # if CDM evaluator available
        "token_level_f1": float,
    }
    """
```

---

## 10. Profiling & Monitoring

### 10.1 Background Monitor (`profiling/monitor.py`)

```python
class GPUMonitor:
    """
    Background process that continuously samples GPU metrics.

    Runs in a separate multiprocessing.Process to avoid
    interfering with benchmark timing. Communicates via
    shared memory events (start, stop, checkpoint).
    """

    def __init__(self, config: ProfilingConfig, gpu_id: int = 0):
        self._process: Optional[Process] = None
        self._start_event = Event()
        self._stop_event = Event()
        self._output_path: Optional[Path] = None

    def start(self, output_path: Path) -> None:
        """Launch background monitoring process."""

    def stop(self) -> ProfilingResult:
        """
        Stop monitoring, wait for process, return results.

        Returns ProfilingResult with path to Parquet file
        and computed summary statistics.
        """

    def checkpoint(self, label: str) -> None:
        """Insert a labeled timestamp in the profiling data."""


@dataclass
class ProfilingResult:
    data_path: Path              # Parquet file with time-series data
    summary: ProfilingSummary
    duration_s: float
    num_samples: int


@dataclass
class ProfilingSummary:
    gpu_temp_mean: float
    gpu_temp_max: float
    power_mean_w: float
    power_max_w: float
    power_total_wh: float
    sm_occupancy_mean: float
    tensor_util_mean: float
    memory_util_mean: float
    memory_peak_gb: float
    throttle_events: int
    throttle_reasons: list[str]
```

### 10.2 Grafana Dashboard

The pre-built dashboard (`monitoring/grafana/dashboards/gpu_benchmark.json`) displays:

| Panel | Metrics | Visualization |
|-------|---------|--------------|
| GPU Temperature | `DCGM_FI_DEV_GPU_TEMP` | Time series with throttle threshold line |
| Power Draw | `DCGM_FI_DEV_POWER_USAGE` | Time series (W) + cumulative energy (Wh) |
| SM Occupancy | `DCGM_FI_PROF_SM_OCCUPANCY` | Time series (%) |
| Tensor Core Utilization | `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | Time series (%) |
| Memory Usage | `DCGM_FI_DEV_FB_USED`, `_FREE` | Stacked area (GB) |
| PCIe Throughput | `DCGM_FI_PROF_PCIE_TX_BYTES`, `_RX_BYTES` | Time series (GB/s) |
| Benchmark Progress | Custom annotations | Event markers (warmup, measure, cooldown) |

---

## 11. Analysis & Reporting

### 11.1 Statistical Analysis (`analysis/statistics.py`)

```python
def welch_t_test(
    group_a: list[float],
    group_b: list[float],
) -> StatTestResult:
    """
    Welch's t-test for independent samples with unequal variance.

    Returns:
    {
        "t_statistic": float,
        "p_value": float,
        "significant": bool,    # p < 0.05
        "cohens_d": float,      # effect size
        "ci_95_diff": (float, float),  # 95% CI of mean difference
    }
    """

def bootstrap_ci(
    data: list[float],
    statistic: Callable = np.mean,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap confidence interval."""

def compute_speedup(
    h100_values: list[float],
    b300_values: list[float],
) -> SpeedupResult:
    """
    Compute speedup ratio with confidence intervals.

    Returns:
    {
        "mean_speedup": float,
        "ci_95": (float, float),
        "p_value": float,
        "effect_size": float,
    }
    """
```

### 11.2 Roofline Model (`analysis/roofline.py`)

```python
def compute_roofline(
    gpu_info: GPUInfo,
    model_config: ModelConfig,
    precision: PrecisionMode,
    measured_tflops: float,
    measured_bandwidth_gbps: float,
) -> RooflinePoint:
    """
    Compute operational intensity and position on roofline.

    Returns:
    {
        "arithmetic_intensity": float,   # FLOP/byte
        "peak_compute_tflops": float,    # GPU peak for this precision
        "peak_bandwidth_tbps": float,    # GPU peak memory bandwidth
        "ridge_point": float,            # compute/bandwidth threshold
        "measured_tflops": float,
        "measured_bandwidth_tbps": float,
        "bottleneck": str,               # "compute" | "memory" | "balanced"
        "utilization_pct": float,        # vs theoretical peak
    }
    """

def plot_roofline(
    gpu_info: GPUInfo,
    points: list[RooflinePoint],
    output_path: Path,
) -> None:
    """Generate roofline plot with all model points annotated."""
```

### 11.3 TCO Calculator (`analysis/tco.py`)

```python
@dataclass
class TCOParams:
    gpu_rental_rate_per_hour: float
    power_cost_per_kwh: float = 0.10
    pages_per_day: int = 1_000_000
    operating_hours_per_day: int = 24
    projection_months: int = 12


def compute_tco(
    inference_result: SingleConfigResult,
    tco_params: TCOParams,
) -> TCOResult:
    """
    Project total cost of ownership for a production OCR service.

    Returns monthly and annual costs broken down by:
    - GPU rental
    - Power
    - Required GPU count to meet throughput target
    - Cost per 1000 pages
    """


@dataclass
class TCOResult:
    gpus_required: int
    monthly_gpu_cost: float
    monthly_power_cost: float
    monthly_total_cost: float
    annual_total_cost: float
    cost_per_1k_pages: float
    pages_per_gpu_per_day: float
```

### 11.4 Report Generator (`analysis/report.py`)

```python
class ReportGenerator:
    """Generate final Markdown report from aggregated results."""

    def generate(
        self,
        inference_results: list[InferenceBenchmarkResult],
        training_results: list[TrainingBenchmarkResult],
        quality_results: list[QualityEvalResult],
        output_path: Path,
    ) -> Path:
        """
        Generate report sections:
        1. Executive Summary (headline speedups, recommendations)
        2. Inference Results (tables, throughput plots, latency CDFs)
        3. Training Results (throughput, convergence, cost)
        4. Quality Analysis (precision degradation, per-doc-type)
        5. Power Efficiency (pages/Wh comparison)
        6. TCO Analysis (break-even, projections)
        7. Roofline Analysis (bottleneck identification)
        8. Pareto Frontier (quality-throughput tradeoff)
        9. Recommendations (model × GPU pairing guide)
        10. Appendix (full config snapshots, raw data links)
        """
```

---

## 12. CLI & Orchestration

### 12.1 CLI Entry Point (`cli.py`)

```python
import typer
from pathlib import Path

app = typer.Typer(name="vlm-ocr-bench", help="B300 vs H100 OCR VLM Benchmark")

@app.command()
def run(
    config: Path = typer.Argument(..., help="Experiment config YAML"),
    phase: Optional[str] = typer.Option(None, help="Run single phase: inference|training|quality"),
    model: Optional[str] = typer.Option(None, help="Run single model (overrides config)"),
    gpu: Optional[str] = typer.Option(None, help="Target GPU: h100_sxm|b300_sxm (auto-detect if omitted)"),
    precision: Optional[str] = typer.Option(None, help="Override precision: bf16|fp8|fp4"),
    batch_size: Optional[int] = typer.Option(None, help="Override batch size"),
    dry_run: bool = typer.Option(False, help="Validate config without running"),
    resume: bool = typer.Option(False, help="Resume from last checkpoint"),
    output_dir: Optional[Path] = typer.Option(None, help="Override output directory"),
):
    """Run benchmark experiment."""

@app.command()
def analyze(
    results_dir: Path = typer.Argument(..., help="Results directory to analyze"),
    report_format: str = typer.Option("markdown", help="markdown|html|pdf"),
):
    """Analyze results and generate report."""

@app.command()
def info():
    """Print detected GPU info and environment."""

@app.command()
def validate(
    config: Path = typer.Argument(..., help="Config YAML to validate"),
):
    """Validate experiment config without running."""

@app.command()
def list_models():
    """List all registered model configurations."""

@app.command()
def download(
    target: str = typer.Argument(..., help="'models' or 'data' or 'all'"),
):
    """Pre-download model weights and/or datasets."""
```

### 12.2 Orchestration Script (`scripts/run_all.sh`)

```bash
#!/usr/bin/env bash
set -euo pipefail

# Full experiment orchestration
CONFIG="${1:-configs/experiments/full_benchmark.yaml}"
GPU="${2:-auto}"

echo "=== VLM-OCR Benchmark Suite ==="
echo "Config: $CONFIG"
echo "GPU: $GPU"

# Step 1: Environment validation
vlm-ocr-bench info
vlm-ocr-bench validate "$CONFIG"

# Step 2: Lock GPU clocks
bash scripts/lock_clocks.sh

# Step 3: Run phases sequentially
for phase in inference training quality; do
    echo "=== Phase: $phase ==="
    vlm-ocr-bench run "$CONFIG" --phase "$phase" --gpu "$GPU"
done

# Step 4: Reset clocks
bash scripts/reset_gpu.sh

# Step 5: Analyze and generate report
vlm-ocr-bench analyze results/ --report-format markdown

echo "=== Benchmark Complete ==="
echo "Report: results/reports/report.md"
```

---

## 13. Containerization

### 13.1 Dockerfile.base

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.10-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget jq htop \
    && rm -rf /var/lib/apt/lists/*

# Python base dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[base]"

# Download DCGM exporter
RUN pip install --no-cache-dir nvidia-dcgm-bindings

WORKDIR /workspace/vlm-ocr-bench
COPY . .
```

### 13.2 Dockerfile.h100

```dockerfile
FROM vlm-ocr-bench:base

# H100-specific: FlashAttention 3
RUN pip install --no-cache-dir flash-attn==2.7.* --no-build-isolation

# vLLM with Hopper optimizations
RUN pip install --no-cache-dir vllm

# Verify H100 compatibility
RUN python -c "import flash_attn; print(f'FA version: {flash_attn.__version__}')"
```

### 13.3 Dockerfile.b300

```dockerfile
FROM vlm-ocr-bench:base

# B300-specific: FlashAttention 4 (CuTe-DSL)
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# vLLM compiled from source for Blackwell
RUN pip install --no-cache-dir vllm

# Verify Blackwell compatibility
RUN python -c "import torch; assert torch.cuda.get_device_capability()[0] >= 10, 'Need Blackwell GPU'"
```

### 13.4 docker-compose.yaml

```yaml
version: "3.8"

services:
  benchmark:
    build:
      context: .
      dockerfile: docker/Dockerfile.${GPU_TYPE:-h100}
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_DEVICE_ORDER=PCI_BUS_ID
    volumes:
      - ./configs:/workspace/vlm-ocr-bench/configs
      - ./results:/workspace/vlm-ocr-bench/results
      - ./data:/workspace/vlm-ocr-bench/data
      - model-cache:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  prometheus:
    image: prom/prometheus:v2.51.0
    volumes:
      - ./monitoring/prometheus:/etc/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:10.4.0
    volumes:
      - ./monitoring/grafana:/etc/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus

  dcgm-exporter:
    image: nvcr.io/nvidia/k8s/dcgm-exporter:3.3.5-3.4.1-ubuntu22.04
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./monitoring/dcgm/dcgm-exporter-config.csv:/etc/dcgm-exporter/customized.csv
    ports:
      - "9400:9400"

volumes:
  model-cache:
```

---

## 14. Dependencies

### 14.1 Core Dependencies (`pyproject.toml`)

```toml
[project]
name = "vlm-ocr-bench"
version = "1.0.0"
requires-python = ">=3.11"
dependencies = [
    # Core ML
    "torch>=2.5.0",
    "transformers>=4.46.0",
    "peft>=0.13.0",
    "accelerate>=1.0.0",
    "deepspeed>=0.15.0",
    # Inference
    "vllm>=0.6.0",
    # FlashAttention (installed per-GPU in Docker)
    # Data
    "datasets>=3.0.0",
    "Pillow>=10.0.0",
    "pdf2image>=1.17.0",
    # Evaluation
    "nltk>=3.9.0",
    "pythainlp>=5.0.0",     # Thai text segmentation
    "editdistance>=0.8.0",
    "sacrebleu>=2.4.0",
    # Profiling
    "pynvml>=12.0.0",
    # Analysis
    "numpy>=1.26.0",
    "pandas>=2.2.0",
    "scipy>=1.13.0",
    "matplotlib>=3.9.0",
    "plotly>=5.22.0",
    "pyarrow>=17.0.0",      # Parquet
    # Config & CLI
    "pydantic>=2.9.0",
    "pyyaml>=6.0.0",
    "typer>=0.12.0",
    "rich>=13.7.0",
    # Logging
    "structlog>=24.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=5.0.0",
    "ruff>=0.5.0",
    "mypy>=1.11.0",
    "pre-commit>=3.7.0",
]

[project.scripts]
vlm-ocr-bench = "vlm_ocr_bench.cli:app"
```

### 14.2 Version Pinning Strategy

| Category | Strategy | Rationale |
|----------|----------|-----------|
| PyTorch, vLLM, FlashAttention | Pin exact in Docker | GPU kernel compatibility is fragile |
| HuggingFace, PEFT | Pin minor version | API stability within minor |
| Analysis (numpy, pandas) | Pin major | Backward compatible |
| CLI, logging | Flexible | No correctness impact |

---

## 15. Testing Strategy

### 15.1 Test Tiers

| Tier | What | GPU Required | CI | Runtime |
|------|------|-------------|-----|---------|
| **Unit** | Config parsing, metrics calculation, statistics, adapters | No | Yes | <30s |
| **Integration** | Model loading, single inference, single train step | Yes (any) | GPU CI | <5min |
| **Smoke** | End-to-end on smallest model, batch_size=1, 5 samples | Yes (any) | GPU CI | <10min |
| **Full** | Complete benchmark (subset) | H100 + B300 | Manual | ~2hr |

### 15.2 Key Test Cases

```python
# tests/unit/test_config.py
class TestConfigLoading:
    def test_defaults_merge(self): ...
    def test_hardware_override(self): ...
    def test_invalid_precision_for_gpu(self): ...
    def test_cli_override(self): ...

# tests/unit/test_metrics.py
class TestEditDistance:
    def test_identical_strings(self): ...
    def test_empty_prediction(self): ...
    def test_unicode_normalization(self): ...
    def test_thai_text(self): ...

class TestBLEU:
    def test_english_bleu(self): ...
    def test_thai_segmented_bleu(self): ...

# tests/unit/test_statistics.py
class TestStatistics:
    def test_welch_t_significant(self): ...
    def test_welch_t_not_significant(self): ...
    def test_bootstrap_ci(self): ...
    def test_speedup_computation(self): ...

# tests/unit/test_adapters.py
class TestDotsOcrAdapter:
    def test_prompt_construction(self): ...
    def test_output_parsing_clean(self): ...
    def test_output_parsing_with_special_tokens(self): ...

# tests/integration/test_inference_engine.py
@pytest.mark.gpu
class TestVLLMEngine:
    def test_model_load(self): ...
    def test_single_inference(self): ...
    def test_batch_inference(self): ...
    def test_fp8_inference(self): ...

# tests/smoke/test_smoke.py
@pytest.mark.gpu
def test_full_pipeline_smoke():
    """Run smallest model, 5 samples, all 3 phases."""
```

### 15.3 CI Configuration

```yaml
# .github/workflows/ci.yml (conceptual)
jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - run: pip install -e ".[dev]"
      - run: pytest tests/unit/ -v --cov

  lint:
    runs-on: ubuntu-latest
    steps:
      - run: ruff check src/ tests/
      - run: mypy src/vlm_ocr_bench/

  gpu-tests:
    runs-on: [self-hosted, gpu]
    steps:
      - run: pip install -e ".[dev]"
      - run: pytest tests/integration/ tests/smoke/ -v --timeout=600
```

---

## 16. ADRs

### ADR-001: vLLM over TensorRT-LLM for Inference

**Status:** Accepted

**Context:** Need a serving engine supporting both H100 and B300 with FA-3/4 and FP8/FP4.

**Decision:** Use vLLM as the primary inference engine.

**Rationale:** vLLM provides native support for PagedAttention, FlashAttention 3 and 4, FP8 and FP4 KV caches, and has the widest model compatibility (all 6 target models work with vLLM). TensorRT-LLM would require per-model engine compilation and has slower iteration cycles.

**Tradeoff:** TensorRT-LLM may produce higher absolute throughput on some models. We accept this in exchange for broader compatibility and reproducibility.

---

### ADR-002: LoRA over Full Fine-Tuning

**Status:** Accepted

**Context:** Need to benchmark training speed across 6 models on 2 GPUs.

**Decision:** Use LoRA (r=64, α=128) for all training benchmarks.

**Rationale:** LoRA is the standard production approach for OCR model customization. Full fine-tuning would add excessive complexity (different memory/compute profiles per model) and wouldn't reflect real-world usage patterns.

**Tradeoff:** Results won't capture full-parameter training dynamics. We accept this because full fine-tuning is rarely done in OCR deployment.

---

### ADR-003: DCGM over PyTorch Profiler for GPU Monitoring

**Status:** Accepted

**Context:** Need non-intrusive GPU monitoring that doesn't affect benchmark timing.

**Decision:** Use DCGM exporter in a separate process with nvidia-smi as fallback.

**Rationale:** DCGM runs outside the CUDA context, providing system-level metrics (power, temperature, throttling) without adding overhead to the benchmark process. PyTorch Profiler operates within the process and can add up to 5% overhead.

**Tradeoff:** DCGM doesn't provide kernel-level traces. We complement with torch.cuda.Event timing for precise latency measurement.

---

### ADR-004: Parquet as Primary Result Format

**Status:** Accepted

**Context:** Profiling produces millions of rows of time-series data per experiment.

**Decision:** Store raw profiling data in Parquet, aggregate results in both Parquet and CSV.

**Rationale:** Parquet provides 10-50x compression over CSV for numerical time-series, supports efficient column selection for analysis, and integrates natively with pandas/polars. CSV exported for external sharing.

---

### ADR-005: Single-GPU Baseline Focus

**Status:** Accepted

**Context:** OCR VLM models are small (0.9B-9B); multi-GPU is typically unnecessary.

**Decision:** All primary benchmarks run on TP=1 (single GPU). Multi-GPU scaling is an optional extension for 7B+ models.

**Rationale:** Production OCR deployment uses single GPUs for models this size. Multi-GPU introduces NVLink/networking variables that obscure the core GPU comparison. We include optional 2-GPU scaling tests only for models where single-GPU approaches memory limits.

---

*End of Codebase Specification*
