# 07 — Training Benchmark (Phase B)

## Overview

Measures LoRA fine-tuning throughput, memory, power, and convergence using HuggingFace Trainer + PEFT + DeepSpeed.

## Checklist

### `training/lora.py`

- [ ] Implement `build_peft_config(model_config, training_config) -> PeftConfig`
  - Qwen2-VL based (OlmOCR, Chandra): target q_proj, k_proj, v_proj, o_proj
  - ERNIE based (PaddleOCR-VL): auto-detect attention layers
  - Fallback: all Linear layers
  - Default: rank=64, alpha=128, dropout=0.05
- [ ] Implement `get_trainable_param_summary(model) -> dict` — total/trainable params, percentage, per-module breakdown

### `training/trainer.py`

- [ ] Implement `TrainingBenchmarkRunner`:
  - `run() -> TrainingBenchmarkResult`
  - `_find_max_batch_size(model, precision) -> int` — binary search for OOM boundary
  - `_run_throughput_measurement(model, dataloader, micro_batch_size, precision, steps) -> ThroughputResult`
- [ ] Algorithm:
  1. Load base model with LoRA config
  2. Load training dataset (Docmatix 50K or Thai domain set)
  3. For each (micro_batch_size, precision):
     - Configure HF Trainer with metric callbacks
     - Warmup steps (discard)
     - For each run (3 default):
       - Reset LoRA weights to initial
       - Start profiling
       - Train for measurement_steps
       - Record throughput, loss, power
       - Stop profiling
     - Determine max feasible micro_batch (OOM boundary)
  4. Full convergence run: best config × 3 epochs
  5. Save results
- [ ] Define `TrainingBenchmarkResult`, `TrainingConfigResult`, `ConvergenceResult` dataclasses (see spec section 8.1)

### `training/data_collator.py`

- [ ] Implement `VLMDataCollator`:
  - Handle variable-size images (pad to batch max)
  - Interleave image + text sequences
  - Handle dynamic vision token counts
  - Mask labels: -100 for vision tokens (only supervise text output)

### `training/callbacks.py`

- [ ] Implement `ThroughputCallback` — log samples/s and tokens/s per step
- [ ] Implement `MemoryCallback` — log peak GPU memory per step
- [ ] Implement `PowerCallback` — log power draw via hardware/power.py

### `training/metrics.py`

- [ ] Define `TrainingMetrics` dataclass and collectors

### `training/deepspeed_config.py`

- [ ] Implement `build_deepspeed_config(training_config, gpu_config) -> dict`
  - ZeRO Stage 2 for 7B+ models
  - BF16/FP16 mixed precision settings
  - Gradient accumulation config

## Key Rules

- Always use LoRA, never full fine-tuning (ADR-002)
- Gradient checkpointing auto-enabled for 7B+ models
- DeepSpeed Stage 2 auto-enabled for 7B+ models
- Binary search for max batch size — catch OOM gracefully
- Reset LoRA weights between runs for fair comparison
- `effective_batch_size = micro_batch_size × gradient_accumulation_steps`
