# Training Benchmark Module (Phase B)

**Date**: 2026-03-18
**Branch**: `feature/training-benchmark`

## Overview

Implements the Phase B training benchmark module that measures LoRA fine-tuning throughput, memory usage, power efficiency, and convergence using HuggingFace Trainer + PEFT + DeepSpeed. The module sweeps across micro batch sizes and precision modes, performs binary search for OOM boundaries, and resets LoRA weights between runs for fair comparison.

## Changes

### New Files
- `src/vlm_ocr_bench/training/metrics.py` — TrainingMetrics dataclass, throughput/power/cost computation
- `src/vlm_ocr_bench/training/lora.py` — PEFT LoRA config builder with model-family detection, trainable param summary
- `src/vlm_ocr_bench/training/deepspeed_config.py` — DeepSpeed ZeRO Stage 2 config builder
- `src/vlm_ocr_bench/training/callbacks.py` — HF Trainer callbacks (ThroughputCallback, MemoryCallback, PowerCallback)
- `src/vlm_ocr_bench/training/data_collator.py` — VLMDataCollator for variable-size image+text batches with label masking
- `src/vlm_ocr_bench/training/trainer.py` — TrainingBenchmarkRunner orchestration + result dataclasses
- `src/vlm_ocr_bench/training/__init__.py` — Module exports
- `tests/unit/test_training.py` — 57 unit tests

### Modified Files
- `CLAUDE.md` — Updated training module status to DONE

## Technical Details

### Architecture
```
TrainingBenchmarkRunner.run()
  ├── For each precision:
  │   ├── _load_model_with_lora() → model + tokenizer
  │   ├── _find_max_batch_size() → binary search OOM boundary
  │   └── For each micro_batch_size × run_id:
  │       ├── _reset_lora_weights() → fair comparison
  │       └── _execute_training_config() → HF Trainer + callbacks
  └── Aggregate results
```

### Model Family Detection
- Qwen2-VL based (OlmOCR, Chandra): targets q_proj, k_proj, v_proj, o_proj
- ERNIE based (PaddleOCR-VL): auto-detect via empty target_modules
- Others: fallback to configured defaults or all Linear layers

### Key Patterns
- GPU-heavy imports (transformers, peft, torch) are lazy-loaded
- Binary search for max batch size with OOM recovery
- LoRA weights saved/restored between runs for reproducibility
- DeepSpeed ZeRO-2 auto-configured for 7B+ models
- Gradient checkpointing auto-enabled for 7B+ models

## Testing

- Unit tests: `tests/unit/test_training.py` (57 tests)
- Coverage areas: metrics computation, LoRA config building, model family detection, DeepSpeed config, callbacks, data collation (padding/truncation/label masking/variable-size images), result dataclasses, runner initialization, LoRA state management
- All tests run without GPU (mocked where needed)
