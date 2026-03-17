# 02 — Configuration System

## Overview

YAML-based hierarchical config with Pydantic validation. Configs compose via deep merge: defaults → hardware → model → phase → experiment → CLI overrides.

## Checklist

### `config/schema.py` — Pydantic Models

- [ ] Define enums: `PrecisionMode`, `FlashAttnVersion`, `OutputFormat`, `GPUType`
- [ ] Define `GPUConfig` model (gpu_type, precision_modes, flash_attn, kv_cache_dtype, tensor_parallel, lock_clocks, max_clock_mhz, power_limit_watts, cuda_version)
- [ ] Define `ModelConfig` model (name, hf_model_id, params_billion, tier, adapter, supported_resolutions, max_output_tokens, default_prompt_key, supports_flash_attn, vision_token_estimate, requires_padding)
- [ ] Define `InferenceConfig` model (batch_sizes, resolutions, output_formats, max_output_tokens_sweep, precision_modes, warmup_requests, measurement_requests, runs_per_config, cooldown_seconds, vllm_args)
- [ ] Define `LoRAConfig` model (rank, alpha, dropout, target_modules, bias, task_type)
- [ ] Define `TrainingConfig` model (lora, optimizer, learning_rate, lr_scheduler, warmup_ratio, epochs, effective_batch_size, micro_batch_sizes, precision_modes, gradient_checkpointing, deepspeed_stage, max_image_resolution, dataset, domain_dataset, warmup_steps, measurement_steps, runs_per_config, early_stop_patience)
- [ ] Define `QualityConfig` model (benchmarks, metrics, precision_modes, per_document_type, max_samples)
- [ ] Define `ProfilingConfig` model (enabled, backend, sample_interval_ms, fields, export_format)
- [ ] Define `ExperimentConfig` top-level model (name, description, seed, output_dir, models, gpus, phases, inference, training, quality, profiling, gpu_configs, model_configs)

### `config/loader.py` — YAML Loading

- [ ] Implement `load_yaml(path: Path) -> dict` — raw YAML parse
- [ ] Implement `deep_merge(base: dict, override: dict) -> dict` — deep merge, lists replaced not appended
- [ ] Implement `load_experiment_config(path: Path, cli_overrides: dict) -> ExperimentConfig`
  - Load defaults.yaml
  - Merge hardware configs referenced by experiment
  - Merge model configs referenced by experiment
  - Merge phase configs
  - Merge experiment manifest
  - Apply CLI overrides
  - Validate via Pydantic
- [ ] Implement config snapshot saving (copy final resolved config to results dir)

### `config/resolvers.py` — Dynamic Resolution

- [ ] Implement GPU auto-detection resolver (fill gpu_type from detected hardware)
- [ ] Implement path resolver (expand `~`, env vars, relative paths)
- [ ] Implement FlashAttention version resolver (`auto` → detect from GPU compute capability)

## Config Files to Create

- [ ] `configs/defaults.yaml` — seed, output_dir, profiling defaults
- [ ] `configs/hardware/h100_sxm.yaml` — H100 precision modes, FA3, clocks
- [ ] `configs/hardware/b300_sxm.yaml` — B300 precision modes, FA4, clocks
- [ ] `configs/models/paddleocr_vl_0.9b.yaml`
- [ ] `configs/models/dots_ocr_1.5_3b.yaml`
- [ ] `configs/models/nanonets_ocr2_3b.yaml`
- [ ] `configs/models/deepseek_ocr_3b.yaml`
- [ ] `configs/models/olmocr2_7b.yaml`
- [ ] `configs/models/chandra_ocr_9b.yaml`
- [ ] `configs/phases/inference.yaml`
- [ ] `configs/phases/training.yaml`
- [ ] `configs/phases/quality.yaml`
- [ ] `configs/experiments/full_benchmark.yaml`
- [ ] `configs/experiments/quick_smoke.yaml`
- [ ] `configs/experiments/single_model_debug.yaml`

## Key Rules

- Merge strategy: deep merge, later values override, lists are **replaced** not appended
- All configs must validate against Pydantic schemas — fail fast on invalid config
- Final resolved config is saved alongside results for reproducibility
