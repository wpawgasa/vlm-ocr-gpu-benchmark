# 06 — Inference Benchmark (Phase A)

## Overview

Measures inference latency, throughput, memory, and power for each model × GPU × precision × batch_size × resolution combination using vLLM.

## Checklist

### `inference/engine.py`

- [ ] Implement `VLLMEngine` class:
  - `__init__(model_config, gpu_config, inference_config)` — store configs, get adapter
  - `initialize() -> EngineInfo` — load model into vLLM, return load time + memory stats
  - `generate_batch(inputs, sampling_params) -> list[GenerationResult]` — batch inference with timing
  - `shutdown()` — release GPU memory
  - `get_memory_stats() -> MemoryStats`
- [ ] Define `EngineInfo` dataclass (model_load_time_s, gpu_memory_allocated/reserved_gb, flash_attn_version, actual_precision, kv_cache_dtype, max_batch_size, max_model_len)
- [ ] Define `GenerationResult` dataclass (output_text, num_output/input_tokens, ttft_ms, generation_time_ms, total_time_ms, tokens_per_second)
- [ ] Use `torch.cuda.Event` for precise GPU timing (not `time.time()`)

### `inference/runner.py`

- [ ] Implement `InferenceBenchmarkRunner`:
  - `run() -> InferenceBenchmarkResult` — full sweep
  - `_run_single_config(resolution, batch_size, precision, max_tokens, run_id) -> SingleConfigResult`
  - `_warmup(engine, n)` — discard warmup results
- [ ] Algorithm:
  1. Initialize vLLM engine
  2. For each (resolution, batch_size, precision, max_tokens):
     - Generate workload batch
     - Warmup (50 requests default)
     - For each run (3 runs default):
       - Start profiling monitor
       - Execute 200 measurement requests
       - Stop profiling, collect GPU metrics
       - Compute metrics (latency percentiles, throughput, power)
     - Cooldown (60s default)
  3. Aggregate across runs
  4. Save raw + aggregated results
- [ ] Define `InferenceBenchmarkResult` dataclass
- [ ] Define `SingleConfigResult` dataclass with all metrics (see spec section 7.2)

### `inference/workload.py`

- [ ] Implement workload generator:
  - Select images from evaluation dataset at target resolution
  - Build prompts via adapter
  - Create batches of specified size
  - Return vLLM-compatible input dicts

### `inference/metrics.py`

- [ ] Define `InferenceMetrics` dataclass
- [ ] Implement metric collectors:
  - Latency percentiles (p50, p95, p99) for TTFT and end-to-end
  - Throughput: pages/second, tokens/second
  - Power efficiency: energy_per_page_wh, pages_per_watt_hour, tokens_per_watt_hour
  - Memory: peak GPU memory, KV cache memory
  - GPU utilization: SM occupancy, tensor core util, memory util

### `inference/vllm_config.py`

- [ ] Implement `build_vllm_engine_args(model_config, gpu_config, precision, adapter) -> dict`
  - H100: enforce_eager=False, FA-3, FP8 quantization/KV cache
  - B300: enforce_eager=False, FA-4, FP4/NVFP4 support
  - `gpu_memory_utilization=0.90` (leave room for profiling)
  - Merge adapter-specific vLLM kwargs
- [ ] Validate precision mode against GPU type (FP4/NVFP4 → B300 only)

## Key Rules

- Always warmup before measurement — GPU needs thermal + JIT stabilization
- Cooldown between configs to prevent thermal throttling artifacts
- Use CUDA events for timing, not wall clock
- Log throttle events — they invalidate benchmark results
- OOM is expected for large batch sizes — catch and record, don't crash
