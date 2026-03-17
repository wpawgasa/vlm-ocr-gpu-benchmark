# 13 — Testing

## Overview

Four-tier testing strategy: unit (no GPU), integration (GPU), smoke (quick e2e), full (complete benchmark).

## Checklist

### `tests/conftest.py`

- [ ] Shared fixtures:
  - `mock_gpu_info` — fake GPUInfo for unit tests
  - `sample_image` — small test document image
  - `sample_config` — minimal ExperimentConfig
  - `sample_ground_truth` — example GT for metric tests
  - `gpu_available` marker — skip if no GPU

### `tests/unit/test_config.py`

- [ ] Test defaults merge
- [ ] Test hardware config override
- [ ] Test invalid precision for GPU type raises error
- [ ] Test CLI override applies correctly
- [ ] Test config snapshot is saved

### `tests/unit/test_metrics.py`

- [ ] Test `normalized_edit_distance`: identical strings → 1.0
- [ ] Test `normalized_edit_distance`: empty prediction → 0.0
- [ ] Test unicode normalization (NFKC) applied
- [ ] Test Thai text edit distance
- [ ] Test English BLEU computation
- [ ] Test Thai segmented BLEU
- [ ] Test table structural accuracy
- [ ] Test formula structural accuracy

### `tests/unit/test_workload.py`

- [ ] Test workload generation produces correct batch sizes
- [ ] Test image preprocessing at different resolutions

### `tests/unit/test_adapters.py`

- [ ] Test each adapter's `build_prompt` output format
- [ ] Test each adapter's `parse_output` with clean text
- [ ] Test `parse_output` with special tokens
- [ ] Test `get_vllm_kwargs` returns valid dict

### `tests/unit/test_statistics.py`

- [ ] Test Welch's t-test with significantly different groups
- [ ] Test Welch's t-test with similar groups (not significant)
- [ ] Test bootstrap CI bounds
- [ ] Test speedup computation

### `tests/unit/test_tco.py`

- [ ] Test TCO computation with known inputs
- [ ] Test GPU count calculation for throughput target

### `tests/integration/test_inference_engine.py` (GPU required)

- [ ] Test model load succeeds
- [ ] Test single inference produces output
- [ ] Test batch inference
- [ ] Test FP8 inference (if supported)

### `tests/integration/test_training_loop.py` (GPU required)

- [ ] Test LoRA config applies correctly
- [ ] Test single training step completes
- [ ] Test gradient computation

### `tests/integration/test_eval_pipeline.py` (GPU + data required)

- [ ] Test evaluation pipeline produces scores

### `tests/smoke/test_smoke.py` (GPU required)

- [ ] Test full pipeline: smallest model, batch_size=1, 5 samples, all 3 phases

## Test Tiers

| Tier | GPU | CI | Runtime |
|------|-----|----|---------|
| Unit | No | Yes | <30s |
| Integration | Yes | GPU CI | <5min |
| Smoke | Yes | GPU CI | <10min |
| Full | H100+B300 | Manual | ~2hr |

## Key Rules

- Mark GPU tests with `@pytest.mark.gpu`
- Unit tests must run without GPU or model downloads
- Use mock/fake GPU info in unit tests
- Integration tests can use smallest model only
- Smoke test covers end-to-end with minimal data
- `--timeout=600` for GPU tests (model loading is slow)
