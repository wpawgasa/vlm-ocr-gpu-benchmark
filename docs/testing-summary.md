# Testing Infrastructure

**Date**: 2026-03-19
**Branch**: `test/testing-infrastructure`

## Overview

Implements the four-tier testing strategy: enhanced shared fixtures, GPU integration tests, and end-to-end smoke tests.

## Changes

### Modified Files
- `tests/conftest.py` — Enhanced with 11 shared fixtures (mock_gpu_info, sample_experiment_config, real_sample_image, sample_ground_truth, minimal phase configs, gpu_available marker)

### New Files
- `tests/integration/test_inference_engine.py` — 4 GPU integration tests (engine init, single/batch inference, full runner)
- `tests/integration/test_training_loop.py` — 2 GPU integration tests (LoRA config, training runner)
- `tests/integration/test_eval_pipeline.py` — 1 GPU integration test (quality evaluation runner)
- `tests/smoke/test_smoke.py` — 2 smoke tests (full inference pipeline e2e, CLI validate)

## Test Tiers

| Tier | GPU | CI | Runtime | Tests |
|------|-----|----|---------|-------|
| Unit | No | Yes | <30s | 505 |
| Integration | Yes | GPU CI | <5min | 7 |
| Smoke | Yes | GPU CI | <10min | 2 |

## Running Tests

```bash
# Unit tests only (no GPU)
pytest tests/unit/ -v --timeout=120 -m "not gpu and not slow"

# Integration tests (GPU required)
pytest tests/integration/ -v --timeout=600

# Smoke tests (GPU required)
pytest tests/smoke/ -v --timeout=600

# All tests
pytest tests/ -v --timeout=600
```
