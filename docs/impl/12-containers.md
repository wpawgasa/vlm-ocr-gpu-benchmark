# 12 — Containerization

## Overview

GPU-specific Docker images (H100 vs B300) with monitoring stack (Prometheus, Grafana, DCGM exporter) via docker-compose.

## Checklist

### `docker/Dockerfile.base`

- [ ] Base: `nvcr.io/nvidia/pytorch:24.10-py3`
- [ ] Install system deps (git, curl, wget, jq, htop)
- [ ] Install Python base deps from pyproject.toml
- [ ] Install nvidia-dcgm-bindings
- [ ] Set WORKDIR to `/workspace/vlm-ocr-bench`
- [ ] Copy project source

### `docker/Dockerfile.h100`

- [ ] FROM vlm-ocr-bench:base
- [ ] Install FlashAttention 3 (`flash-attn==2.7.*`)
- [ ] Install vLLM (Hopper optimized)
- [ ] Verify FA version at build time

### `docker/Dockerfile.b300`

- [ ] FROM vlm-ocr-bench:base
- [ ] Install FlashAttention 4 (CuTe-DSL)
- [ ] Install vLLM (Blackwell compiled)
- [ ] Verify compute capability ≥ 10.0 at build time

### `docker/docker-compose.yaml`

- [ ] Service: `benchmark` — GPU-specific Dockerfile, nvidia runtime, volume mounts (configs, results, data, model cache)
- [ ] Service: `prometheus` — scrape DCGM exporter
- [ ] Service: `grafana` — pre-provisioned dashboard, port 3000
- [ ] Service: `dcgm-exporter` — nvidia runtime, custom field config, port 9400
- [ ] Volume: `model-cache` for HuggingFace cache persistence

### `docker/.env.template`

- [ ] `GPU_TYPE=h100` (or b300)
- [ ] `NVIDIA_VISIBLE_DEVICES=all`
- [ ] `HF_TOKEN=` (for gated model access)
- [ ] `CUDA_DEVICE_ORDER=PCI_BUS_ID`

## Key Rules

- FlashAttention is GPU-specific — installed in per-GPU Dockerfiles, NOT in pyproject.toml
- Pin exact versions of torch, vLLM, flash-attn in Dockerfiles (GPU kernel compatibility is fragile)
- Use `--no-build-isolation` for flash-attn pip install
- Model cache is a named volume for persistence across container restarts
- DCGM exporter runs alongside benchmark in separate container
