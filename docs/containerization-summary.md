# Containerization Module

**Date**: 2026-03-19
**Branch**: `feature/containerization`

## Overview

GPU-specific Docker images (H100 vs B300) with monitoring stack (Prometheus, Grafana, DCGM exporter) via docker-compose.

## Changes

### New Files

**Docker** (`docker/`):
- `Dockerfile.base` — Base image on `nvcr.io/nvidia/pytorch:24.10-py3` with project dependencies
- `Dockerfile.h100` — H100/Hopper optimized with FlashAttention 3 + vLLM
- `Dockerfile.b300` — B300/Blackwell optimized with FlashAttention 4 + vLLM, CC >= 10.0 verification
- `docker-compose.yaml` — 4 services: benchmark, dcgm-exporter, prometheus, grafana
- `.env.template` — Environment variable template (GPU_TYPE, HF_TOKEN, etc.)
- `.dockerignore` — Build context exclusions

**Monitoring** (`monitoring/`):
- `prometheus/prometheus.yml` — Scrape DCGM exporter at :9400
- `grafana/provisioning/datasources.yaml` — Auto-provision Prometheus datasource
- `grafana/provisioning/dashboards/dashboards.yaml` — Auto-provision dashboard
- `grafana/dashboards/gpu_benchmark.json` — 6-panel dashboard (temp, power, SM occupancy, tensor util, memory, PCIe)
- `dcgm/dcgm-exporter-config.csv` — DCGM field selection (15 fields)

## Usage

```bash
# Build for H100
GPU_TYPE=h100 docker compose -f docker/docker-compose.yaml build

# Run full benchmark with monitoring
GPU_TYPE=h100 docker compose -f docker/docker-compose.yaml up

# Access Grafana dashboard
open http://localhost:3000
```
