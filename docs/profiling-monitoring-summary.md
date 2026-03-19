# Profiling & Monitoring Module

**Date**: 2026-03-19
**Branch**: `feature/profiling-monitoring`

## Overview

Implements background GPU profiling that runs in a separate process to avoid benchmark interference. Includes pynvml-based monitoring, nvidia-smi fallback, throttle event detection, and Parquet time-series export.

## Changes

### New Files
- `profiling/monitor.py` — GPUMonitor (multiprocessing.Process wrapper), ProfilingResult, ProfilingSummary dataclasses, background metric collection
- `profiling/nvidia_smi.py` — nvidia-smi CLI polling fallback (NvidiaSmiSample, query_nvidia_smi, is_nvidia_smi_available)
- `profiling/events.py` — Throttle reason parsing, critical throttle detection, event summarization
- `tests/unit/test_profiling.py` — 40 unit tests

## Technical Details

- Monitor runs in `multiprocessing.Process` (never in benchmark process per ADR-003)
- Benchmark continues if profiling fails (fail-safe via try/except)
- Collects: GPU temp, power, utilization, memory, throttle reasons at configurable interval
- Checkpoint labels at warmup/measure/cooldown boundaries
- Parquet export for time-series data
- Existing `hardware/power.py` (PowerReader/PowerSummary) preserved — used by training callbacks

## Testing

- 40 unit tests, 505 total passing, ruff clean, mypy clean
