# 09 — Profiling & Monitoring

## Overview

Background GPU monitoring via DCGM (primary) or nvidia-smi (fallback). Runs in a separate process to avoid benchmark interference. Includes Prometheus/Grafana observability stack.

## Checklist

### `profiling/monitor.py`

- [ ] Implement `GPUMonitor` class:
  - Runs in `multiprocessing.Process` — separate from benchmark process
  - Uses `multiprocessing.Event` for start/stop/checkpoint communication
  - `start(output_path)` — launch background monitoring
  - `stop() -> ProfilingResult` — stop, wait for process, return results
  - `checkpoint(label)` — insert labeled timestamp marker
- [ ] Define `ProfilingResult` dataclass (data_path to Parquet, summary, duration_s, num_samples)
- [ ] Define `ProfilingSummary` dataclass (gpu_temp mean/max, power mean/max/total, SM occupancy, tensor util, memory util/peak, throttle events/reasons)
- [ ] Fail-safe: benchmark continues even if monitoring process crashes

### `profiling/dcgm.py`

- [ ] Implement DCGM field collection:
  - Fields: gpu_temp, gpu_power, sm_active, sm_occupancy, tensor_active, dram_active, pcie_tx/rx_bytes, gpu_mem_used/total, throttle_reasons
  - 100ms default sample interval
  - Output to Parquet time-series

### `profiling/nvidia_smi.py`

- [ ] Implement nvidia-smi polling fallback:
  - Parse `nvidia-smi --query-gpu=...` output
  - Same fields as DCGM where available
  - Used when DCGM is not installed

### `profiling/power.py`

- [ ] Implement power measurement:
  - Integrate power samples over time → total energy (Joules, Wh)
  - Detect and count throttle events
  - Compute mean/peak/min power

### `profiling/events.py`

- [ ] Implement thermal throttle detection:
  - Parse throttle reason codes from DCGM/nvidia-smi
  - Log warnings when throttling occurs
  - Count throttle events per benchmark run

## Monitoring Stack Files

- [ ] `monitoring/prometheus/prometheus.yml` — scrape DCGM exporter at :9400
- [ ] `monitoring/grafana/provisioning/dashboards.yaml` — auto-provision dashboard
- [ ] `monitoring/grafana/dashboards/gpu_benchmark.json` — panels: temperature, power, SM occupancy, tensor util, memory, PCIe throughput, benchmark progress
- [ ] `monitoring/dcgm/dcgm-exporter-config.csv` — selected DCGM fields

## Key Rules

- Monitor MUST run in separate process — never in benchmark process (ADR-003)
- Benchmark continues if profiling fails — wrap in try/except, log warning
- DCGM is preferred (per-SM metrics at 100ms) — nvidia-smi is fallback only
- Parquet is the output format for time-series data (efficient for millions of rows)
- Insert checkpoint labels at warmup/measure/cooldown boundaries
