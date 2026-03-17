# 03 — Hardware Abstraction

## Overview

GPU detection, power monitoring, and clock management. All GPU-specific logic lives here — core modules must remain GPU-agnostic.

## Checklist

### `hardware/detector.py`

- [ ] Define `GPUInfo` dataclass (name, gpu_type, compute_capability, vram_gb, driver_version, cuda_version, flash_attn_version, vllm_version, pytorch_version, num_sms, clock_mhz, power_limit_w)
- [ ] Implement `detect_gpu() -> GPUInfo` using pynvml
  - Query device name, compute capability, memory
  - Detect driver/CUDA version
  - Detect installed flash-attn, vllm, torch versions
  - Map device name → `GPUType` enum
- [ ] Implement `validate_gpu_for_config(gpu: GPUInfo, config: GPUConfig) -> list[str]`
  - Check precision mode support (FP4/NVFP4 → B300 only)
  - Check flash attention version compatibility
  - Check VRAM sufficiency for model size
  - Return warnings/errors list

### `hardware/profiles.py`

- [ ] Define hardware profile dataclasses with theoretical peaks:
  - H100 SXM: 80GB HBM3, 3.35 TB/s, 989 TFLOPS BF16, 1979 TFLOPS FP8, 700W TDP
  - B300 SXM: 288GB HBM3e, ~8 TB/s, ~2500 TFLOPS BF16, ~5000 TFLOPS FP8, ~1000W TDP
- [ ] Provide `get_profile(gpu_type: GPUType) -> HardwareProfile`

### `hardware/power.py`

- [ ] Implement `PowerReader` class
  - `__init__(backend, interval_ms)` — select DCGM or nvidia-smi
  - `start()` — begin background power sampling
  - `stop() -> PowerSummary` — stop, compute stats
  - `get_current_watts()` — instantaneous reading
- [ ] Define `PowerSummary` dataclass (mean/peak/min watts, total energy joules/Wh, duration, samples, throttle_events)

### `hardware/clock.py`

- [ ] Implement `lock_gpu_clocks(gpu_id, max_mhz)` — lock SM and memory clocks via pynvml
- [ ] Implement `reset_gpu_clocks(gpu_id)` — restore defaults
- [ ] Implement `get_clock_info(gpu_id) -> dict` — current clocks + throttle reasons

## Key Rules

- Use `pynvml` for all GPU queries — never shell out to `nvidia-smi` for detection
- `nvidia-smi` is only used as a fallback in `power.py` when DCGM is unavailable
- Clock locking requires root/sudo — handle permission errors gracefully
- All functions should work without a GPU present (return mock/error info) for unit testing
