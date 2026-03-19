"""GPU monitor — background profiling in a separate process."""

from __future__ import annotations

import multiprocessing
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class ProfilingSummary:
    """Summary statistics from a profiling session."""

    # Temperature (Celsius)
    gpu_temp_mean: float = 0.0
    gpu_temp_max: float = 0.0

    # Power (Watts)
    power_mean: float = 0.0
    power_max: float = 0.0
    power_min: float = 0.0
    total_energy_joules: float = 0.0
    total_energy_wh: float = 0.0

    # GPU utilization (0-100)
    sm_occupancy_mean: float = 0.0
    tensor_util_mean: float = 0.0
    memory_util_mean: float = 0.0
    memory_used_peak_gb: float = 0.0

    # Throttle
    throttle_events: int = 0
    throttle_reasons: list[str] = field(default_factory=list)

    # Timing
    duration_s: float = 0.0
    num_samples: int = 0


@dataclass
class ProfilingResult:
    """Result from a completed profiling session."""

    data_path: Path | None = None
    summary: ProfilingSummary = field(default_factory=ProfilingSummary)
    duration_s: float = 0.0
    num_samples: int = 0
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def _monitor_process(
    device_id: int,
    interval_ms: int,
    stop_event: multiprocessing.synchronize.Event,
    checkpoint_queue: multiprocessing.Queue[Any],
    result_queue: multiprocessing.Queue[Any],
    output_path: str,
) -> None:
    """Background monitoring process function.

    Runs in a separate process to avoid interfering with benchmarks.
    Collects GPU metrics via pynvml at the configured interval.
    """
    samples: list[dict[str, float]] = []
    checkpoints: list[dict[str, Any]] = []
    start_time = time.monotonic()

    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    except Exception as exc:
        result_queue.put({"error": str(exc), "samples": [], "checkpoints": []})
        return

    interval_s = interval_ms / 1000.0

    try:
        while not stop_event.is_set():
            ts = time.monotonic() - start_time
            sample: dict[str, float] = {"timestamp_s": ts}

            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                sample["gpu_temp_c"] = float(temp)
            except Exception:
                pass

            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                sample["power_w"] = power_mw / 1000.0
            except Exception:
                pass

            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                sample["gpu_util_pct"] = float(util.gpu)
                sample["mem_util_pct"] = float(util.memory)
            except Exception:
                pass

            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                sample["mem_used_gb"] = mem.used / (1024**3)
                sample["mem_total_gb"] = mem.total / (1024**3)
            except Exception:
                pass

            try:
                throttle = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
                sample["throttle_reasons"] = float(throttle)
            except Exception:
                pass

            samples.append(sample)

            # Check for checkpoint labels
            while not checkpoint_queue.empty():
                try:
                    label = checkpoint_queue.get_nowait()
                    checkpoints.append({"label": label, "timestamp_s": ts})
                except Exception:
                    break

            time.sleep(interval_s)

    except Exception:
        pass
    finally:
        import contextlib

        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()

    # Save to parquet if we have data
    data_path = None
    if samples:
        try:
            import pandas as pd

            df = pd.DataFrame(samples)
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out, index=False)
            data_path = output_path
        except Exception:
            pass

    result_queue.put(
        {
            "samples": samples,
            "checkpoints": checkpoints,
            "data_path": data_path,
        }
    )


class GPUMonitor:
    """Background GPU monitor running in a separate process.

    Usage:
        monitor = GPUMonitor(device_id=0, interval_ms=100)
        monitor.start(output_path="results/profiling/run_001.parquet")
        # ... run benchmark ...
        monitor.checkpoint("warmup_end")
        # ... more benchmark ...
        result = monitor.stop()
    """

    def __init__(
        self,
        device_id: int = 0,
        interval_ms: int = 100,
    ) -> None:
        self._device_id = device_id
        self._interval_ms = interval_ms
        self._process: multiprocessing.Process | None = None
        self._stop_event: multiprocessing.synchronize.Event | None = None
        self._checkpoint_queue: multiprocessing.Queue[Any] | None = None
        self._result_queue: multiprocessing.Queue[Any] | None = None
        self._start_time: float = 0.0

    def start(self, output_path: str | Path = "profiling_data.parquet") -> None:
        """Launch the background monitoring process."""
        if self._process is not None and self._process.is_alive():
            logger.warning("monitor_already_running")
            return

        self._stop_event = multiprocessing.Event()
        self._checkpoint_queue = multiprocessing.Queue()
        self._result_queue = multiprocessing.Queue()
        self._start_time = time.monotonic()

        self._process = multiprocessing.Process(
            target=_monitor_process,
            args=(
                self._device_id,
                self._interval_ms,
                self._stop_event,
                self._checkpoint_queue,
                self._result_queue,
                str(output_path),
            ),
            daemon=True,
        )
        self._process.start()
        logger.info("gpu_monitor_started", device_id=self._device_id)

    def checkpoint(self, label: str) -> None:
        """Insert a labeled timestamp marker into the profiling data."""
        if self._checkpoint_queue is not None:
            self._checkpoint_queue.put(label)

    def stop(self) -> ProfilingResult:
        """Stop monitoring and return profiling results.

        Blocks until the monitor process finishes (with timeout).
        Returns empty ProfilingResult if monitoring wasn't running or failed.
        """
        if self._process is None or self._stop_event is None:
            return ProfilingResult(error="Monitor was not started")

        # Signal stop
        self._stop_event.set()
        self._process.join(timeout=10.0)

        if self._process.is_alive():
            self._process.terminate()
            logger.warning("gpu_monitor_terminated_forcefully")

        duration = time.monotonic() - self._start_time

        # Collect results from queue
        try:
            if self._result_queue is not None and not self._result_queue.empty():
                raw = self._result_queue.get(timeout=5.0)
            else:
                return ProfilingResult(duration_s=duration, error="No results from monitor")
        except Exception:
            return ProfilingResult(duration_s=duration, error="Failed to retrieve results")

        if "error" in raw:
            return ProfilingResult(duration_s=duration, error=raw["error"])

        samples: list[dict[str, float]] = raw.get("samples", [])
        checkpoints: list[dict[str, Any]] = raw.get("checkpoints", [])

        summary = _compute_summary(samples, duration)
        data_path = raw.get("data_path")

        result = ProfilingResult(
            data_path=Path(data_path) if data_path else None,
            summary=summary,
            duration_s=duration,
            num_samples=len(samples),
            checkpoints=checkpoints,
        )

        logger.info(
            "gpu_monitor_stopped",
            samples=result.num_samples,
            duration_s=round(duration, 1),
            power_mean=summary.power_mean,
        )
        return result


def _compute_summary(
    samples: list[dict[str, float]],
    duration: float,
) -> ProfilingSummary:
    """Compute summary statistics from raw profiling samples."""
    if not samples:
        return ProfilingSummary(duration_s=duration)

    import numpy as np

    temps = [s["gpu_temp_c"] for s in samples if "gpu_temp_c" in s]
    powers = [s["power_w"] for s in samples if "power_w" in s]
    gpu_utils = [s["gpu_util_pct"] for s in samples if "gpu_util_pct" in s]
    mem_utils = [s["mem_util_pct"] for s in samples if "mem_util_pct" in s]
    mem_used = [s["mem_used_gb"] for s in samples if "mem_used_gb" in s]
    throttles = [s["throttle_reasons"] for s in samples if "throttle_reasons" in s]

    # Count non-zero throttle events
    throttle_count = sum(1 for t in throttles if t != 0.0)

    # Identify unique throttle reasons
    throttle_reasons: list[str] = []
    reason_map = {
        1: "GpuIdle",
        2: "ApplicationsClocks",
        4: "SwPowerCap",
        8: "HwSlowdown",
        16: "SyncBoost",
        32: "SwThermalSlowdown",
        64: "HwThermalSlowdown",
        128: "HwPowerBrakeSlowdown",
    }
    seen: set[str] = set()
    for t in throttles:
        code = int(t)
        for bit, name in reason_map.items():
            if code & bit and name not in seen:
                throttle_reasons.append(name)
                seen.add(name)

    power_mean = float(np.mean(powers)) if powers else 0.0
    energy_j = power_mean * duration
    energy_wh = energy_j / 3600.0

    return ProfilingSummary(
        gpu_temp_mean=float(np.mean(temps)) if temps else 0.0,
        gpu_temp_max=float(np.max(temps)) if temps else 0.0,
        power_mean=round(power_mean, 1),
        power_max=round(float(np.max(powers)), 1) if powers else 0.0,
        power_min=round(float(np.min(powers)), 1) if powers else 0.0,
        total_energy_joules=round(energy_j, 2),
        total_energy_wh=round(energy_wh, 4),
        sm_occupancy_mean=float(np.mean(gpu_utils)) if gpu_utils else 0.0,
        tensor_util_mean=0.0,  # Not available from basic pynvml
        memory_util_mean=float(np.mean(mem_utils)) if mem_utils else 0.0,
        memory_used_peak_gb=round(float(np.max(mem_used)), 2) if mem_used else 0.0,
        throttle_events=throttle_count,
        throttle_reasons=throttle_reasons,
        duration_s=duration,
        num_samples=len(samples),
    )
