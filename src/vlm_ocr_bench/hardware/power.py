"""GPU power monitoring with background sampling."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import structlog

logger = structlog.get_logger()


@dataclass
class PowerSummary:
    """Summary of power measurements over a sampling period."""

    mean_watts: float = 0.0
    peak_watts: float = 0.0
    min_watts: float = 0.0
    total_energy_joules: float = 0.0
    total_energy_wh: float = 0.0
    duration_seconds: float = 0.0
    num_samples: int = 0
    throttle_events: int = 0


class PowerReader:
    """Background GPU power sampling using pynvml.

    Usage:
        reader = PowerReader(device_id=0, interval_ms=100)
        reader.start()
        # ... run workload ...
        summary = reader.stop()
    """

    def __init__(self, device_id: int = 0, interval_ms: int = 100) -> None:
        self._device_id = device_id
        self._interval_s = interval_ms / 1000.0
        self._samples: list[float] = []
        self._throttle_events = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self._start_time: float = 0.0
        self._stop_time: float = 0.0
        self._handle: object | None = None
        self._nvml_initialized = False

    def _init_nvml(self) -> bool:
        """Initialize NVML and get device handle."""
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_id)
            self._nvml_initialized = True
            return True
        except Exception:
            logger.warning("power_reader_nvml_init_failed", exc_info=True)
            return False

    def _shutdown_nvml(self) -> None:
        """Shutdown NVML cleanly."""
        if self._nvml_initialized:
            try:
                import pynvml

                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False

    def _sample_loop(self) -> None:
        """Background sampling loop."""
        import pynvml

        while self._running:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                self._samples.append(power_mw / 1000.0)  # Convert mW to W

                # Check for throttling
                try:
                    throttle = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(self._handle)
                    if throttle & (
                        pynvml.nvmlClocksThrottleReasonSwThermalSlowdown
                        | pynvml.nvmlClocksThrottleReasonHwThermalSlowdown
                        | pynvml.nvmlClocksThrottleReasonSwPowerCap
                    ):
                        self._throttle_events += 1
                except Exception:
                    pass

            except Exception:
                pass  # Skip failed sample, continue loop

            time.sleep(self._interval_s)

    def start(self) -> None:
        """Begin background power sampling."""
        if self._running:
            return

        if not self._init_nvml():
            logger.warning("power_reader_start_failed", reason="NVML init failed")
            return

        self._samples = []
        self._throttle_events = 0
        self._running = True
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        logger.info("power_reader_started", device_id=self._device_id)

    def stop(self) -> PowerSummary:
        """Stop sampling and return power summary."""
        self._running = False
        self._stop_time = time.monotonic()

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        self._shutdown_nvml()
        duration = self._stop_time - self._start_time

        if not self._samples:
            logger.warning("power_reader_no_samples")
            return PowerSummary(duration_seconds=duration)

        mean_w = sum(self._samples) / len(self._samples)
        peak_w = max(self._samples)
        min_w = min(self._samples)
        energy_j = mean_w * duration
        energy_wh = energy_j / 3600.0

        summary = PowerSummary(
            mean_watts=round(mean_w, 1),
            peak_watts=round(peak_w, 1),
            min_watts=round(min_w, 1),
            total_energy_joules=round(energy_j, 2),
            total_energy_wh=round(energy_wh, 4),
            duration_seconds=round(duration, 3),
            num_samples=len(self._samples),
            throttle_events=self._throttle_events,
        )
        logger.info(
            "power_reader_stopped",
            mean_watts=summary.mean_watts,
            peak_watts=summary.peak_watts,
            samples=summary.num_samples,
        )
        return summary

    def get_current_watts(self) -> float | None:
        """Get instantaneous power reading in watts."""
        if not self._nvml_initialized or self._handle is None:
            return None
        try:
            import pynvml

            power_mw: int = pynvml.nvmlDeviceGetPowerUsage(self._handle)
            return power_mw / 1000.0
        except Exception:
            return None
