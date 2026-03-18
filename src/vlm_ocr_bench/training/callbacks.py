"""HuggingFace Trainer callbacks for benchmark metrics collection."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class StepMetric:
    """Metric recorded at a single training step."""

    step: int
    value: float
    timestamp: float = 0.0


class ThroughputCallback:
    """Track samples/s and tokens/s per training step.

    Integrates with HF Trainer's callback protocol by providing
    ``on_step_begin`` / ``on_step_end`` / ``on_log`` methods.
    """

    def __init__(self) -> None:
        self.samples_per_second: list[StepMetric] = []
        self.tokens_per_second: list[StepMetric] = []
        self._step_start_time: float = 0.0
        self._step_samples: int = 0
        self._step_tokens: int = 0

    def on_step_begin(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        """Record step start time."""
        self._step_start_time = time.monotonic()

    def on_step_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        """Compute and store throughput for the completed step."""
        elapsed = time.monotonic() - self._step_start_time
        if elapsed > 0 and self._step_samples > 0:
            sps = self._step_samples / elapsed
            tps = self._step_tokens / elapsed if self._step_tokens > 0 else 0.0
            step = state.global_step if state else 0
            now = time.monotonic()

            self.samples_per_second.append(StepMetric(step=step, value=sps, timestamp=now))
            self.tokens_per_second.append(StepMetric(step=step, value=tps, timestamp=now))

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Extract sample/token counts from trainer logs."""
        if logs is None:
            return
        # HF Trainer provides these in logs when available
        self._step_samples = int(logs.get("train_samples_per_second", 0) or 0)
        self._step_tokens = int(logs.get("train_tokens_per_second", 0) or 0)

    def get_mean_samples_per_second(self) -> float:
        """Average samples/s across all recorded steps."""
        if not self.samples_per_second:
            return 0.0
        return sum(m.value for m in self.samples_per_second) / len(self.samples_per_second)

    def get_mean_tokens_per_second(self) -> float:
        """Average tokens/s across all recorded steps."""
        if not self.tokens_per_second:
            return 0.0
        return sum(m.value for m in self.tokens_per_second) / len(self.tokens_per_second)


class MemoryCallback:
    """Track peak GPU memory per training step."""

    def __init__(self) -> None:
        self.peak_memory_gb: list[StepMetric] = []
        self._torch_available = False

        try:
            import torch

            self._torch_available = torch.cuda.is_available()
        except ImportError:
            pass

    def on_step_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        """Record peak GPU memory after each step."""
        if not self._torch_available:
            return

        import torch

        peak_bytes = torch.cuda.max_memory_allocated()
        peak_gb = peak_bytes / (1024**3)
        step = state.global_step if state else 0

        self.peak_memory_gb.append(StepMetric(step=step, value=peak_gb, timestamp=time.monotonic()))

    def get_peak_memory_gb(self) -> float:
        """Maximum peak memory across all steps."""
        if not self.peak_memory_gb:
            return 0.0
        return max(m.value for m in self.peak_memory_gb)

    def get_memory_breakdown(self) -> dict[str, float]:
        """Get current GPU memory breakdown.

        Returns dict with model, optimizer, activation memory estimates in GB.
        """
        if not self._torch_available:
            return {"model_gb": 0.0, "optimizer_gb": 0.0, "activation_gb": 0.0}

        import torch

        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)

        return {
            "model_gb": 0.0,  # Requires hooking into model loading
            "optimizer_gb": 0.0,  # Requires hooking into optimizer state
            "activation_gb": max(0.0, reserved - allocated),
        }


@dataclass
class PowerSample:
    """A single power measurement."""

    step: int
    watts: float
    timestamp: float = 0.0


class PowerCallback:
    """Track power draw via hardware/power.py during training.

    Uses the existing PowerReader for background sampling, and records
    per-step snapshots for correlation with training events.
    """

    def __init__(self, device_id: int = 0, interval_ms: int = 100) -> None:
        self._device_id = device_id
        self._interval_ms = interval_ms
        self.power_samples: list[PowerSample] = []
        self._reader: Any = None

    def on_train_begin(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        """Start power monitoring when training begins."""
        try:
            from vlm_ocr_bench.hardware.power import PowerReader

            self._reader = PowerReader(
                device_id=self._device_id,
                interval_ms=self._interval_ms,
            )
            self._reader.start()
        except Exception:
            logger.warning("power_callback_start_failed", exc_info=True)

    def on_step_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        """Record instantaneous power at step end."""
        if self._reader is None:
            return

        watts = self._reader.get_current_watts()
        if watts is not None:
            step = state.global_step if state else 0
            self.power_samples.append(
                PowerSample(step=step, watts=watts, timestamp=time.monotonic())
            )

    def on_train_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        """Stop power monitoring and collect summary."""
        if self._reader is not None:
            self._power_summary = self._reader.stop()
            self._reader = None

    def get_mean_power_watts(self) -> float:
        """Average power across all samples."""
        if not self.power_samples:
            return 0.0
        return sum(s.watts for s in self.power_samples) / len(self.power_samples)

    def get_peak_power_watts(self) -> float:
        """Peak power across all samples."""
        if not self.power_samples:
            return 0.0
        return max(s.watts for s in self.power_samples)

    def get_power_summary(self) -> Any:
        """Return the PowerSummary from the underlying reader, if available."""
        return getattr(self, "_power_summary", None)
