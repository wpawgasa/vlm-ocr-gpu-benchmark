"""Unit tests for the profiling & monitoring module (no GPU)."""

from __future__ import annotations

from vlm_ocr_bench.profiling.events import (
    CRITICAL_THROTTLE_MASK,
    ThrottleEvent,
    detect_throttle_event,
    is_critical_throttle,
    parse_throttle_reasons,
    summarize_throttle_events,
)
from vlm_ocr_bench.profiling.monitor import (
    GPUMonitor,
    ProfilingResult,
    ProfilingSummary,
    _compute_summary,
)
from vlm_ocr_bench.profiling.nvidia_smi import (
    NvidiaSmiSample,
    _parse_float,
)

# ─── ProfilingSummary Tests ───


class TestProfilingSummary:
    def test_defaults(self) -> None:
        s = ProfilingSummary()
        assert s.power_mean == 0.0
        assert s.gpu_temp_mean == 0.0
        assert s.throttle_events == 0
        assert s.throttle_reasons == []
        assert s.num_samples == 0

    def test_with_values(self) -> None:
        s = ProfilingSummary(
            gpu_temp_mean=65.0,
            gpu_temp_max=80.0,
            power_mean=500.0,
            power_max=600.0,
            throttle_events=3,
            num_samples=100,
        )
        assert s.gpu_temp_mean == 65.0
        assert s.power_mean == 500.0


# ─── ProfilingResult Tests ───


class TestProfilingResult:
    def test_defaults(self) -> None:
        r = ProfilingResult()
        assert r.data_path is None
        assert r.num_samples == 0
        assert r.error is None
        assert r.checkpoints == []

    def test_with_error(self) -> None:
        r = ProfilingResult(error="GPU not found")
        assert r.error == "GPU not found"


# ─── _compute_summary Tests ───


class TestComputeSummary:
    def test_empty_samples(self) -> None:
        summary = _compute_summary([], 10.0)
        assert summary.num_samples == 0
        assert summary.power_mean == 0.0
        assert summary.duration_s == 10.0

    def test_with_samples(self) -> None:
        samples = [
            {
                "timestamp_s": 0.0,
                "gpu_temp_c": 60.0,
                "power_w": 400.0,
                "gpu_util_pct": 90.0,
                "mem_util_pct": 50.0,
                "mem_used_gb": 20.0,
                "throttle_reasons": 0.0,
            },
            {
                "timestamp_s": 0.1,
                "gpu_temp_c": 70.0,
                "power_w": 500.0,
                "gpu_util_pct": 95.0,
                "mem_util_pct": 55.0,
                "mem_used_gb": 22.0,
                "throttle_reasons": 0.0,
            },
        ]
        summary = _compute_summary(samples, 10.0)
        assert summary.num_samples == 2
        assert summary.gpu_temp_mean == 65.0
        assert summary.gpu_temp_max == 70.0
        assert summary.power_mean == 450.0
        assert summary.power_max == 500.0
        assert summary.power_min == 400.0
        assert summary.memory_used_peak_gb == 22.0
        assert summary.throttle_events == 0

    def test_with_throttle(self) -> None:
        samples = [
            {"timestamp_s": 0.0, "power_w": 500.0, "throttle_reasons": 0.0},
            {
                "timestamp_s": 0.1,
                "power_w": 500.0,
                "throttle_reasons": float(0x20),
            },  # SwThermalSlowdown
        ]
        summary = _compute_summary(samples, 1.0)
        assert summary.throttle_events == 1
        assert "SwThermalSlowdown" in summary.throttle_reasons

    def test_partial_fields(self) -> None:
        samples = [
            {"timestamp_s": 0.0, "power_w": 300.0},
            {"timestamp_s": 0.1, "gpu_temp_c": 55.0},
        ]
        summary = _compute_summary(samples, 5.0)
        assert summary.power_mean == 300.0
        assert summary.gpu_temp_mean == 55.0
        assert summary.sm_occupancy_mean == 0.0

    def test_energy_calculation(self) -> None:
        samples = [
            {"timestamp_s": 0.0, "power_w": 100.0},
            {"timestamp_s": 1.0, "power_w": 100.0},
        ]
        summary = _compute_summary(samples, 3600.0)  # 1 hour
        assert summary.total_energy_joules == 100.0 * 3600.0
        assert abs(summary.total_energy_wh - 100.0) < 0.01


# ─── GPUMonitor Tests ───


class TestGPUMonitor:
    def test_init(self) -> None:
        monitor = GPUMonitor(device_id=0, interval_ms=200)
        assert monitor._device_id == 0
        assert monitor._interval_ms == 200

    def test_stop_without_start(self) -> None:
        monitor = GPUMonitor()
        result = monitor.stop()
        assert result.error is not None
        assert "not started" in result.error.lower()

    def test_checkpoint_without_start(self) -> None:
        monitor = GPUMonitor()
        # Should not raise
        monitor.checkpoint("test_label")


# ─── ThrottleEvent Tests ───


class TestParseThrottleReasons:
    def test_no_throttle(self) -> None:
        assert parse_throttle_reasons(0) == []

    def test_single_reason(self) -> None:
        reasons = parse_throttle_reasons(0x0020)  # SwThermalSlowdown
        assert reasons == ["SwThermalSlowdown"]

    def test_multiple_reasons(self) -> None:
        code = 0x0004 | 0x0020  # SwPowerCap + SwThermalSlowdown
        reasons = parse_throttle_reasons(code)
        assert "SwPowerCap" in reasons
        assert "SwThermalSlowdown" in reasons

    def test_all_reasons(self) -> None:
        code = 0xFF  # all bits set
        reasons = parse_throttle_reasons(code)
        assert len(reasons) == 8

    def test_gpu_idle(self) -> None:
        reasons = parse_throttle_reasons(0x0001)
        assert reasons == ["GpuIdle"]


class TestIsCriticalThrottle:
    def test_no_throttle(self) -> None:
        assert is_critical_throttle(0) is False

    def test_gpu_idle_not_critical(self) -> None:
        assert is_critical_throttle(0x0001) is False

    def test_thermal_is_critical(self) -> None:
        assert is_critical_throttle(0x0020) is True  # SwThermalSlowdown
        assert is_critical_throttle(0x0040) is True  # HwThermalSlowdown

    def test_power_cap_is_critical(self) -> None:
        assert is_critical_throttle(0x0004) is True

    def test_hw_slowdown_is_critical(self) -> None:
        assert is_critical_throttle(0x0008) is True

    def test_critical_mask_value(self) -> None:
        expected = 0x04 | 0x08 | 0x20 | 0x40 | 0x80
        assert expected == CRITICAL_THROTTLE_MASK


class TestDetectThrottleEvent:
    def test_no_throttle(self) -> None:
        assert detect_throttle_event(0) is None

    def test_gpu_idle_only(self) -> None:
        # GpuIdle alone is not a real throttle event
        assert detect_throttle_event(0x0001) is None

    def test_critical_event(self) -> None:
        event = detect_throttle_event(0x0020, timestamp_s=5.0)
        assert event is not None
        assert event.is_critical is True
        assert "SwThermalSlowdown" in event.reasons
        assert event.timestamp_s == 5.0

    def test_non_critical_event(self) -> None:
        event = detect_throttle_event(0x0002, timestamp_s=1.0)  # ApplicationsClocks
        assert event is not None
        assert event.is_critical is False


class TestSummarizeThrottleEvents:
    def test_empty(self) -> None:
        assert summarize_throttle_events([]) == {}

    def test_single_event(self) -> None:
        events = [
            ThrottleEvent(timestamp_s=0.0, reason_code=0x0020, reasons=["SwThermalSlowdown"]),
        ]
        counts = summarize_throttle_events(events)
        assert counts == {"SwThermalSlowdown": 1}

    def test_multiple_events(self) -> None:
        events = [
            ThrottleEvent(
                timestamp_s=0.0,
                reason_code=0x0020,
                reasons=["SwThermalSlowdown"],
            ),
            ThrottleEvent(
                timestamp_s=1.0,
                reason_code=0x0024,
                reasons=["SwPowerCap", "SwThermalSlowdown"],
            ),
        ]
        counts = summarize_throttle_events(events)
        assert counts["SwThermalSlowdown"] == 2
        assert counts["SwPowerCap"] == 1


class TestThrottleEvent:
    def test_defaults(self) -> None:
        e = ThrottleEvent(timestamp_s=0.0, reason_code=0)
        assert e.reasons == []
        assert e.is_critical is False


# ─── NvidiaSmi Tests ───


class TestNvidiaSmiSample:
    def test_defaults(self) -> None:
        s = NvidiaSmiSample()
        assert s.gpu_temp_c == 0.0
        assert s.power_w == 0.0

    def test_with_values(self) -> None:
        s = NvidiaSmiSample(
            gpu_temp_c=65.0,
            power_w=500.0,
            gpu_util_pct=95.0,
            mem_used_mb=40000.0,
        )
        assert s.gpu_temp_c == 65.0
        assert s.mem_used_mb == 40000.0


class TestParseFloat:
    def test_normal(self) -> None:
        assert _parse_float("42.5") == 42.5

    def test_whitespace(self) -> None:
        assert _parse_float("  42.5  ") == 42.5

    def test_na(self) -> None:
        assert _parse_float("N/A") == 0.0
        assert _parse_float("[N/A]") == 0.0
        assert _parse_float("Not Supported") == 0.0

    def test_empty(self) -> None:
        assert _parse_float("") == 0.0

    def test_invalid(self) -> None:
        assert _parse_float("abc") == 0.0

    def test_integer(self) -> None:
        assert _parse_float("100") == 100.0


# ─── Module Export Tests ───


class TestModuleExports:
    def test_profiling_module_imports(self) -> None:
        from vlm_ocr_bench.profiling import (
            GPUMonitor,
            NvidiaSmiSample,
            ProfilingResult,
            ProfilingSummary,
            ThrottleEvent,
            detect_throttle_event,
            is_critical_throttle,
            is_nvidia_smi_available,
            parse_throttle_reasons,
            query_nvidia_smi,
            summarize_throttle_events,
        )

        assert GPUMonitor is not None
        assert ProfilingResult is not None
        assert ProfilingSummary is not None
        assert ThrottleEvent is not None
        assert detect_throttle_event is not None
        assert is_critical_throttle is not None
        assert parse_throttle_reasons is not None
        assert summarize_throttle_events is not None
        assert NvidiaSmiSample is not None
        assert query_nvidia_smi is not None
        assert is_nvidia_smi_available is not None
