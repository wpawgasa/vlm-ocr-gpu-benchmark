"""Tests for the hardware abstraction layer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vlm_ocr_bench.config.schema import FlashAttnVersion, GPUConfig, GPUType, PrecisionMode
from vlm_ocr_bench.hardware.clock import get_clock_info, lock_gpu_clocks, reset_gpu_clocks
from vlm_ocr_bench.hardware.detector import (
    GPUInfo,
    _map_device_name_to_gpu_type,
    detect_gpu,
    validate_gpu_for_config,
)
from vlm_ocr_bench.hardware.power import PowerReader, PowerSummary
from vlm_ocr_bench.hardware.profiles import get_profile, list_profiles

# ─── GPUInfo Tests ───


class TestGPUInfo:
    """Test GPUInfo dataclass."""

    def test_defaults(self) -> None:
        info = GPUInfo()
        assert info.name == "unknown"
        assert info.gpu_type is None
        assert info.vram_gb == 0.0

    def test_compute_capability_str(self) -> None:
        info = GPUInfo(compute_capability=(9, 0))
        assert info.compute_capability_str == "9.0"

    def test_supports_fp8_hopper(self) -> None:
        info = GPUInfo(compute_capability=(9, 0))
        assert info.supports_fp8 is True
        assert info.supports_fp4 is False

    def test_supports_fp4_blackwell(self) -> None:
        info = GPUInfo(compute_capability=(10, 0))
        assert info.supports_fp8 is True
        assert info.supports_fp4 is True

    def test_no_fp8_ampere(self) -> None:
        info = GPUInfo(compute_capability=(8, 0))
        assert info.supports_fp8 is False
        assert info.supports_fp4 is False

    def test_frozen(self) -> None:
        info = GPUInfo()
        try:
            info.name = "changed"  # type: ignore[misc]
            raise AssertionError("Should not allow mutation")
        except AttributeError:
            pass


# ─── Device Name Mapping ───


class TestMapDeviceName:
    """Test device name to GPUType mapping."""

    def test_h100(self) -> None:
        assert _map_device_name_to_gpu_type("NVIDIA H100 SXM") == GPUType.H100_SXM

    def test_b300(self) -> None:
        assert _map_device_name_to_gpu_type("NVIDIA B300 SXM") == GPUType.B300_SXM

    def test_blackwell(self) -> None:
        assert _map_device_name_to_gpu_type("Blackwell Ultra") == GPUType.B300_SXM

    def test_unknown(self) -> None:
        assert _map_device_name_to_gpu_type("NVIDIA A100") is None

    def test_case_insensitive(self) -> None:
        assert _map_device_name_to_gpu_type("nvidia h100 sxm5") == GPUType.H100_SXM


# ─── detect_gpu Tests ───


class TestDetectGPU:
    """Test GPU detection with mocked pynvml."""

    def test_no_pynvml(self) -> None:
        """When pynvml init fails, returns defaults with package versions."""
        with patch.dict("sys.modules", {"pynvml": None}):
            info = detect_gpu()
            assert info.name == "unknown"
            assert info.gpu_type is None

    def test_detect_with_mock_pynvml(self) -> None:
        """Test detection with a fully mocked pynvml."""
        mock_pynvml = MagicMock()
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetName.return_value = "NVIDIA H100 SXM"

        mem_info = MagicMock()
        mem_info.total = 80 * (1024**3)
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mem_info

        mock_pynvml.nvmlDeviceGetCudaComputeCapability.return_value = (9, 0)
        mock_pynvml.nvmlSystemGetDriverVersion.return_value = "535.129"
        mock_pynvml.nvmlSystemGetCudaDriverVersion_v2.return_value = 12060
        mock_pynvml.nvmlDeviceGetNumGpuCores.return_value = 132
        mock_pynvml.nvmlDeviceGetMaxClockInfo.return_value = 1980
        mock_pynvml.nvmlDeviceGetPowerManagementLimit.return_value = 700000
        mock_pynvml.NVML_CLOCK_SM = 0

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            info = detect_gpu()
            assert info.name == "NVIDIA H100 SXM"
            assert info.gpu_type == GPUType.H100_SXM
            assert info.vram_gb == 80.0
            assert info.compute_capability == (9, 0)
            assert info.cuda_version == "12.6"
            assert info.num_sms == 132
            assert info.power_limit_w == 700


# ─── Validation Tests ───


class TestValidateGPU:
    """Test GPU config validation."""

    def test_h100_valid_config(self) -> None:
        gpu = GPUInfo(
            name="NVIDIA H100 SXM",
            compute_capability=(9, 0),
            vram_gb=80.0,
        )
        config = GPUConfig(
            gpu_type=GPUType.H100_SXM,
            precision_modes=[PrecisionMode.BF16, PrecisionMode.FP8],
            flash_attn=FlashAttnVersion.FA3,
        )
        issues = validate_gpu_for_config(gpu, config)
        assert issues == []

    def test_fp4_on_hopper_fails(self) -> None:
        gpu = GPUInfo(
            name="NVIDIA H100 SXM",
            compute_capability=(9, 0),
        )
        config = GPUConfig(
            gpu_type=GPUType.H100_SXM,
            precision_modes=[PrecisionMode.FP4],
        )
        issues = validate_gpu_for_config(gpu, config)
        assert len(issues) == 1
        assert "FP4" in issues[0] or "fp4" in issues[0]

    def test_nvfp4_on_hopper_fails(self) -> None:
        gpu = GPUInfo(compute_capability=(9, 0))
        config = GPUConfig(
            gpu_type=GPUType.H100_SXM,
            precision_modes=[PrecisionMode.NVFP4],
        )
        issues = validate_gpu_for_config(gpu, config)
        assert len(issues) == 1

    def test_fa4_on_hopper_fails(self) -> None:
        gpu = GPUInfo(compute_capability=(9, 0))
        config = GPUConfig(
            gpu_type=GPUType.H100_SXM,
            precision_modes=[PrecisionMode.BF16],
            flash_attn=FlashAttnVersion.FA4,
        )
        issues = validate_gpu_for_config(gpu, config)
        assert len(issues) == 1
        assert "FlashAttention 4" in issues[0]

    def test_fa3_on_ampere_fails(self) -> None:
        gpu = GPUInfo(compute_capability=(8, 0))
        config = GPUConfig(
            gpu_type=GPUType.H100_SXM,
            precision_modes=[PrecisionMode.BF16],
            flash_attn=FlashAttnVersion.FA3,
        )
        issues = validate_gpu_for_config(gpu, config)
        assert any("FlashAttention 3" in i for i in issues)

    def test_b300_all_precisions_valid(self) -> None:
        gpu = GPUInfo(compute_capability=(10, 0), vram_gb=288.0)
        config = GPUConfig(
            gpu_type=GPUType.B300_SXM,
            precision_modes=[
                PrecisionMode.BF16,
                PrecisionMode.FP8,
                PrecisionMode.FP4,
                PrecisionMode.NVFP4,
            ],
            flash_attn=FlashAttnVersion.FA4,
        )
        issues = validate_gpu_for_config(gpu, config)
        assert issues == []


# ─── Profiles Tests ───


class TestProfiles:
    """Test hardware profiles."""

    def test_h100_profile(self) -> None:
        profile = get_profile(GPUType.H100_SXM)
        assert profile.vram_gb == 80
        assert profile.tdp_w == 700
        assert profile.architecture == "Hopper"
        assert profile.bf16_tflops == 989.4
        assert profile.memory_bandwidth_tb_s == 3.35

    def test_b300_profile(self) -> None:
        profile = get_profile(GPUType.B300_SXM)
        assert profile.vram_gb == 288
        assert profile.tdp_w == 1000
        assert profile.architecture == "Blackwell"
        assert profile.fp4_tflops > 0

    def test_unknown_profile_raises(self) -> None:
        try:
            get_profile("nonexistent")  # type: ignore[arg-type]
            raise AssertionError("Should have raised KeyError")
        except KeyError:
            pass

    def test_list_profiles(self) -> None:
        profiles = list_profiles()
        assert len(profiles) == 2
        names = {p.gpu_type for p in profiles}
        assert GPUType.H100_SXM in names
        assert GPUType.B300_SXM in names


# ─── PowerSummary Tests ───


class TestPowerSummary:
    """Test PowerSummary dataclass."""

    def test_defaults(self) -> None:
        summary = PowerSummary()
        assert summary.mean_watts == 0.0
        assert summary.num_samples == 0
        assert summary.throttle_events == 0


# ─── PowerReader Tests ───


class TestPowerReader:
    """Test PowerReader (no GPU needed)."""

    def test_stop_without_start(self) -> None:
        reader = PowerReader()
        summary = reader.stop()
        assert summary.num_samples == 0
        assert summary.duration_seconds >= 0

    def test_start_without_gpu(self) -> None:
        """Start should handle missing GPU gracefully."""
        reader = PowerReader()
        reader.start()
        # Should not crash, just warn
        summary = reader.stop()
        assert isinstance(summary, PowerSummary)

    def test_get_current_watts_without_init(self) -> None:
        reader = PowerReader()
        assert reader.get_current_watts() is None


# ─── Clock Tests ───


class TestClockFunctions:
    """Test clock management functions (no GPU needed)."""

    def test_lock_clocks_no_gpu(self) -> None:
        result = lock_gpu_clocks(gpu_id=0)
        assert result is False

    def test_reset_clocks_no_gpu(self) -> None:
        result = reset_gpu_clocks(gpu_id=0)
        assert result is False

    def test_get_clock_info_no_gpu(self) -> None:
        info = get_clock_info(gpu_id=0)
        assert info == {}
