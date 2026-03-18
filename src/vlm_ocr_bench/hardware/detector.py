"""GPU detection and validation using pynvml."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

import structlog

from vlm_ocr_bench.config.schema import FlashAttnVersion, GPUConfig, GPUType, PrecisionMode

logger = structlog.get_logger()


@dataclass(frozen=True)
class GPUInfo:
    """Detected GPU hardware information."""

    name: str = "unknown"
    gpu_type: GPUType | None = None
    compute_capability: tuple[int, int] = (0, 0)
    vram_gb: float = 0.0
    driver_version: str = "unknown"
    cuda_version: str = "unknown"
    flash_attn_version: str = "not installed"
    vllm_version: str = "not installed"
    pytorch_version: str = "not installed"
    num_sms: int = 0
    clock_mhz: int = 0
    power_limit_w: int = 0

    @property
    def compute_capability_str(self) -> str:
        return f"{self.compute_capability[0]}.{self.compute_capability[1]}"

    @property
    def supports_fp8(self) -> bool:
        """FP8 requires compute capability >= 8.9 (Ada/Hopper+)."""
        return self.compute_capability >= (8, 9)

    @property
    def supports_fp4(self) -> bool:
        """FP4/NVFP4 requires compute capability >= 10.0 (Blackwell+)."""
        return self.compute_capability >= (10, 0)


def _get_package_version(pkg: str) -> str:
    """Get installed package version or 'not installed'."""
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "not installed"


def _map_device_name_to_gpu_type(name: str) -> GPUType | None:
    """Map NVIDIA device name string to GPUType enum."""
    lower = name.lower()
    if "b300" in lower or "blackwell" in lower:
        return GPUType.B300_SXM
    if "h100" in lower:
        return GPUType.H100_SXM
    return None


def detect_gpu(device_id: int = 0) -> GPUInfo:
    """Detect GPU hardware info using pynvml.

    Returns GPUInfo with defaults if no GPU is available.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception:
        logger.warning("pynvml_init_failed", exc_info=True)
        return GPUInfo(
            pytorch_version=_get_package_version("torch"),
            vllm_version=_get_package_version("vllm"),
            flash_attn_version=_get_package_version("flash-attn"),
        )

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")

        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_gb = mem_info.total / (1024**3)

        cc_major = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        if isinstance(cc_major, tuple):
            compute_capability = cc_major
        else:
            cc_minor = 0
            compute_capability = (cc_major, cc_minor)

        driver_version = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver_version, bytes):
            driver_version = driver_version.decode("utf-8")

        cuda_version_raw = pynvml.nvmlSystemGetCudaDriverVersion_v2()
        cuda_major = cuda_version_raw // 1000
        cuda_minor = (cuda_version_raw % 1000) // 10
        cuda_version = f"{cuda_major}.{cuda_minor}"

        try:
            num_sms = pynvml.nvmlDeviceGetNumGpuCores(handle)
        except Exception:
            num_sms = 0

        try:
            clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
        except Exception:
            clock_mhz = 0

        try:
            power_limit_w = pynvml.nvmlDeviceGetPowerManagementLimit(handle) // 1000
        except Exception:
            power_limit_w = 0

        gpu_type = _map_device_name_to_gpu_type(name)

        info = GPUInfo(
            name=name,
            gpu_type=gpu_type,
            compute_capability=compute_capability,
            vram_gb=round(vram_gb, 1),
            driver_version=driver_version,
            cuda_version=cuda_version,
            flash_attn_version=_get_package_version("flash-attn"),
            vllm_version=_get_package_version("vllm"),
            pytorch_version=_get_package_version("torch"),
            num_sms=num_sms,
            clock_mhz=clock_mhz,
            power_limit_w=power_limit_w,
        )
        logger.info(
            "gpu_detected",
            name=info.name,
            gpu_type=str(info.gpu_type),
            vram_gb=info.vram_gb,
            cc=info.compute_capability_str,
        )
        return info

    except Exception:
        logger.warning("gpu_detection_failed", exc_info=True)
        return GPUInfo(
            pytorch_version=_get_package_version("torch"),
            vllm_version=_get_package_version("vllm"),
            flash_attn_version=_get_package_version("flash-attn"),
        )
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()


def validate_gpu_for_config(gpu: GPUInfo, config: GPUConfig) -> list[str]:
    """Validate that the detected GPU supports the requested config.

    Returns a list of warning/error messages. Empty list means all checks pass.
    """
    issues: list[str] = []

    # Check precision mode support
    for mode in config.precision_modes:
        if mode in (PrecisionMode.FP8,) and not gpu.supports_fp8:
            issues.append(
                f"GPU {gpu.name} (CC {gpu.compute_capability_str}) "
                f"does not support {mode.value} (requires CC >= 8.9)"
            )
        if mode in (PrecisionMode.FP4, PrecisionMode.NVFP4) and not gpu.supports_fp4:
            issues.append(
                f"GPU {gpu.name} (CC {gpu.compute_capability_str}) "
                f"does not support {mode.value} (requires CC >= 10.0, Blackwell+)"
            )

    # Check flash attention version compatibility
    if config.flash_attn == FlashAttnVersion.FA4 and gpu.compute_capability < (10, 0):
        issues.append(
            f"FlashAttention 4 requires Blackwell+ (CC >= 10.0), got {gpu.compute_capability_str}"
        )
    if config.flash_attn == FlashAttnVersion.FA3 and gpu.compute_capability < (9, 0):
        issues.append(
            f"FlashAttention 3 requires Hopper+ (CC >= 9.0), got {gpu.compute_capability_str}"
        )

    # Log issues
    if issues:
        logger.warning("gpu_config_validation_issues", issues=issues)
    else:
        logger.info("gpu_config_validated", gpu=gpu.name, config_gpu_type=config.gpu_type)

    return issues
