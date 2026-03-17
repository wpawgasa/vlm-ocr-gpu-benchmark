"""Hardware abstraction: GPU detection, profiles, power monitoring, and clock management."""

from vlm_ocr_bench.hardware.clock import get_clock_info, lock_gpu_clocks, reset_gpu_clocks
from vlm_ocr_bench.hardware.detector import GPUInfo, detect_gpu, validate_gpu_for_config
from vlm_ocr_bench.hardware.power import PowerReader, PowerSummary
from vlm_ocr_bench.hardware.profiles import HardwareProfile, get_profile, list_profiles

__all__ = [
    "GPUInfo",
    "HardwareProfile",
    "PowerReader",
    "PowerSummary",
    "detect_gpu",
    "get_clock_info",
    "get_profile",
    "list_profiles",
    "lock_gpu_clocks",
    "reset_gpu_clocks",
    "validate_gpu_for_config",
]
