"""Hardware profile dataclasses with theoretical peak performance specs."""

from __future__ import annotations

from dataclasses import dataclass

from vlm_ocr_bench.config.schema import GPUType


@dataclass(frozen=True)
class HardwareProfile:
    """Theoretical peak performance specifications for a GPU."""

    gpu_type: GPUType
    name: str
    architecture: str
    compute_capability: str
    vram_gb: int
    memory_type: str
    memory_bandwidth_tb_s: float
    tdp_w: int
    # Peak TFLOPS at various precisions
    bf16_tflops: float
    fp8_tflops: float
    fp4_tflops: float = 0.0
    fp16_tflops: float = 0.0
    # SM count
    num_sms: int = 0


_PROFILES: dict[GPUType, HardwareProfile] = {
    GPUType.H100_SXM: HardwareProfile(
        gpu_type=GPUType.H100_SXM,
        name="NVIDIA H100 SXM",
        architecture="Hopper",
        compute_capability="9.0",
        vram_gb=80,
        memory_type="HBM3",
        memory_bandwidth_tb_s=3.35,
        tdp_w=700,
        bf16_tflops=989.4,
        fp16_tflops=989.4,
        fp8_tflops=1978.9,
        num_sms=132,
    ),
    GPUType.B300_SXM: HardwareProfile(
        gpu_type=GPUType.B300_SXM,
        name="NVIDIA B300 SXM",
        architecture="Blackwell",
        compute_capability="10.0",
        vram_gb=288,
        memory_type="HBM3e",
        memory_bandwidth_tb_s=8.0,
        tdp_w=1000,
        bf16_tflops=2500.0,
        fp16_tflops=2250.0,
        fp8_tflops=5000.0,
        fp4_tflops=10000.0,
        num_sms=192,
    ),
}


def get_profile(gpu_type: GPUType) -> HardwareProfile:
    """Get the hardware profile for a given GPU type.

    Raises KeyError if the GPU type is not known.
    """
    if gpu_type not in _PROFILES:
        msg = f"No hardware profile for GPU type: {gpu_type}"
        raise KeyError(msg)
    return _PROFILES[gpu_type]


def list_profiles() -> list[HardwareProfile]:
    """Return all known hardware profiles."""
    return list(_PROFILES.values())
