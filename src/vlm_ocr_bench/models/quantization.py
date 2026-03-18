"""Quantization config factories for FP8/FP4 precision modes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from vlm_ocr_bench.hardware.detector import GPUInfo

from vlm_ocr_bench.config.schema import PrecisionMode

logger = structlog.get_logger()


def get_quantization_config(
    precision: PrecisionMode,
    gpu_info: GPUInfo | None = None,
) -> dict[str, Any] | None:
    """Return a BitsAndBytes / transformers quantization config dict.

    Returns None for precisions that don't need quantization (bf16, fp16).
    All torch/transformers imports are deferred to keep module GPU-free at import time.
    """
    if precision in (PrecisionMode.BF16, PrecisionMode.FP16):
        return None

    if precision == PrecisionMode.FP8:
        logger.info("quantization_config", precision=precision.value)
        return {
            "load_in_8bit": True,
            "llm_int8_threshold": 6.0,
        }

    if precision in (PrecisionMode.FP4, PrecisionMode.NVFP4):
        logger.info("quantization_config", precision=precision.value)
        return {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }

    return None


def validate_quantization(
    precision: PrecisionMode,
    gpu_info: GPUInfo,
) -> list[str]:
    """Validate that the GPU supports the requested precision.

    Returns list of issues. Empty means all checks pass.
    """
    issues: list[str] = []

    if precision == PrecisionMode.FP8 and not gpu_info.supports_fp8:
        issues.append(
            f"GPU {gpu_info.name} (CC {gpu_info.compute_capability_str}) "
            f"does not support FP8 (requires CC >= 8.9)"
        )

    if precision in (PrecisionMode.FP4, PrecisionMode.NVFP4) and not gpu_info.supports_fp4:
        issues.append(
            f"GPU {gpu_info.name} (CC {gpu_info.compute_capability_str}) "
            f"does not support {precision.value} (requires CC >= 10.0, Blackwell+)"
        )

    if issues:
        logger.warning("quantization_validation_issues", precision=precision.value, issues=issues)

    return issues
