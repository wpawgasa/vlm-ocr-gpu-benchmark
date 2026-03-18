"""vLLM engine argument builder with GPU-specific optimizations."""

from __future__ import annotations

from typing import Any

import structlog

from vlm_ocr_bench.config.schema import (
    GPUConfig,
    GPUType,
    ModelConfig,
    PrecisionMode,
)
from vlm_ocr_bench.models.adapters.base import OCRModelAdapter

logger = structlog.get_logger()


def build_vllm_engine_args(
    model_config: ModelConfig,
    gpu_config: GPUConfig,
    precision: PrecisionMode,
    adapter: OCRModelAdapter,
) -> dict[str, Any]:
    """Build vLLM engine arguments for a specific model + GPU + precision combo.

    Merges base args, GPU-specific optimizations, precision config,
    and adapter-specific kwargs.
    """
    # Base args
    args: dict[str, Any] = {
        "model": model_config.hf_model_id,
        "trust_remote_code": True,
        "tensor_parallel_size": gpu_config.tensor_parallel,
        "gpu_memory_utilization": 0.90,
        "max_model_len": model_config.max_output_tokens * 2,
        "enforce_eager": False,
    }

    # GPU-specific flash attention
    if gpu_config.gpu_type == GPUType.B300_SXM or gpu_config.gpu_type == GPUType.H100_SXM:
        args["enable_flashattn"] = True

    # Precision-specific config
    _apply_precision_config(args, precision, gpu_config)

    # Merge adapter-specific vLLM kwargs (trust_remote_code, max_model_len, dtype, etc.)
    adapter_kwargs = adapter.get_vllm_kwargs(gpu_config)
    for key, value in adapter_kwargs.items():
        # Adapter max_model_len overrides our default if larger
        if key == "max_model_len":
            args[key] = max(args.get(key, 0), value)
        else:
            args[key] = value

    # Merge user-provided vllm_args from config (highest priority)
    # These come from InferenceConfig.vllm_args at the runner level

    logger.info(
        "vllm_engine_args_built",
        model=model_config.name,
        gpu=gpu_config.gpu_type.value,
        precision=precision.value,
        max_model_len=args.get("max_model_len"),
    )

    return args


def _apply_precision_config(
    args: dict[str, Any],
    precision: PrecisionMode,
    gpu_config: GPUConfig,
) -> None:
    """Apply precision-specific configuration to vLLM engine args."""
    if precision == PrecisionMode.BF16:
        args["dtype"] = "bfloat16"

    elif precision == PrecisionMode.FP16:
        args["dtype"] = "float16"

    elif precision == PrecisionMode.FP8:
        args["dtype"] = "bfloat16"
        args["quantization"] = "fp8"
        args["kv_cache_dtype"] = "fp8"

    elif precision in (PrecisionMode.FP4, PrecisionMode.NVFP4):
        if gpu_config.gpu_type != GPUType.B300_SXM:
            logger.warning(
                "fp4_requires_blackwell",
                precision=precision.value,
                gpu_type=gpu_config.gpu_type.value,
            )
        args["dtype"] = "bfloat16"
        args["quantization"] = "fp4" if precision == PrecisionMode.FP4 else "nvfp4"
        args["kv_cache_dtype"] = "fp8"


def validate_vllm_config(
    model_config: ModelConfig,
    gpu_config: GPUConfig,
    precision: PrecisionMode,
) -> list[str]:
    """Validate that a vLLM configuration is viable.

    Returns list of issues. Empty means the config is valid.
    """
    issues: list[str] = []

    # FP4/NVFP4 requires Blackwell
    is_fp4 = precision in (PrecisionMode.FP4, PrecisionMode.NVFP4)
    if is_fp4 and gpu_config.gpu_type != GPUType.B300_SXM:
        issues.append(
            f"{precision.value} requires Blackwell GPU (B300), got {gpu_config.gpu_type.value}"
        )

    # Check resolution support
    if not model_config.supported_resolutions:
        issues.append(f"Model {model_config.name} has no supported resolutions")

    if issues:
        logger.warning("vllm_config_validation_issues", issues=issues)

    return issues
