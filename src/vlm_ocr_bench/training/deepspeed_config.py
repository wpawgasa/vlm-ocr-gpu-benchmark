"""DeepSpeed configuration builder for training benchmarks."""

from __future__ import annotations

from typing import Any

import structlog

from vlm_ocr_bench.config.schema import GPUConfig, PrecisionMode, TrainingConfig

logger = structlog.get_logger()


def build_deepspeed_config(
    training_config: TrainingConfig,
    gpu_config: GPUConfig,
    micro_batch_size: int,
    precision: PrecisionMode,
) -> dict[str, Any]:
    """Build a DeepSpeed configuration dict for ZeRO Stage 2.

    Args:
        training_config: Training phase configuration.
        gpu_config: GPU hardware configuration.
        micro_batch_size: Per-device micro batch size.
        precision: Precision mode for this run.

    Returns:
        DeepSpeed config dict suitable for HF Trainer deepspeed argument.
    """
    gradient_accumulation_steps = max(1, training_config.effective_batch_size // micro_batch_size)

    ds_config: dict[str, Any] = {
        "zero_optimization": {
            "stage": training_config.deepspeed_stage or 2,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e7,
            "allgather_bucket_size": 5e7,
        },
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": 1.0,
        "steps_per_print": 100,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "wall_clock_breakdown": False,
    }

    # Precision configuration
    if precision in (PrecisionMode.BF16, PrecisionMode.FP8):
        ds_config["bf16"] = {"enabled": True}
        ds_config["fp16"] = {"enabled": False}
    elif precision == PrecisionMode.FP16:
        ds_config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        }
        ds_config["bf16"] = {"enabled": False}
    else:
        # Default to bf16
        ds_config["bf16"] = {"enabled": True}
        ds_config["fp16"] = {"enabled": False}

    logger.info(
        "deepspeed_config_built",
        stage=ds_config["zero_optimization"]["stage"],
        precision=precision.value,
        micro_batch_size=micro_batch_size,
        grad_accum=gradient_accumulation_steps,
    )

    return ds_config
