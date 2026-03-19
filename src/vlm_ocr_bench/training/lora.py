"""LoRA configuration builder and trainable parameter utilities."""

from __future__ import annotations

from typing import Any

import structlog

from vlm_ocr_bench.config.schema import ModelConfig, TrainingConfig

logger = structlog.get_logger()

# Model family → target modules mapping.
# Families NOT listed here (e.g. "ernie") return an empty list from
# _get_target_modules, which signals PEFT to auto-detect all Linear layers.
_MODEL_TARGET_MODULES: dict[str, list[str]] = {
    # Qwen2-VL based models (OlmOCR, Chandra)
    "qwen2_vl": ["q_proj", "k_proj", "v_proj", "o_proj"],
    # Default fallback for unrecognised model families — targets common attention projections.
    # Unlike "ernie", this returns an explicit list rather than triggering auto-detection.
    "default": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

# HF model ID substrings → model family
_MODEL_FAMILY_PATTERNS: dict[str, str] = {
    "olmocr": "qwen2_vl",
    "olm-ocr": "qwen2_vl",
    "chandra": "qwen2_vl",
    "qwen2": "qwen2_vl",
    "paddleocr": "ernie",
    "ernie": "ernie",
}


def _detect_model_family(model_config: ModelConfig) -> str:
    """Detect model family from HF model ID for target module selection."""
    model_id_lower = model_config.hf_model_id.lower()
    for pattern, family in _MODEL_FAMILY_PATTERNS.items():
        if pattern in model_id_lower:
            return family
    return "default"


def _get_target_modules(
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> list[str]:
    """Determine LoRA target modules based on model and config.

    Priority:
    1. Explicit config (training_config.lora.target_modules)
    2. Model-family-specific defaults
    3. Fallback to all Linear layers (empty list signals auto-detect)
    """
    # If user explicitly configured target modules, use those
    if training_config.lora.target_modules:
        return training_config.lora.target_modules

    family = _detect_model_family(model_config)
    if family in _MODEL_TARGET_MODULES:
        return _MODEL_TARGET_MODULES[family]

    # Fallback: return empty to signal PEFT auto-detection of all Linear layers
    return []


def build_peft_config(
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> dict[str, Any]:
    """Build a PEFT LoRA configuration dict.

    Returns a dict of kwargs for ``peft.LoraConfig()``. The caller is
    responsible for creating the actual ``LoraConfig`` object so that
    ``peft`` is only imported where GPU access is available.

    Handles model-specific target module detection:
    - Qwen2-VL based (OlmOCR, Chandra): q_proj, k_proj, v_proj, o_proj
    - ERNIE based (PaddleOCR-VL): auto-detect via empty target_modules
    - Custom VLMs: fallback to all Linear layers
    """
    lora_cfg = training_config.lora
    target_modules = _get_target_modules(model_config, training_config)
    family = _detect_model_family(model_config)

    config_dict: dict[str, Any] = {
        "r": lora_cfg.rank,
        "lora_alpha": lora_cfg.alpha,
        "lora_dropout": lora_cfg.dropout,
        "bias": lora_cfg.bias,
        "task_type": lora_cfg.task_type,
    }

    # Only set target_modules if we have explicit ones;
    # omit to let PEFT auto-detect all Linear layers
    if target_modules:
        config_dict["target_modules"] = target_modules

    logger.info(
        "peft_config_built",
        model=model_config.name,
        family=family,
        rank=lora_cfg.rank,
        alpha=lora_cfg.alpha,
        target_modules=target_modules or "auto",
    )
    return config_dict


def get_trainable_param_summary(model: Any) -> dict[str, Any]:
    """Compute trainable parameter summary for a PEFT model.

    Returns:
        {
            "total_params": int,
            "trainable_params": int,
            "trainable_pct": float,
            "lora_params_by_module": dict[str, int],
        }
    """
    total_params = 0
    trainable_params = 0
    lora_params_by_module: dict[str, int] = {}

    for name, param in model.named_parameters():
        num = param.numel()
        total_params += num
        if param.requires_grad:
            trainable_params += num
            # Group by module name (strip .weight/.bias suffix)
            module_name = name.rsplit(".", 1)[0] if "." in name else name
            lora_params_by_module[module_name] = lora_params_by_module.get(module_name, 0) + num

    trainable_pct = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0

    summary = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_pct": round(trainable_pct, 4),
        "lora_params_by_module": lora_params_by_module,
    }

    logger.info(
        "trainable_param_summary",
        total=total_params,
        trainable=trainable_params,
        pct=f"{trainable_pct:.2f}%",
    )
    return summary
