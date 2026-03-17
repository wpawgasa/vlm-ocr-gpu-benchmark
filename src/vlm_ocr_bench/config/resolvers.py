"""Dynamic resolution for config values (GPU detection, paths, FlashAttention)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import structlog

from vlm_ocr_bench.config.schema import FlashAttnVersion, GPUType

logger = structlog.get_logger()


def resolve_gpu_type() -> GPUType | None:
    """Auto-detect GPU type from CUDA device properties.

    Returns None if no GPU is available or the GPU is not a known type.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            logger.warning("no_gpu_detected")
            return None
        name = torch.cuda.get_device_name(0).lower()
        if "b300" in name or "blackwell" in name:
            return GPUType.B300_SXM
        if "h100" in name or "hopper" in name:
            return GPUType.H100_SXM
        logger.warning("unknown_gpu_type", device_name=name)
        return None
    except Exception:
        logger.warning("gpu_detection_failed", exc_info=True)
        return None


def resolve_flash_attn_version(gpu_type: GPUType | None) -> FlashAttnVersion:
    """Resolve 'auto' FlashAttention version based on GPU type.

    B300 (Blackwell, CC 10.0) → FA4
    H100 (Hopper, CC 9.0) → FA3
    Unknown → FA2 (safe fallback)
    """
    if gpu_type == GPUType.B300_SXM:
        return FlashAttnVersion.FA4
    if gpu_type == GPUType.H100_SXM:
        return FlashAttnVersion.FA3
    return FlashAttnVersion.FA2


def resolve_path(path: str | Path) -> Path:
    """Expand ~ and environment variables in a path, then resolve to absolute."""
    expanded = os.path.expandvars(os.path.expanduser(str(path)))
    return Path(expanded).resolve()


def resolve_config_values(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Apply all dynamic resolvers to a raw config dict.

    - Resolves output_dir path
    - Detects GPU type if not explicitly set
    - Resolves flash_attn 'auto' to concrete version per GPU config
    """
    # Resolve output_dir
    if "output_dir" in config_dict:
        config_dict["output_dir"] = str(resolve_path(config_dict["output_dir"]))

    # Resolve GPU configs
    gpu_configs = config_dict.get("gpu_configs", {})
    detected_gpu = None

    for gpu_name, gpu_cfg in gpu_configs.items():
        # Resolve flash_attn auto
        if gpu_cfg.get("flash_attn") == "auto":
            gpu_type_str = gpu_cfg.get("gpu_type", gpu_name)
            try:
                gpu_type = GPUType(gpu_type_str)
            except ValueError:
                gpu_type = None
            if detected_gpu is None:
                detected_gpu = resolve_gpu_type()
            resolved = resolve_flash_attn_version(gpu_type or detected_gpu)
            gpu_cfg["flash_attn"] = resolved.value
            logger.info(
                "resolved_flash_attn",
                gpu=gpu_name,
                version=resolved.value,
            )

    return config_dict
