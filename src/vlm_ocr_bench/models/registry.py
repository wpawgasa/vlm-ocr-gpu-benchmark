"""Model registry mapping model names to configs and adapter classes."""

from __future__ import annotations

import structlog

from vlm_ocr_bench.config.schema import FlashAttnVersion, ModelConfig
from vlm_ocr_bench.models.adapters.base import OCRModelAdapter
from vlm_ocr_bench.models.adapters.chandra_ocr import ChandraOcrAdapter
from vlm_ocr_bench.models.adapters.deepseek_ocr import DeepSeekOcrAdapter
from vlm_ocr_bench.models.adapters.dots_ocr import DotsOcrAdapter
from vlm_ocr_bench.models.adapters.nanonets_ocr import NanonetsOcrAdapter
from vlm_ocr_bench.models.adapters.olmocr import OlmOcrAdapter
from vlm_ocr_bench.models.adapters.paddleocr_vl import PaddleOcrVlAdapter

logger = structlog.get_logger()

_ADAPTER_MAP: dict[str, type[OCRModelAdapter]] = {
    "PaddleOcrVlAdapter": PaddleOcrVlAdapter,
    "DotsOcrAdapter": DotsOcrAdapter,
    "NanonetsOcrAdapter": NanonetsOcrAdapter,
    "DeepSeekOcrAdapter": DeepSeekOcrAdapter,
    "OlmOcrAdapter": OlmOcrAdapter,
    "ChandraOcrAdapter": ChandraOcrAdapter,
}

_FA_ALL = [FlashAttnVersion.FA2, FlashAttnVersion.FA3, FlashAttnVersion.FA4]

MODEL_REGISTRY: dict[str, ModelConfig] = {
    "paddleocr_vl_0.9b": ModelConfig(
        name="paddleocr_vl_0.9b",
        hf_model_id="PaddlePaddle/PaddleOCR-VL-1.5",
        params_billion=0.9,
        tier="ultra_compact",
        adapter="PaddleOcrVlAdapter",
        supported_resolutions=[1024],
        max_output_tokens=4096,
        supports_flash_attn=_FA_ALL,
    ),
    "dots_ocr_1.5_3b": ModelConfig(
        name="dots_ocr_1.5_3b",
        hf_model_id="rednote-hilab/dots.ocr-1.5",
        params_billion=3.0,
        tier="compact",
        adapter="DotsOcrAdapter",
        supported_resolutions=[1024, 1536, 2048],
        max_output_tokens=8192,
        supports_flash_attn=_FA_ALL,
    ),
    "nanonets_ocr2_3b": ModelConfig(
        name="nanonets_ocr2_3b",
        hf_model_id="nanonets/Nanonets-OCR-s",
        params_billion=3.0,
        tier="compact",
        adapter="NanonetsOcrAdapter",
        supported_resolutions=[1024, 1536, 2048],
        max_output_tokens=4096,
        supports_flash_attn=_FA_ALL,
    ),
    "deepseek_ocr_3b": ModelConfig(
        name="deepseek_ocr_3b",
        hf_model_id="deepseek-ai/DeepSeek-OCR-3B",
        params_billion=3.0,
        tier="compact",
        adapter="DeepSeekOcrAdapter",
        supported_resolutions=[1024, 1536, 2048],
        max_output_tokens=4096,
        supports_flash_attn=_FA_ALL,
    ),
    "olmocr2_7b": ModelConfig(
        name="olmocr2_7b",
        hf_model_id="allenai/olmOCR2-7B-0225-preview",
        params_billion=7.0,
        tier="midsize",
        adapter="OlmOcrAdapter",
        supported_resolutions=[1024, 1536, 2048],
        max_output_tokens=8192,
        supports_flash_attn=_FA_ALL,
    ),
    "chandra_ocr_9b": ModelConfig(
        name="chandra_ocr_9b",
        hf_model_id="amaai-lab/Chandra-OCR-9B",
        params_billion=9.0,
        tier="midsize",
        adapter="ChandraOcrAdapter",
        supported_resolutions=[1024, 1536, 2048],
        max_output_tokens=8192,
        supports_flash_attn=_FA_ALL,
    ),
}


def get_model_config(name: str) -> ModelConfig:
    """Get model config by name.

    Raises KeyError if the model name is not in the registry.
    """
    if name not in MODEL_REGISTRY:
        msg = f"Unknown model: {name!r}. Available: {list(MODEL_REGISTRY)}"
        raise KeyError(msg)
    return MODEL_REGISTRY[name]


def list_models() -> list[str]:
    """Return all registered model names."""
    return list(MODEL_REGISTRY.keys())


def get_adapter(name: str) -> OCRModelAdapter:
    """Create an adapter instance for the given model name.

    Raises KeyError if model name or adapter class is not found.
    """
    config = get_model_config(name)
    adapter_cls = _ADAPTER_MAP.get(config.adapter)
    if adapter_cls is None:
        msg = f"Unknown adapter: {config.adapter!r} for model {name!r}"
        raise KeyError(msg)
    logger.debug("adapter_created", model=name, adapter=config.adapter)
    return adapter_cls()
