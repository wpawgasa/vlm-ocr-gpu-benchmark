"""Models: registry, loader, quantization, and adapter interfaces."""

from vlm_ocr_bench.models.adapters import (
    ChandraOcrAdapter,
    DeepSeekOcrAdapter,
    DocElement,
    DotsOcrAdapter,
    NanonetsOcrAdapter,
    OCRModelAdapter,
    OlmOcrAdapter,
    PaddleOcrVlAdapter,
    ParsedDocument,
)
from vlm_ocr_bench.models.registry import get_adapter, get_model_config, list_models

__all__ = [
    "ChandraOcrAdapter",
    "DeepSeekOcrAdapter",
    "DocElement",
    "DotsOcrAdapter",
    "NanonetsOcrAdapter",
    "OCRModelAdapter",
    "OlmOcrAdapter",
    "PaddleOcrVlAdapter",
    "ParsedDocument",
    "get_adapter",
    "get_model_config",
    "list_models",
]
