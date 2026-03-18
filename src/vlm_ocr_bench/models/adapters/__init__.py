"""OCR model adapters — base types and all concrete implementations."""

from vlm_ocr_bench.models.adapters.base import DocElement, OCRModelAdapter, ParsedDocument
from vlm_ocr_bench.models.adapters.chandra_ocr import ChandraOcrAdapter
from vlm_ocr_bench.models.adapters.deepseek_ocr import DeepSeekOcrAdapter
from vlm_ocr_bench.models.adapters.dots_ocr import DotsOcrAdapter
from vlm_ocr_bench.models.adapters.nanonets_ocr import NanonetsOcrAdapter
from vlm_ocr_bench.models.adapters.olmocr import OlmOcrAdapter
from vlm_ocr_bench.models.adapters.paddleocr_vl import PaddleOcrVlAdapter

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
]
