"""Adapter for PaddleOCR-VL (0.9B, ERNIE-based)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vlm_ocr_bench.models.adapters.base import OCRModelAdapter, ParsedDocument

if TYPE_CHECKING:
    from PIL import Image

    from vlm_ocr_bench.config.schema import GPUConfig, OutputFormat


class PaddleOcrVlAdapter(OCRModelAdapter):
    """Adapter for PaddlePaddle/PaddleOCR-VL-1.5."""

    def get_model_id(self) -> str:
        return "PaddlePaddle/PaddleOCR-VL-1.5"

    def build_prompt(
        self,
        image: Image.Image,
        output_format: OutputFormat = "markdown",  # type: ignore[assignment]
        task: str = "full_page_parse",
    ) -> dict[str, Any]:
        prompt_map = {
            "full_page_parse": "Parse this document image to markdown format.",
            "layout_only": "Detect the layout of this document image.",
        }
        prompt_text = prompt_map.get(task, prompt_map["full_page_parse"])
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
            "images": [image],
        }

    def parse_output(self, raw_output: str) -> ParsedDocument:
        cleaned = raw_output.strip()
        return ParsedDocument(
            raw_text=raw_output,
            markdown=cleaned,
            html=None,
            elements=[],
            parse_time_ms=0.0,
            token_count=len(raw_output.split()),
        )

    def get_vllm_kwargs(self, gpu_config: GPUConfig) -> dict[str, Any]:
        from vlm_ocr_bench.config.schema import PrecisionMode

        return {
            "trust_remote_code": True,
            "dtype": "bfloat16" if gpu_config.kv_cache_dtype == PrecisionMode.BF16 else "auto",
            "max_model_len": 4096,
        }

    def get_supported_resolutions(self) -> list[int]:
        return [1024]

    def get_max_output_tokens(self) -> int:
        return 4096

    def estimate_vision_tokens(self, width: int, height: int) -> int:
        patch_size = 14
        return (width // patch_size) * (height // patch_size)
