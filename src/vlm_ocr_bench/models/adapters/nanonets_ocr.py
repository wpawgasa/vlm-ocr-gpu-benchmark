"""Adapter for Nanonets-OCR2 (3B)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vlm_ocr_bench.models.adapters.base import OCRModelAdapter, ParsedDocument

if TYPE_CHECKING:
    from PIL import Image

    from vlm_ocr_bench.config.schema import GPUConfig, OutputFormat


class NanonetsOcrAdapter(OCRModelAdapter):
    """Adapter for nanonets/Nanonets-OCR-s."""

    def get_model_id(self) -> str:
        return "nanonets/Nanonets-OCR-s"

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
            "max_model_len": 8192,
        }
