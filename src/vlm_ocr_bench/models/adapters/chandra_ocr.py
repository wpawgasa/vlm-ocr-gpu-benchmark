"""Adapter for Chandra-OCR (9B, Qwen2-VL based)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vlm_ocr_bench.models.adapters.base import OCRModelAdapter, ParsedDocument

if TYPE_CHECKING:
    from PIL import Image

    from vlm_ocr_bench.config.schema import GPUConfig, OutputFormat


class ChandraOcrAdapter(OCRModelAdapter):
    """Adapter for amaai-lab/Chandra-OCR-9B."""

    def get_model_id(self) -> str:
        return "amaai-lab/Chandra-OCR-9B"

    def build_prompt(
        self,
        image: Image.Image,
        output_format: OutputFormat = "markdown",  # type: ignore[assignment]
        task: str = "full_page_parse",
    ) -> dict[str, Any]:
        prompt_map = {
            "full_page_parse": (
                "Below is a document image. Parse the full page content into clean markdown format."
            ),
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
            "max_model_len": 16384,
        }

    def get_max_output_tokens(self) -> int:
        return 8192

    def estimate_vision_tokens(self, width: int, height: int) -> int:
        """Qwen2-VL style: dynamic resolution with 28px patches."""
        patch_size = 28
        return (width // patch_size) * (height // patch_size)
