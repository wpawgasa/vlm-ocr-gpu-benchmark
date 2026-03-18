"""Adapter for dots.ocr-1.5 (3B)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from vlm_ocr_bench.models.adapters.base import DocElement, OCRModelAdapter, ParsedDocument

if TYPE_CHECKING:
    from PIL import Image

    from vlm_ocr_bench.config.schema import GPUConfig, OutputFormat


class DotsOcrAdapter(OCRModelAdapter):
    """Adapter for rednote-hilab/dots.ocr-1.5."""

    _SPECIAL_TOKEN_RE = re.compile(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>")

    def get_model_id(self) -> str:
        return "rednote-hilab/dots.ocr-1.5"

    def build_prompt(
        self,
        image: Image.Image,
        output_format: OutputFormat = "markdown",  # type: ignore[assignment]
        task: str = "full_page_parse",
    ) -> dict[str, Any]:
        prompt_map = {
            "full_page_parse": (
                "<|im_start|>user\n<image>\n"
                "Parse this document to markdown.<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
            "layout_only": (
                "<|im_start|>user\n<image>\n"
                "prompt_layout_only_en<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
        }
        return {
            "prompt": prompt_map.get(task, prompt_map["full_page_parse"]),
            "images": [image],
        }

    def parse_output(self, raw_output: str) -> ParsedDocument:
        cleaned = self._strip_special_tokens(raw_output)
        elements = self._extract_elements(cleaned)
        return ParsedDocument(
            raw_text=raw_output,
            markdown=cleaned,
            html=None,
            elements=elements,
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

    def _strip_special_tokens(self, text: str) -> str:
        """Remove model-specific special tokens from output."""
        return self._SPECIAL_TOKEN_RE.sub("", text).strip()

    def _extract_elements(self, text: str) -> list[DocElement]:
        """Extract document elements from cleaned markdown output."""
        elements: list[DocElement] = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                elements.append(DocElement(type="title", content=line))
            elif line.startswith("|"):
                elements.append(DocElement(type="table", content=line))
            else:
                elements.append(DocElement(type="text", content=line))
        return elements
