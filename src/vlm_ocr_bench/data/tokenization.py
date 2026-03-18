"""Prompt construction and tokenization utilities."""

from __future__ import annotations

from typing import Any

import structlog
from PIL import Image

from vlm_ocr_bench.config.schema import OutputFormat
from vlm_ocr_bench.data.image_utils import preprocess_image
from vlm_ocr_bench.models.adapters.base import OCRModelAdapter

logger = structlog.get_logger()


PROMPT_TEMPLATES: dict[str, str] = {
    "document_parse_md": (
        "Convert this document image to Markdown format. "
        "Preserve the reading order, table structure, "
        "and mathematical formulas (in LaTeX)."
    ),
    "document_parse_html": (
        "Convert this document image to HTML format. "
        "Use <table> for tables and inline LaTeX for formulas."
    ),
    "ocr_text_only": (
        "Extract all text from this document image, preserving reading order. Output as plain text."
    ),
}


def build_inference_input(
    adapter: OCRModelAdapter,
    image: Image.Image,
    output_format: OutputFormat = OutputFormat.MARKDOWN,
    resolution: int = 1024,
    requires_padding: bool = False,
) -> dict[str, Any]:
    """Build complete inference input for a model.

    Preprocesses the image, builds prompt via adapter, and returns
    a dict compatible with vLLM's generate() interface.

    Returns:
        {
            "prompt": str | list[dict],
            "multi_modal_data": {"image": Image},
        }
    """
    # Preprocess image
    processed = preprocess_image(
        image,
        target_resolution=resolution,
        requires_padding=requires_padding,
    )

    # Build model-specific prompt
    prompt_data = adapter.build_prompt(
        processed,
        output_format=output_format,
        task="full_page_parse",
    )

    # Construct vLLM-compatible input
    result: dict[str, Any] = {
        "multi_modal_data": {"image": processed},
    }

    # Adapters return either 'prompt' (completion-style) or 'messages' (chat-style)
    if "prompt" in prompt_data:
        result["prompt"] = prompt_data["prompt"]
    elif "messages" in prompt_data:
        result["prompt"] = prompt_data["messages"]

    return result
