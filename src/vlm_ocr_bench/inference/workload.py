"""Workload generator for inference benchmarking."""

from __future__ import annotations

from typing import Any

import structlog
from PIL import Image, ImageDraw

from vlm_ocr_bench.config.schema import OutputFormat
from vlm_ocr_bench.data.image_utils import preprocess_image
from vlm_ocr_bench.models.adapters.base import OCRModelAdapter

logger = structlog.get_logger()


def generate_synthetic_image(resolution: int) -> Image.Image:
    """Generate a synthetic document-like image for benchmarking.

    Draws deterministic horizontal text-line bands so the vision encoder
    sees edges and contrast rather than a blank white image, giving more
    representative GPU utilisation.  Real benchmarking should use actual
    document images from the dataset.
    """
    width = resolution
    height = int(resolution * 1.414)
    img = Image.new("RGB", (width, height), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)

    margin = width // 10
    line_h = max(width // 60, 3)
    spacing = line_h * 4

    y = margin
    line_idx = 0
    while y + line_h < height - margin:
        # Every 5th line is shorter (paragraph indent / end-of-line effect)
        line_w = width - 2 * margin if line_idx % 5 != 4 else (width - 2 * margin) * 2 // 3
        draw.rectangle([margin, y, margin + line_w, y + line_h], fill=(70, 70, 70))
        y += spacing
        line_idx += 1

    return img


def build_workload_batch(
    adapter: OCRModelAdapter,
    batch_size: int,
    resolution: int,
    images: list[Image.Image] | None = None,
    output_format: OutputFormat = OutputFormat.MARKDOWN,
    requires_padding: bool = False,
) -> list[dict[str, Any]]:
    """Build a batch of vLLM-compatible inputs for benchmarking.

    Args:
        adapter: Model adapter for prompt construction.
        batch_size: Number of inputs in the batch.
        resolution: Target image resolution.
        images: Optional list of real images. If None, synthetic images are used.
        output_format: Output format for prompts.
        requires_padding: Whether to pad images to square.

    Returns list of dicts with 'prompt' and 'multi_modal_data' keys.
    """
    batch: list[dict[str, Any]] = []

    for i in range(batch_size):
        # Use provided image or generate synthetic
        image = images[i] if images and i < len(images) else generate_synthetic_image(resolution)

        # Preprocess
        processed = preprocess_image(
            image,
            target_resolution=resolution,
            requires_padding=requires_padding,
        )

        # Build prompt via adapter
        prompt_data = adapter.build_prompt(
            processed,
            output_format=output_format,
            task="full_page_parse",
        )

        # Build vLLM-compatible input
        inp: dict[str, Any] = {
            "multi_modal_data": {"image": processed},
        }
        if "prompt" in prompt_data:
            inp["prompt"] = prompt_data["prompt"]
        elif "messages" in prompt_data:
            inp["prompt"] = prompt_data["messages"]

        batch.append(inp)

    logger.debug(
        "workload_batch_built",
        batch_size=batch_size,
        resolution=resolution,
    )
    return batch
