"""Workload generator for inference benchmarking."""

from __future__ import annotations

from typing import Any

import structlog
from PIL import Image

from vlm_ocr_bench.config.schema import OutputFormat
from vlm_ocr_bench.data.image_utils import preprocess_image
from vlm_ocr_bench.models.adapters.base import OCRModelAdapter

logger = structlog.get_logger()


def generate_synthetic_image(resolution: int) -> Image.Image:
    """Generate a synthetic document-like image for benchmarking.

    Creates a simple RGB image at the target resolution.
    Real benchmarking should use actual document images from the dataset.
    """
    return Image.new("RGB", (resolution, int(resolution * 1.414)), color=(255, 255, 255))


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
