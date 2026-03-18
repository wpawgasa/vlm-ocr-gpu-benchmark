"""Image preprocessing utilities for document OCR benchmarking."""

from __future__ import annotations

from typing import Any

import structlog
from PIL import Image

logger = structlog.get_logger()


def preprocess_image(
    image: Image.Image,
    target_resolution: int,
    model_name: str | None = None,
    requires_padding: bool = False,
) -> Image.Image:
    """Resize document image to target resolution, preserving aspect ratio.

    Strategy:
    - Convert to RGB (drop alpha channel)
    - Resize longest side to target_resolution
    - Pad to square if model requires it
    - No augmentation (deterministic benchmarking)
    """
    # Convert to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize: longest side to target_resolution
    w, h = image.size
    if max(w, h) != target_resolution:
        scale = target_resolution / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Pad to square if required by model
    if requires_padding:
        w, h = image.size
        if w != h:
            size = max(w, h)
            padded = Image.new("RGB", (size, size), (255, 255, 255))
            padded.paste(image, ((size - w) // 2, (size - h) // 2))
            image = padded

    return image


def get_image_stats(image: Image.Image) -> dict[str, Any]:
    """Return basic image statistics.

    Returns dict with: width, height, channels, mode.
    """
    w, h = image.size
    channels = len(image.getbands())
    return {
        "width": w,
        "height": h,
        "channels": channels,
        "mode": image.mode,
    }
