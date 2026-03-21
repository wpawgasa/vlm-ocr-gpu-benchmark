"""VLM data collator for variable-size image + text training batches."""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger()

# Label ignore index for cross-entropy loss (standard PyTorch convention)
IGNORE_INDEX = -100


class VLMDataCollator:
    """Data collator for VLM fine-tuning.

    Handles:
    - Variable-size images (padded to batch max)
    - Image + text interleaved sequences
    - Dynamic vision token counts
    - Label masking: -100 for vision/prompt tokens (only supervise text output)

    Args:
        tokenizer: HuggingFace tokenizer instance.
        max_length: Maximum sequence length for truncation.
        max_image_resolution: Maximum image dimension (longest side).
        pad_to_multiple_of: Pad sequence lengths to a multiple of this value.
    """

    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 2048,
        max_image_resolution: int = 2048,
        pad_to_multiple_of: int | None = 8,
    ) -> None:
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._max_image_resolution = max_image_resolution
        self._pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate a batch of (image, text) pairs.

        Each feature dict should contain:
        - ``input_ids``: list[int] — tokenized prompt + response
        - ``attention_mask``: list[int] — 1 for real tokens, 0 for padding
        - ``labels``: list[int] — target token IDs (-100 for masked positions)
        - ``pixel_values`` (optional): tensor — preprocessed image pixels
        - ``image_sizes`` (optional): tuple — (height, width)

        Returns dict compatible with HF Trainer:
        {
            "input_ids": Tensor,
            "attention_mask": Tensor,
            "labels": Tensor,
            "pixel_values": Tensor (optional),
            "image_sizes": Tensor (optional),
        }
        """
        import torch

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_pixel_values = []
        batch_image_sizes = []
        has_images = False

        # Find max sequence length in batch
        max_seq_len = 0
        for f in features:
            seq_len = len(f["input_ids"])
            if seq_len > max_seq_len:
                max_seq_len = seq_len

        # Truncate to max_length
        max_seq_len = min(max_seq_len, self._max_length)

        # Pad to multiple if requested
        if self._pad_to_multiple_of is not None and self._pad_to_multiple_of > 1:
            remainder = max_seq_len % self._pad_to_multiple_of
            if remainder != 0:
                max_seq_len += self._pad_to_multiple_of - remainder

        # Some VLM processors (e.g. Qwen2_5_VLProcessor) don't expose
        # pad_token_id directly — fall back to the inner tokenizer.
        pad_token_id = getattr(self._tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            inner = getattr(self._tokenizer, "tokenizer", None)
            if inner is not None:
                pad_token_id = getattr(inner, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0

        for f in features:
            input_ids = f["input_ids"][:max_seq_len]
            attention_mask = f.get("attention_mask", [1] * len(input_ids))[:max_seq_len]
            labels = f.get("labels", input_ids[:])[:max_seq_len]

            # Pad sequences
            pad_len = max_seq_len - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len
                labels = labels + [IGNORE_INDEX] * pad_len

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

            # Collect image data if present
            if "pixel_values" in f and f["pixel_values"] is not None:
                has_images = True
                batch_pixel_values.append(f["pixel_values"])
                if "image_sizes" in f:
                    batch_image_sizes.append(f["image_sizes"])

        result: dict[str, Any] = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }

        if has_images and batch_pixel_values:
            # Stack pixel values — assumes preprocessor already handles sizing
            if isinstance(batch_pixel_values[0], torch.Tensor):
                # Pad images to same spatial dimensions
                result["pixel_values"] = _pad_and_stack_images(batch_pixel_values)
            else:
                result["pixel_values"] = torch.stack(batch_pixel_values)

            if batch_image_sizes:
                result["image_sizes"] = torch.tensor(batch_image_sizes)

        return result


def mask_labels_for_prompt(
    input_ids: list[int],
    response_start_idx: int,
) -> list[int]:
    """Create labels with prompt tokens masked to IGNORE_INDEX.

    Args:
        input_ids: Full sequence of token IDs (prompt + response).
        response_start_idx: Index where the model response begins.

    Returns:
        Labels list with -100 for prompt/vision tokens.
    """
    labels = [IGNORE_INDEX] * len(input_ids)
    for i in range(response_start_idx, len(input_ids)):
        labels[i] = input_ids[i]
    return labels


def _pad_and_stack_images(images: list[Any]) -> Any:
    """Pad variable-size image tensors to batch max and stack.

    Args:
        images: List of tensors with shape (C, H, W).

    Returns:
        Stacked tensor with shape (B, C, max_H, max_W), zero-padded.
    """
    import torch

    if not images:
        return torch.empty(0)

    # Find max spatial dimensions
    max_h = max(img.shape[-2] for img in images)
    max_w = max(img.shape[-1] for img in images)

    padded = []
    for img in images:
        h, w = img.shape[-2], img.shape[-1]
        if h == max_h and w == max_w:
            padded.append(img)
        else:
            pad_h = max_h - h
            pad_w = max_w - w
            padded_img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), value=0)
            padded.append(padded_img)

    return torch.stack(padded)
