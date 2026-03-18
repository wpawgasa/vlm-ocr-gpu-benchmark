"""HuggingFace model loading with quantization and dtype handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

from vlm_ocr_bench.config.schema import ModelConfig, PrecisionMode

if TYPE_CHECKING:
    from vlm_ocr_bench.config.schema import GPUConfig

logger = structlog.get_logger()


@dataclass
class LoadedModel:
    """Container for a loaded model, tokenizer, and optional processor."""

    model: Any  # transformers.PreTrainedModel
    tokenizer: Any  # transformers.PreTrainedTokenizer
    processor: Any | None = None  # transformers.ProcessorMixin


def _get_torch_dtype(precision: PrecisionMode) -> Any:
    """Map precision mode to torch dtype. Import torch at call time."""
    import torch

    dtype_map = {
        PrecisionMode.BF16: torch.bfloat16,
        PrecisionMode.FP16: torch.float16,
        PrecisionMode.FP8: torch.bfloat16,  # load in bf16, quantize separately
        PrecisionMode.FP4: torch.bfloat16,
        PrecisionMode.NVFP4: torch.bfloat16,
    }
    return dtype_map.get(precision, torch.bfloat16)


def load_model_and_tokenizer(
    model_config: ModelConfig,
    gpu_config: GPUConfig,
) -> LoadedModel:
    """Load a HuggingFace model, tokenizer, and processor.

    All transformers/torch imports happen inside this function so the module
    can be imported without GPU dependencies (enabling GPU-free unit tests).
    """
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    from vlm_ocr_bench.models.quantization import get_quantization_config

    model_id = model_config.hf_model_id
    precision = gpu_config.precision_modes[0] if gpu_config.precision_modes else PrecisionMode.BF16
    torch_dtype = _get_torch_dtype(precision)

    logger.info(
        "loading_model",
        model_id=model_id,
        precision=precision.value,
        dtype=str(torch_dtype),
    )

    # Build quantization kwargs
    quant_config = get_quantization_config(precision)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "device_map": "auto",
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Try loading processor (for vision models)
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        logger.debug("no_processor_found", model_id=model_id)

    logger.info("model_loaded", model_id=model_id)
    return LoadedModel(model=model, tokenizer=tokenizer, processor=processor)
