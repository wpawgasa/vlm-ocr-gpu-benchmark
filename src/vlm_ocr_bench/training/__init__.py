"""Training benchmark (Phase B): LoRA fine-tuning with HF Trainer + PEFT."""

from vlm_ocr_bench.training.callbacks import (
    MemoryCallback,
    PowerCallback,
    ThroughputCallback,
)
from vlm_ocr_bench.training.data_collator import (
    IGNORE_INDEX,
    VLMDataCollator,
    mask_labels_for_prompt,
)
from vlm_ocr_bench.training.deepspeed_config import build_deepspeed_config
from vlm_ocr_bench.training.lora import build_peft_config, get_trainable_param_summary
from vlm_ocr_bench.training.metrics import TrainingMetrics, compute_training_metrics
from vlm_ocr_bench.training.trainer import (
    ConvergenceResult,
    TrainingBenchmarkResult,
    TrainingBenchmarkRunner,
    TrainingConfigResult,
)

__all__ = [
    "IGNORE_INDEX",
    "ConvergenceResult",
    "MemoryCallback",
    "PowerCallback",
    "ThroughputCallback",
    "TrainingBenchmarkResult",
    "TrainingBenchmarkRunner",
    "TrainingConfigResult",
    "TrainingMetrics",
    "VLMDataCollator",
    "build_deepspeed_config",
    "build_peft_config",
    "compute_training_metrics",
    "get_trainable_param_summary",
    "mask_labels_for_prompt",
]
