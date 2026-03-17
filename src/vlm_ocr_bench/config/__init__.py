"""Configuration system: schema, loading, and dynamic resolution."""

from vlm_ocr_bench.config.loader import (
    deep_merge,
    load_experiment_config,
    load_yaml,
    save_config_snapshot,
    save_config_yaml_copy,
)
from vlm_ocr_bench.config.schema import (
    ExperimentConfig,
    FlashAttnVersion,
    GPUConfig,
    GPUType,
    InferenceConfig,
    LoRAConfig,
    ModelConfig,
    OutputFormat,
    PrecisionMode,
    ProfilingConfig,
    QualityConfig,
    TrainingConfig,
)

__all__ = [
    "ExperimentConfig",
    "FlashAttnVersion",
    "GPUConfig",
    "GPUType",
    "InferenceConfig",
    "LoRAConfig",
    "ModelConfig",
    "OutputFormat",
    "PrecisionMode",
    "ProfilingConfig",
    "QualityConfig",
    "TrainingConfig",
    "deep_merge",
    "load_experiment_config",
    "load_yaml",
    "save_config_snapshot",
    "save_config_yaml_copy",
]
