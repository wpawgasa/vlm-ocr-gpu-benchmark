"""Pydantic models for all configuration types."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

# ─── Enums ───


class PrecisionMode(StrEnum):
    BF16 = "bf16"
    FP16 = "fp16"
    FP8 = "fp8"
    FP4 = "fp4"
    NVFP4 = "nvfp4"


class FlashAttnVersion(StrEnum):
    FA2 = "flash_attn_2"
    FA3 = "flash_attn_3"
    FA4 = "flash_attn_4"
    AUTO = "auto"


class OutputFormat(StrEnum):
    MARKDOWN = "markdown"
    HTML = "html"


class GPUType(StrEnum):
    H100_SXM = "h100_sxm"
    B300_SXM = "b300_sxm"


# ─── Hardware Config ───


class GPUConfig(BaseModel):
    """GPU-specific hardware configuration."""

    gpu_type: GPUType
    precision_modes: list[PrecisionMode]
    flash_attn: FlashAttnVersion = FlashAttnVersion.AUTO
    kv_cache_dtype: PrecisionMode = PrecisionMode.FP8
    tensor_parallel: int = Field(default=1, ge=1, le=8)
    lock_clocks: bool = True
    max_clock_mhz: int | None = None
    power_limit_watts: int | None = None
    cuda_version: str = "12.6"


# ─── Model Config ───


class ModelConfig(BaseModel):
    """Per-model configuration."""

    name: str
    hf_model_id: str
    params_billion: float
    tier: str
    adapter: str
    supported_resolutions: list[int]
    max_output_tokens: int = 4096
    default_prompt_key: str = "document_parse_md"
    supports_flash_attn: list[FlashAttnVersion] = []
    vision_token_estimate: str = "dynamic"
    requires_padding: bool = False


# ─── Inference Phase Config ───


class InferenceConfig(BaseModel):
    """Phase A: Inference benchmark configuration."""

    batch_sizes: list[int] = [1, 2, 4, 8, 16, 32]
    resolutions: list[int] = [1024, 1536, 2048]
    output_formats: list[OutputFormat] = [OutputFormat.MARKDOWN]
    max_output_tokens_sweep: list[int] = [2048, 4096, 8192]
    precision_modes: list[PrecisionMode] = [PrecisionMode.BF16, PrecisionMode.FP8]
    warmup_requests: int = 50
    measurement_requests: int = 200
    runs_per_config: int = 3
    cooldown_seconds: int = 60
    vllm_args: dict[str, object] = {}


# ─── Training Phase Config ───


class LoRAConfig(BaseModel):
    """LoRA fine-tuning configuration."""

    rank: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: list[str] = ["q_proj", "v_proj", "k_proj", "o_proj"]
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class TrainingConfig(BaseModel):
    """Phase B: Training benchmark configuration."""

    lora: LoRAConfig = LoRAConfig()
    optimizer: str = "adamw_torch"
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    epochs: int = 3
    effective_batch_size: int = 32
    micro_batch_sizes: list[int] = [1, 2, 4, 8, 16]
    precision_modes: list[PrecisionMode] = [PrecisionMode.BF16, PrecisionMode.FP8]
    gradient_checkpointing: bool = True
    deepspeed_stage: int | None = None
    max_image_resolution: int = 2048
    dataset: str = "docmatix_50k"
    domain_dataset: str | None = "thai_fin_legal_5k"
    warmup_steps: int = 500
    measurement_steps: int = 2000
    runs_per_config: int = 3
    early_stop_patience: int = 3


# ─── Quality Evaluation Config ───


class QualityConfig(BaseModel):
    """Phase C: Quality evaluation configuration."""

    benchmarks: list[str] = [
        "omnidocbench_v1.5",
        "olmocr_bench",
        "real5_omnidocbench",
        "thai_docs_500",
    ]
    metrics: list[str] = [
        "edit_distance",
        "bleu",
        "meteor",
        "cdm",
        "structural_table",
        "structural_formula",
    ]
    precision_modes: list[PrecisionMode] = [
        PrecisionMode.BF16,
        PrecisionMode.FP8,
        PrecisionMode.FP4,
    ]
    per_document_type: bool = True
    max_samples: int | None = None


# ─── Profiling Config ───


class ProfilingConfig(BaseModel):
    """GPU profiling and monitoring configuration."""

    enabled: bool = True
    backend: str = "dcgm"
    sample_interval_ms: int = 100
    fields: list[str] = [
        "gpu_temp",
        "gpu_power",
        "sm_active",
        "sm_occupancy",
        "tensor_active",
        "dram_active",
        "pcie_tx_bytes",
        "pcie_rx_bytes",
        "gpu_mem_used",
        "gpu_mem_total",
        "throttle_reasons",
    ]
    export_format: str = "parquet"


# ─── Top-Level Experiment Config ───


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration combining all sections."""

    name: str
    description: str = ""
    seed: int = 42
    output_dir: Path = Path("results")
    models: list[str]
    gpus: list[GPUType]
    phases: list[str] = ["inference", "training", "quality"]
    inference: InferenceConfig = InferenceConfig()
    training: TrainingConfig = TrainingConfig()
    quality: QualityConfig = QualityConfig()
    profiling: ProfilingConfig = ProfilingConfig()
    gpu_configs: dict[str, GPUConfig] = {}
    model_configs: dict[str, ModelConfig] = {}
