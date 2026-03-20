"""Shared test fixtures for all test tiers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vlm_ocr_bench.config.schema import (
    ExperimentConfig,
    GPUConfig,
    GPUType,
    InferenceConfig,
    ModelConfig,
    PrecisionMode,
    QualityConfig,
    TrainingConfig,
)
from vlm_ocr_bench.hardware.detector import GPUInfo

# ─── GPU availability ───


def _gpu_available() -> bool:
    """Check if a CUDA GPU is available."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


gpu_available = pytest.mark.skipif(
    not _gpu_available(),
    reason="No CUDA GPU available",
)
"""Marker to skip tests that require a GPU. Usage: @gpu_available"""


# ─── Config Fixtures ───


@pytest.fixture
def sample_config_path(tmp_path: Path) -> Path:
    """Create a minimal config file for testing."""
    config = tmp_path / "test_config.yaml"
    config.write_text(
        "name: test\n"
        "description: test config\n"
        "models: [paddleocr_vl_0.9b]\n"
        "gpus: [h100_sxm]\n"
        "phases: [inference]\n"
    )
    return config


@pytest.fixture
def sample_experiment_config() -> ExperimentConfig:
    """Minimal ExperimentConfig for testing."""
    return ExperimentConfig(
        name="test_experiment",
        description="Test experiment config",
        models=["paddleocr_vl_0.9b"],
        gpus=[GPUType.H100_SXM],
        phases=["inference"],
        seed=42,
    )


# ─── GPU Config Fixtures ───


@pytest.fixture
def h100_gpu_config() -> GPUConfig:
    """H100 GPU config with bf16 precision."""
    return GPUConfig(
        gpu_type=GPUType.H100_SXM,
        precision_modes=[PrecisionMode.BF16],
    )


@pytest.fixture
def b300_gpu_config() -> GPUConfig:
    """B300 GPU config with multiple precision modes."""
    return GPUConfig(
        gpu_type=GPUType.B300_SXM,
        precision_modes=[PrecisionMode.BF16, PrecisionMode.FP8, PrecisionMode.FP4],
    )


# ─── Model Config Fixtures ───


@pytest.fixture
def smallest_model_config() -> ModelConfig:
    """PaddleOCR-VL 0.9B — smallest model for fast testing."""
    return ModelConfig(
        name="paddleocr_vl_0.9b",
        hf_model_id="PaddlePaddle/PaddleOCR-VL-1.5",
        params_billion=0.9,
        tier="ultra_compact",
        adapter="PaddleOcrVlAdapter",
        supported_resolutions=[1024],
        max_output_tokens=4096,
    )


# ─── Image Fixtures ───


@pytest.fixture
def sample_image() -> MagicMock:
    """Mock PIL Image for adapter testing."""
    img = MagicMock()
    img.size = (1024, 768)
    img.mode = "RGB"
    return img


@pytest.fixture
def real_sample_image() -> object:
    """Create a real PIL Image for integration testing."""
    from PIL import Image

    return Image.new("RGB", (1024, 768), color=(255, 255, 255))


# ─── GPU Info Fixtures ───


@pytest.fixture
def mock_gpu_info() -> GPUInfo:
    """Fake GPUInfo for unit tests (no real GPU needed)."""
    return GPUInfo(
        name="Mock H100 SXM",
        gpu_type=GPUType.H100_SXM,
        compute_capability=(9, 0),
        vram_gb=80.0,
        driver_version="555.42.06",
        cuda_version="12.6",
        flash_attn_version="2.7.4",
        vllm_version="0.6.6",
        pytorch_version="2.5.0",
        num_sms=132,
        clock_mhz=1980,
        power_limit_w=700,
    )


# ─── Ground Truth Fixtures ───


@pytest.fixture
def sample_ground_truth() -> dict[str, str]:
    """Example ground truth for metric tests."""
    return {
        "s1": "# Title\n\nThis is a sample document with text content.",
        "s2": "| Column A | Column B |\n|---|---|\n| Value 1 | Value 2 |",
        "s3": "$$E = mc^2$$",
    }


# ─── Phase Config Fixtures ───


@pytest.fixture
def minimal_inference_config() -> InferenceConfig:
    """Minimal inference config for fast testing."""
    return InferenceConfig(
        batch_sizes=[1],
        resolutions=[1024],
        max_output_tokens_sweep=[256],
        precision_modes=[PrecisionMode.BF16],
        warmup_requests=0,
        measurement_requests=2,
        runs_per_config=1,
        cooldown_seconds=0,
    )


@pytest.fixture
def minimal_training_config() -> TrainingConfig:
    """Minimal training config for fast testing."""
    return TrainingConfig(
        micro_batch_sizes=[1],
        precision_modes=[PrecisionMode.BF16],
        warmup_steps=1,
        measurement_steps=2,
        runs_per_config=1,
    )


@pytest.fixture
def minimal_quality_config() -> QualityConfig:
    """Minimal quality config for fast testing."""
    return QualityConfig(
        benchmarks=["omnidocbench_v1.5"],
        precision_modes=[PrecisionMode.BF16],
        max_samples=5,
    )
