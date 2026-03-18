"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vlm_ocr_bench.config.schema import GPUConfig, GPUType, PrecisionMode


@pytest.fixture
def sample_config_path(tmp_path: Path) -> Path:
    """Create a minimal config file for testing."""
    config = tmp_path / "test_config.yaml"
    config.write_text("experiment:\n  name: test\n  description: test config\n")
    return config


@pytest.fixture
def sample_image() -> MagicMock:
    """Mock PIL Image for adapter testing."""
    img = MagicMock()
    img.size = (1024, 768)
    img.mode = "RGB"
    return img


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
