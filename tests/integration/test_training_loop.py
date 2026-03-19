"""Integration tests for the training loop (GPU required).

These tests require a CUDA GPU and will download model weights on first run.
Run with: pytest tests/integration/test_training_loop.py -v --timeout=600
"""

from __future__ import annotations

import pytest

from vlm_ocr_bench.config.schema import (
    GPUConfig,
    GPUType,
    ModelConfig,
    PrecisionMode,
    TrainingConfig,
)

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.fixture
def paddle_model_config() -> ModelConfig:
    return ModelConfig(
        name="paddleocr_vl_0.9b",
        hf_model_id="PaddlePaddle/PaddleOCR-VL-0.9B",
        params_billion=0.9,
        tier="ultra_compact",
        adapter="PaddleOcrVlAdapter",
        supported_resolutions=[1024],
        max_output_tokens=256,
    )


@pytest.fixture
def h100_config() -> GPUConfig:
    return GPUConfig(
        gpu_type=GPUType.H100_SXM,
        precision_modes=[PrecisionMode.BF16],
    )


@pytest.fixture
def minimal_training() -> TrainingConfig:
    return TrainingConfig(
        micro_batch_sizes=[1],
        precision_modes=[PrecisionMode.BF16],
        warmup_steps=1,
        measurement_steps=3,
        runs_per_config=1,
    )


class TestTrainingLoop:
    """Integration tests for training — requires GPU."""

    def test_lora_config_applies(self, paddle_model_config: ModelConfig) -> None:
        from vlm_ocr_bench.training.lora import build_peft_config

        config = TrainingConfig()
        peft_config = build_peft_config(paddle_model_config, config)

        assert "r" in peft_config
        assert "lora_alpha" in peft_config
        assert peft_config["task_type"] == "CAUSAL_LM"

    def test_training_runner_completes(
        self,
        paddle_model_config: ModelConfig,
        h100_config: GPUConfig,
        minimal_training: TrainingConfig,
    ) -> None:
        from vlm_ocr_bench.training.trainer import TrainingBenchmarkRunner

        runner = TrainingBenchmarkRunner(paddle_model_config, h100_config, minimal_training)
        result = runner.run()

        assert result.model_name == "paddleocr_vl_0.9b"
        assert result.total_wall_time_s > 0
