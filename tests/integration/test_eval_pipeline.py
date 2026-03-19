"""Integration tests for the evaluation pipeline (GPU + data required).

Run with: pytest tests/integration/test_eval_pipeline.py -v --timeout=600
"""

from __future__ import annotations

import pytest

from vlm_ocr_bench.config.schema import (
    GPUConfig,
    GPUType,
    ModelConfig,
    PrecisionMode,
    QualityConfig,
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
def minimal_quality() -> QualityConfig:
    return QualityConfig(
        benchmarks=["omnidocbench_v1.5"],
        precision_modes=[PrecisionMode.BF16],
        max_samples=3,
    )


class TestEvalPipeline:
    """Integration tests for quality evaluation — requires GPU + datasets."""

    def test_evaluation_runner_completes(
        self,
        paddle_model_config: ModelConfig,
        h100_config: GPUConfig,
        minimal_quality: QualityConfig,
    ) -> None:
        from vlm_ocr_bench.evaluation.runner import QualityEvalRunner

        runner = QualityEvalRunner(paddle_model_config, h100_config, minimal_quality)
        result = runner.run()

        assert result.model_name == "paddleocr_vl_0.9b"
        assert result.total_wall_time_s > 0
