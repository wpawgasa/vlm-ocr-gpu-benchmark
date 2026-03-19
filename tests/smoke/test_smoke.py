"""Smoke test — full end-to-end pipeline with smallest model.

Tests the complete benchmark pipeline: config loading → inference → analysis.
Uses PaddleOCR-VL 0.9B (smallest model) with minimal settings.

Run with: pytest tests/smoke/test_smoke.py -v --timeout=600
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


class TestSmokePipeline:
    """End-to-end smoke test — requires GPU."""

    def test_full_inference_pipeline(self) -> None:
        """Test: load config → run inference → aggregate → generate report."""
        from vlm_ocr_bench.analysis.aggregator import (
            aggregate_inference_results,
            flatten_inference_result,
        )
        from vlm_ocr_bench.analysis.report import ReportData, ReportGenerator
        from vlm_ocr_bench.config.schema import (
            GPUConfig,
            GPUType,
            InferenceConfig,
            ModelConfig,
            PrecisionMode,
        )
        from vlm_ocr_bench.inference.runner import InferenceBenchmarkRunner

        # 1. Configure
        model_cfg = ModelConfig(
            name="paddleocr_vl_0.9b",
            hf_model_id="PaddlePaddle/PaddleOCR-VL-0.9B",
            params_billion=0.9,
            tier="ultra_compact",
            adapter="PaddleOcrVlAdapter",
            supported_resolutions=[1024],
            max_output_tokens=256,
        )
        gpu_cfg = GPUConfig(
            gpu_type=GPUType.H100_SXM,
            precision_modes=[PrecisionMode.BF16],
        )
        inference_cfg = InferenceConfig(
            batch_sizes=[1],
            resolutions=[1024],
            max_output_tokens_sweep=[256],
            precision_modes=[PrecisionMode.BF16],
            warmup_requests=2,
            measurement_requests=5,
            runs_per_config=1,
            cooldown_seconds=0,
        )

        # 2. Run inference
        runner = InferenceBenchmarkRunner(model_cfg, gpu_cfg, inference_cfg)
        result = runner.run()

        assert result.model_name == "paddleocr_vl_0.9b"
        assert len(result.configs) > 0

        # 3. Aggregate
        flat = flatten_inference_result(result.model_name, result.gpu_type, result.configs)
        aggregated = aggregate_inference_results(flat)
        assert len(aggregated) > 0

        # 4. Generate report
        with tempfile.TemporaryDirectory() as tmpdir:
            data = ReportData(
                experiment_name="smoke_test",
                gpu_types=[result.gpu_type],
                model_names=[result.model_name],
                inference_results=aggregated,
            )
            gen = ReportGenerator(data, output_dir=tmpdir)
            report_path = gen.generate(filename="smoke_report.md")

            assert report_path.exists()
            content = report_path.read_text()
            assert "paddleocr_vl_0.9b" in content

    def test_cli_smoke_config(self) -> None:
        """Test: vlm-ocr-bench validate works with quick_smoke.yaml."""
        from typer.testing import CliRunner

        from vlm_ocr_bench.cli import app

        runner = CliRunner()
        smoke_config = Path("configs/experiments/quick_smoke.yaml")
        if not smoke_config.exists():
            pytest.skip("quick_smoke.yaml not found")

        result = runner.invoke(app, ["validate", str(smoke_config)])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()
