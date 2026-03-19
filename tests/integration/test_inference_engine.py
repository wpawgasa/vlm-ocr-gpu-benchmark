"""Integration tests for the inference engine (GPU required).

These tests require a CUDA GPU and will download model weights on first run.
Run with: pytest tests/integration/test_inference_engine.py -v --timeout=600
"""

from __future__ import annotations

import pytest

from vlm_ocr_bench.config.schema import (
    GPUConfig,
    GPUType,
    InferenceConfig,
    ModelConfig,
    PrecisionMode,
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
def minimal_inference() -> InferenceConfig:
    return InferenceConfig(
        batch_sizes=[1],
        resolutions=[1024],
        max_output_tokens_sweep=[256],
        precision_modes=[PrecisionMode.BF16],
        warmup_requests=2,
        measurement_requests=5,
        runs_per_config=1,
        cooldown_seconds=0,
    )


class TestInferenceEngine:
    """Integration tests for VLLMEngine — requires GPU."""

    def test_engine_initialize(
        self,
        paddle_model_config: ModelConfig,
        h100_config: GPUConfig,
    ) -> None:
        from vlm_ocr_bench.inference.engine import VLLMEngine
        from vlm_ocr_bench.inference.vllm_config import build_vllm_engine_args
        from vlm_ocr_bench.models.registry import get_adapter

        adapter = get_adapter(paddle_model_config.name)
        engine_args = build_vllm_engine_args(
            paddle_model_config, h100_config, PrecisionMode.BF16, adapter
        )
        engine = VLLMEngine(engine_args)

        try:
            info = engine.initialize()
            assert info.model_load_time_s > 0
            assert info.gpu_memory_allocated_gb > 0
        finally:
            engine.shutdown()

    def test_single_inference(
        self,
        paddle_model_config: ModelConfig,
        h100_config: GPUConfig,
    ) -> None:
        from vlm_ocr_bench.inference.engine import VLLMEngine
        from vlm_ocr_bench.inference.vllm_config import build_vllm_engine_args
        from vlm_ocr_bench.inference.workload import build_workload_batch
        from vlm_ocr_bench.models.registry import get_adapter

        adapter = get_adapter(paddle_model_config.name)
        engine_args = build_vllm_engine_args(
            paddle_model_config, h100_config, PrecisionMode.BF16, adapter
        )
        engine = VLLMEngine(engine_args)

        try:
            engine.initialize()
            workload = build_workload_batch(adapter=adapter, batch_size=1, resolution=1024)

            from vllm import SamplingParams

            params = SamplingParams(max_tokens=64, temperature=0.0)
            results = engine.generate_batch(workload, params)
            assert len(results) == 1
            assert len(results[0].output_text) > 0
            assert results[0].num_output_tokens > 0
        finally:
            engine.shutdown()

    def test_batch_inference(
        self,
        paddle_model_config: ModelConfig,
        h100_config: GPUConfig,
    ) -> None:
        from vlm_ocr_bench.inference.engine import VLLMEngine
        from vlm_ocr_bench.inference.vllm_config import build_vllm_engine_args
        from vlm_ocr_bench.inference.workload import build_workload_batch
        from vlm_ocr_bench.models.registry import get_adapter

        adapter = get_adapter(paddle_model_config.name)
        engine_args = build_vllm_engine_args(
            paddle_model_config, h100_config, PrecisionMode.BF16, adapter
        )
        engine = VLLMEngine(engine_args)

        try:
            engine.initialize()
            workload = build_workload_batch(adapter=adapter, batch_size=2, resolution=1024)

            from vllm import SamplingParams

            params = SamplingParams(max_tokens=64, temperature=0.0)
            results = engine.generate_batch(workload, params)
            assert len(results) == 2
            for r in results:
                assert r.num_output_tokens > 0
        finally:
            engine.shutdown()


class TestInferenceBenchmarkRunner:
    """Integration test for the full inference runner."""

    def test_runner_completes(
        self,
        paddle_model_config: ModelConfig,
        h100_config: GPUConfig,
        minimal_inference: InferenceConfig,
    ) -> None:
        from vlm_ocr_bench.inference.runner import InferenceBenchmarkRunner

        runner = InferenceBenchmarkRunner(paddle_model_config, h100_config, minimal_inference)
        result = runner.run()

        assert result.model_name == "paddleocr_vl_0.9b"
        assert len(result.configs) > 0
        assert result.total_wall_time_s > 0

        # At least one config should have valid metrics
        valid = [c for c in result.configs if not c.oom and c.error is None]
        assert len(valid) > 0
        assert valid[0].metrics.pages_per_second > 0
