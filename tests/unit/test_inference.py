"""Unit tests for the inference benchmark module (no GPU, no vLLM)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PIL import Image

from vlm_ocr_bench.config.schema import (
    GPUConfig,
    GPUType,
    InferenceConfig,
    ModelConfig,
    PrecisionMode,
)
from vlm_ocr_bench.inference.engine import EngineInfo, GenerationResult, MemoryStats, VLLMEngine
from vlm_ocr_bench.inference.metrics import (
    InferenceMetrics,
    compute_latency_percentiles,
    compute_metrics,
    compute_power_efficiency,
    compute_throughput,
)
from vlm_ocr_bench.inference.runner import (
    InferenceBenchmarkResult,
    SingleConfigResult,
)
from vlm_ocr_bench.inference.vllm_config import (
    build_vllm_engine_args,
    validate_vllm_config,
)
from vlm_ocr_bench.inference.workload import build_workload_batch, generate_synthetic_image

# ─── Fixtures ───


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        name="dots_ocr_1.5_3b",
        hf_model_id="rednote-hilab/dots.ocr-1.5",
        params_billion=3.0,
        tier="compact",
        adapter="DotsOcrAdapter",
        supported_resolutions=[1024, 1536, 2048],
        max_output_tokens=8192,
    )


@pytest.fixture
def h100_gpu_config() -> GPUConfig:
    return GPUConfig(
        gpu_type=GPUType.H100_SXM,
        precision_modes=[PrecisionMode.BF16, PrecisionMode.FP8],
    )


@pytest.fixture
def b300_gpu_config() -> GPUConfig:
    return GPUConfig(
        gpu_type=GPUType.B300_SXM,
        precision_modes=[PrecisionMode.BF16, PrecisionMode.FP8, PrecisionMode.FP4],
    )


@pytest.fixture
def inference_config() -> InferenceConfig:
    return InferenceConfig(
        batch_sizes=[1, 2],
        resolutions=[1024],
        max_output_tokens_sweep=[256],
        precision_modes=[PrecisionMode.BF16],
        warmup_requests=0,
        measurement_requests=2,
        runs_per_config=1,
        cooldown_seconds=0,
    )


# ─── EngineInfo Tests ───


class TestEngineInfo:
    def test_defaults(self) -> None:
        info = EngineInfo()
        assert info.model_load_time_s == 0.0
        assert info.gpu_memory_allocated_gb == 0.0
        assert info.flash_attn_version == "unknown"

    def test_with_values(self) -> None:
        info = EngineInfo(
            model_load_time_s=5.2,
            gpu_memory_allocated_gb=12.5,
            max_model_len=16384,
        )
        assert info.model_load_time_s == 5.2
        assert info.max_model_len == 16384


# ─── GenerationResult Tests ───


class TestGenerationResult:
    def test_defaults(self) -> None:
        r = GenerationResult()
        assert r.output_text == ""
        assert r.num_output_tokens == 0
        assert r.tokens_per_second == 0.0

    def test_with_values(self) -> None:
        r = GenerationResult(
            output_text="# Title",
            num_output_tokens=50,
            ttft_ms=10.5,
            total_time_ms=100.0,
            tokens_per_second=500.0,
        )
        assert r.output_text == "# Title"
        assert r.tokens_per_second == 500.0


# ─── MemoryStats Tests ───


class TestMemoryStats:
    def test_defaults(self) -> None:
        m = MemoryStats()
        assert m.allocated_gb == 0.0
        assert m.reserved_gb == 0.0
        assert m.peak_allocated_gb == 0.0


# ─── Metrics Tests ───


class TestComputeLatencyPercentiles:
    def test_empty(self) -> None:
        assert compute_latency_percentiles([]) == (0.0, 0.0, 0.0)

    def test_single_value(self) -> None:
        p50, p95, p99 = compute_latency_percentiles([10.0])
        assert p50 == 10.0
        assert p95 == 10.0
        assert p99 == 10.0

    def test_multiple_values(self) -> None:
        latencies = list(range(1, 101))  # 1..100
        p50, p95, p99 = compute_latency_percentiles([float(x) for x in latencies])
        assert 49.0 <= p50 <= 51.0
        assert 94.0 <= p95 <= 96.0
        assert 98.0 <= p99 <= 100.0


class TestComputeThroughput:
    def test_zero_time(self) -> None:
        assert compute_throughput(10, 500, 0.0) == (0.0, 0.0)

    def test_normal(self) -> None:
        pps, tps = compute_throughput(10, 500, 5.0)
        assert pps == 2.0
        assert tps == 100.0


class TestComputePowerEfficiency:
    def test_zero_power(self) -> None:
        assert compute_power_efficiency(2.0, 100.0, 0.0) == (0.0, 0.0, 0.0)

    def test_zero_throughput(self) -> None:
        assert compute_power_efficiency(0.0, 0.0, 500.0) == (0.0, 0.0, 0.0)

    def test_normal(self) -> None:
        epw, ppwh, tpwh = compute_power_efficiency(2.0, 100.0, 500.0)
        # energy_per_page = 500 * 0.5 / 3600 ≈ 0.0694 Wh
        assert 0.068 <= epw <= 0.071
        # pages_per_watt_hour = 2.0 * 3600 / 500 = 14.4
        assert 14.3 <= ppwh <= 14.5
        # tokens_per_watt_hour = 100.0 * 3600 / 500 = 720
        assert 719 <= tpwh <= 721


class TestComputeMetrics:
    def test_full_computation(self) -> None:
        metrics = compute_metrics(
            per_request_latencies_ms=[10.0, 20.0, 30.0, 40.0, 50.0],
            per_request_ttft_ms=[1.0, 2.0, 3.0, 4.0, 5.0],
            total_output_tokens=500,
            wall_time_s=5.0,
            mean_power_watts=400.0,
        )
        assert metrics.pages_per_second == 1.0
        assert metrics.tokens_per_second == 100.0
        assert metrics.e2e_latency_p50 == 30.0
        assert metrics.ttft_p50 == 3.0
        assert metrics.mean_power_watts == 400.0
        assert metrics.energy_per_page_wh > 0

    def test_empty_latencies(self) -> None:
        metrics = compute_metrics(
            per_request_latencies_ms=[],
            per_request_ttft_ms=[],
            total_output_tokens=0,
            wall_time_s=0.0,
        )
        assert metrics.pages_per_second == 0.0
        assert metrics.e2e_latency_p50 == 0.0


class TestInferenceMetrics:
    def test_defaults(self) -> None:
        m = InferenceMetrics()
        assert m.ttft_p50 == 0.0
        assert m.pages_per_second == 0.0
        assert m.per_request_latencies_ms == []
        assert m.throttle_events == 0


# ─── vLLM Config Tests ───


class TestBuildVllmEngineArgs:
    def test_bf16_h100(self, model_config: ModelConfig, h100_gpu_config: GPUConfig) -> None:
        adapter = MagicMock()
        adapter.get_vllm_kwargs.return_value = {"trust_remote_code": True}

        args = build_vllm_engine_args(model_config, h100_gpu_config, PrecisionMode.BF16, adapter)
        assert args["model"] == "rednote-hilab/dots.ocr-1.5"
        assert args["dtype"] == "bfloat16"
        assert args["trust_remote_code"] is True
        assert args["gpu_memory_utilization"] == 0.90
        assert args["enforce_eager"] is False

    def test_fp8_h100(self, model_config: ModelConfig, h100_gpu_config: GPUConfig) -> None:
        adapter = MagicMock()
        adapter.get_vllm_kwargs.return_value = {}

        args = build_vllm_engine_args(model_config, h100_gpu_config, PrecisionMode.FP8, adapter)
        assert args["quantization"] == "fp8"
        assert args["kv_cache_dtype"] == "fp8"

    def test_fp4_b300(self, model_config: ModelConfig, b300_gpu_config: GPUConfig) -> None:
        adapter = MagicMock()
        adapter.get_vllm_kwargs.return_value = {}

        args = build_vllm_engine_args(model_config, b300_gpu_config, PrecisionMode.FP4, adapter)
        assert args["quantization"] == "fp4"

    def test_nvfp4_b300(self, model_config: ModelConfig, b300_gpu_config: GPUConfig) -> None:
        adapter = MagicMock()
        adapter.get_vllm_kwargs.return_value = {}

        args = build_vllm_engine_args(model_config, b300_gpu_config, PrecisionMode.NVFP4, adapter)
        assert args["quantization"] == "nvfp4"

    def test_adapter_kwargs_merged(
        self, model_config: ModelConfig, h100_gpu_config: GPUConfig
    ) -> None:
        adapter = MagicMock()
        adapter.get_vllm_kwargs.return_value = {
            "trust_remote_code": True,
            "max_model_len": 32768,
        }

        args = build_vllm_engine_args(model_config, h100_gpu_config, PrecisionMode.BF16, adapter)
        # Adapter max_model_len should win since it's larger
        assert args["max_model_len"] == 32768

    def test_tensor_parallel(self, model_config: ModelConfig) -> None:
        gpu_config = GPUConfig(
            gpu_type=GPUType.H100_SXM,
            precision_modes=[PrecisionMode.BF16],
            tensor_parallel=4,
        )
        adapter = MagicMock()
        adapter.get_vllm_kwargs.return_value = {}

        args = build_vllm_engine_args(model_config, gpu_config, PrecisionMode.BF16, adapter)
        assert args["tensor_parallel_size"] == 4


class TestValidateVllmConfig:
    def test_valid_bf16(self, model_config: ModelConfig, h100_gpu_config: GPUConfig) -> None:
        issues = validate_vllm_config(model_config, h100_gpu_config, PrecisionMode.BF16)
        assert issues == []

    def test_fp4_on_h100(self, model_config: ModelConfig, h100_gpu_config: GPUConfig) -> None:
        issues = validate_vllm_config(model_config, h100_gpu_config, PrecisionMode.FP4)
        assert len(issues) == 1
        assert "Blackwell" in issues[0]

    def test_nvfp4_on_h100(self, model_config: ModelConfig, h100_gpu_config: GPUConfig) -> None:
        issues = validate_vllm_config(model_config, h100_gpu_config, PrecisionMode.NVFP4)
        assert len(issues) == 1

    def test_fp4_on_b300(self, model_config: ModelConfig, b300_gpu_config: GPUConfig) -> None:
        issues = validate_vllm_config(model_config, b300_gpu_config, PrecisionMode.FP4)
        assert issues == []

    def test_no_resolutions(self, h100_gpu_config: GPUConfig) -> None:
        cfg = ModelConfig(
            name="test",
            hf_model_id="test/test",
            params_billion=1.0,
            tier="compact",
            adapter="TestAdapter",
            supported_resolutions=[],
        )
        issues = validate_vllm_config(cfg, h100_gpu_config, PrecisionMode.BF16)
        assert len(issues) == 1
        assert "resolutions" in issues[0].lower()


# ─── Workload Tests ───


class TestGenerateSyntheticImage:
    def test_basic(self) -> None:
        img = generate_synthetic_image(1024)
        assert img.mode == "RGB"
        assert img.size[0] == 1024

    def test_aspect_ratio(self) -> None:
        img = generate_synthetic_image(512)
        # Should be portrait (A4-ish ratio)
        assert img.size[1] > img.size[0]


class TestBuildWorkloadBatch:
    def test_basic_batch(self) -> None:
        adapter = MagicMock()
        adapter.build_prompt.return_value = {
            "prompt": "Parse this document.",
            "images": [],
        }

        batch = build_workload_batch(
            adapter=adapter,
            batch_size=3,
            resolution=1024,
        )
        assert len(batch) == 3
        assert "prompt" in batch[0]
        assert "multi_modal_data" in batch[0]

    def test_with_real_images(self) -> None:
        adapter = MagicMock()
        adapter.build_prompt.return_value = {"prompt": "test", "images": []}
        images = [Image.new("RGB", (800, 600)) for _ in range(2)]

        batch = build_workload_batch(
            adapter=adapter,
            batch_size=2,
            resolution=1024,
            images=images,
        )
        assert len(batch) == 2

    def test_chat_style_prompt(self) -> None:
        messages = [{"role": "user", "content": "Parse."}]
        adapter = MagicMock()
        adapter.build_prompt.return_value = {"messages": messages, "images": []}

        batch = build_workload_batch(adapter=adapter, batch_size=1, resolution=1024)
        assert batch[0]["prompt"] == messages


# ─── Engine Tests ───


class TestVLLMEngine:
    def test_init(self) -> None:
        engine = VLLMEngine({"model": "test/model"})
        assert engine._engine is None

    def test_generate_without_init_raises(self) -> None:
        engine = VLLMEngine({"model": "test/model"})
        with pytest.raises(RuntimeError, match="not initialized"):
            engine.generate_batch([{"prompt": "test"}])

    def test_shutdown_without_init(self) -> None:
        engine = VLLMEngine({"model": "test/model"})
        # Should not raise
        engine.shutdown()

    def test_memory_stats_no_gpu(self) -> None:
        engine = VLLMEngine({"model": "test/model"})
        stats = engine.get_memory_stats()
        assert isinstance(stats, MemoryStats)


# ─── Runner Tests ───


class TestSingleConfigResult:
    def test_defaults(self) -> None:
        r = SingleConfigResult(
            resolution=1024,
            batch_size=1,
            precision="bf16",
            max_output_tokens=4096,
            run_id=0,
            metrics=InferenceMetrics(),
        )
        assert r.oom is False
        assert r.error is None
        assert r.wall_time_s == 0.0

    def test_oom_result(self) -> None:
        r = SingleConfigResult(
            resolution=2048,
            batch_size=32,
            precision="fp8",
            max_output_tokens=8192,
            run_id=0,
            metrics=InferenceMetrics(),
            oom=True,
            error="CUDA out of memory",
        )
        assert r.oom is True
        assert r.error is not None
        assert "out of memory" in r.error


class TestInferenceBenchmarkResult:
    def test_defaults(self) -> None:
        r = InferenceBenchmarkResult(
            model_name="test_model",
            gpu_type="h100_sxm",
        )
        assert r.configs == []
        assert r.engine_info is None
        assert r.total_wall_time_s == 0.0


# ─── Module Export Tests ───


class TestModuleExports:
    def test_inference_module_imports(self) -> None:
        from vlm_ocr_bench.inference import (
            InferenceBenchmarkRunner,
            VLLMEngine,
        )

        assert VLLMEngine is not None
        assert InferenceBenchmarkRunner is not None
