"""Unit tests for the training benchmark module (no GPU, no HF Trainer)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vlm_ocr_bench.config.schema import (
    GPUConfig,
    GPUType,
    LoRAConfig,
    ModelConfig,
    PrecisionMode,
    TrainingConfig,
)
from vlm_ocr_bench.training.callbacks import (
    MemoryCallback,
    PowerCallback,
    StepMetric,
    ThroughputCallback,
)
from vlm_ocr_bench.training.data_collator import (
    IGNORE_INDEX,
    VLMDataCollator,
    mask_labels_for_prompt,
)
from vlm_ocr_bench.training.deepspeed_config import build_deepspeed_config
from vlm_ocr_bench.training.lora import (
    _detect_model_family,
    _get_target_modules,
    build_peft_config,
    get_trainable_param_summary,
)
from vlm_ocr_bench.training.metrics import (
    TrainingMetrics,
    compute_cost_estimate,
    compute_power_efficiency,
    compute_throughput,
    compute_training_metrics,
)
from vlm_ocr_bench.training.trainer import (
    ConvergenceResult,
    TrainingBenchmarkResult,
    TrainingBenchmarkRunner,
    TrainingConfigResult,
)

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
def olmocr_model_config() -> ModelConfig:
    return ModelConfig(
        name="olmocr2_7b",
        hf_model_id="allenai/olmOCR2-7B-0225-preview",
        params_billion=7.0,
        tier="midsize",
        adapter="OlmOcrAdapter",
        supported_resolutions=[1024, 1536, 2048],
        max_output_tokens=8192,
    )


@pytest.fixture
def chandra_model_config() -> ModelConfig:
    return ModelConfig(
        name="chandra_ocr_9b",
        hf_model_id="amaai-lab/Chandra-OCR-9B",
        params_billion=9.0,
        tier="midsize",
        adapter="ChandraOcrAdapter",
        supported_resolutions=[1024, 1536, 2048],
        max_output_tokens=8192,
    )


@pytest.fixture
def paddleocr_model_config() -> ModelConfig:
    return ModelConfig(
        name="paddleocr_vl_0.9b",
        hf_model_id="PaddlePaddle/PaddleOCR-VL-1.5",
        params_billion=0.9,
        tier="ultra_compact",
        adapter="PaddleOcrVlAdapter",
        supported_resolutions=[1024],
        max_output_tokens=4096,
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
def training_config() -> TrainingConfig:
    return TrainingConfig(
        micro_batch_sizes=[1, 2, 4],
        precision_modes=[PrecisionMode.BF16],
        warmup_steps=5,
        measurement_steps=10,
        runs_per_config=1,
    )


@pytest.fixture
def training_config_deepspeed() -> TrainingConfig:
    return TrainingConfig(
        micro_batch_sizes=[1, 2],
        precision_modes=[PrecisionMode.BF16],
        deepspeed_stage=2,
        warmup_steps=5,
        measurement_steps=10,
        runs_per_config=1,
    )


# ─── Metrics Tests ───


class TestComputeThroughput:
    def test_zero_time(self) -> None:
        assert compute_throughput(10, 500, 0.0) == (0.0, 0.0)

    def test_normal(self) -> None:
        sps, tps = compute_throughput(100, 5000, 10.0)
        assert sps == 10.0
        assert tps == 500.0


class TestComputePowerEfficiency:
    def test_zero_power(self) -> None:
        assert compute_power_efficiency(10.0, 0.0) == 0.0

    def test_zero_throughput(self) -> None:
        assert compute_power_efficiency(0.0, 500.0) == 0.0

    def test_normal(self) -> None:
        # 10 samples/s, 500W -> energy per sample = 500/10 / 3600 Wh
        # per 1K samples = 500/10/3600 * 1000 ≈ 13.889 Wh
        result = compute_power_efficiency(10.0, 500.0)
        assert 13.8 <= result <= 13.9


class TestComputeCostEstimate:
    def test_zero_throughput(self) -> None:
        cost, hours = compute_cost_estimate(0.0, 50000)
        assert cost == 0.0
        assert hours == 0.0

    def test_normal(self) -> None:
        # 10 sps, 50K samples = 5000s = 1.389h, cost = 1.389 * $2 ≈ $2.778
        cost, hours = compute_cost_estimate(10.0, 50000, 2.0)
        assert 1.38 <= hours <= 1.40
        assert 2.77 <= cost <= 2.79


class TestComputeTrainingMetrics:
    def test_full_computation(self) -> None:
        metrics = compute_training_metrics(
            num_samples=100,
            num_tokens=5000,
            wall_time_s=10.0,
            peak_gpu_memory_gb=20.5,
            mean_power_watts=400.0,
            loss_values=[2.0, 1.5, 1.0],
        )
        assert metrics.samples_per_second == 10.0
        assert metrics.tokens_per_second == 500.0
        assert metrics.peak_gpu_memory_gb == 20.5
        assert metrics.mean_power_watts == 400.0
        assert len(metrics.loss_values) == 3
        assert metrics.energy_per_1k_samples_wh > 0

    def test_empty(self) -> None:
        metrics = compute_training_metrics(
            num_samples=0,
            num_tokens=0,
            wall_time_s=0.0,
        )
        assert metrics.samples_per_second == 0.0
        assert metrics.tokens_per_second == 0.0
        assert metrics.energy_per_1k_samples_wh == 0.0


class TestTrainingMetrics:
    def test_defaults(self) -> None:
        m = TrainingMetrics()
        assert m.samples_per_second == 0.0
        assert m.tokens_per_second == 0.0
        assert m.loss_values == []
        assert m.gradient_norm_values == []
        assert m.peak_gpu_memory_gb == 0.0


# ─── LoRA Tests ───


class TestDetectModelFamily:
    def test_olmocr(self, olmocr_model_config: ModelConfig) -> None:
        assert _detect_model_family(olmocr_model_config) == "qwen2_vl"

    def test_chandra(self, chandra_model_config: ModelConfig) -> None:
        assert _detect_model_family(chandra_model_config) == "qwen2_vl"

    def test_paddleocr(self, paddleocr_model_config: ModelConfig) -> None:
        assert _detect_model_family(paddleocr_model_config) == "ernie"

    def test_unknown(self, model_config: ModelConfig) -> None:
        assert _detect_model_family(model_config) == "default"


class TestGetTargetModules:
    def test_explicit_config(self, model_config: ModelConfig) -> None:
        config = TrainingConfig(
            lora=LoRAConfig(target_modules=["q_proj", "v_proj"]),
        )
        modules = _get_target_modules(model_config, config)
        assert modules == ["q_proj", "v_proj"]

    def test_qwen2_vl_defaults(self, olmocr_model_config: ModelConfig) -> None:
        config = TrainingConfig(lora=LoRAConfig(target_modules=[]))
        modules = _get_target_modules(olmocr_model_config, config)
        assert "q_proj" in modules
        assert "k_proj" in modules
        assert "v_proj" in modules
        assert "o_proj" in modules

    def test_ernie_empty_for_auto(self, paddleocr_model_config: ModelConfig) -> None:
        config = TrainingConfig(lora=LoRAConfig(target_modules=[]))
        modules = _get_target_modules(paddleocr_model_config, config)
        # ERNIE not in _MODEL_TARGET_MODULES → returns empty (auto-detect)
        assert modules == []


class TestBuildPeftConfig:
    def test_basic(self, model_config: ModelConfig) -> None:
        config = TrainingConfig()
        result = build_peft_config(model_config, config)
        assert result["r"] == 64
        assert result["lora_alpha"] == 128
        assert result["lora_dropout"] == 0.05
        assert result["bias"] == "none"
        assert result["task_type"] == "CAUSAL_LM"

    def test_custom_rank(self, model_config: ModelConfig) -> None:
        config = TrainingConfig(lora=LoRAConfig(rank=32, alpha=64))
        result = build_peft_config(model_config, config)
        assert result["r"] == 32
        assert result["lora_alpha"] == 64

    def test_qwen2_target_modules(self, olmocr_model_config: ModelConfig) -> None:
        config = TrainingConfig()
        result = build_peft_config(olmocr_model_config, config)
        # Default lora config has target_modules set, so those are used
        assert "target_modules" in result

    def test_auto_detect_modules_omitted(self, paddleocr_model_config: ModelConfig) -> None:
        config = TrainingConfig(lora=LoRAConfig(target_modules=[]))
        result = build_peft_config(paddleocr_model_config, config)
        # ERNIE → auto-detect → target_modules omitted
        assert "target_modules" not in result


class TestGetTrainableParamSummary:
    def test_with_mock_model(self) -> None:
        model = MagicMock()

        def _mock_param(n: int, *, grad: bool) -> MagicMock:
            p = MagicMock(numel=MagicMock(return_value=n))
            p.requires_grad = grad
            return p

        params = [
            ("layer1.weight", _mock_param(1000, grad=True)),
            ("layer2.weight", _mock_param(2000, grad=False)),
            ("lora.layer1.weight", _mock_param(500, grad=True)),
        ]
        model.named_parameters.return_value = params

        summary = get_trainable_param_summary(model)
        assert summary["total_params"] == 3500
        assert summary["trainable_params"] == 1500
        assert 42.0 <= summary["trainable_pct"] <= 43.0
        assert "layer1" in summary["lora_params_by_module"]
        assert "lora.layer1" in summary["lora_params_by_module"]

    def test_no_trainable(self) -> None:
        model = MagicMock()
        params = [
            ("layer1.weight", MagicMock(numel=MagicMock(return_value=1000), requires_grad=False)),
        ]
        model.named_parameters.return_value = params

        summary = get_trainable_param_summary(model)
        assert summary["trainable_params"] == 0
        assert summary["trainable_pct"] == 0.0

    def test_empty_model(self) -> None:
        model = MagicMock()
        model.named_parameters.return_value = []

        summary = get_trainable_param_summary(model)
        assert summary["total_params"] == 0
        assert summary["trainable_pct"] == 0.0


# ─── DeepSpeed Config Tests ───


class TestBuildDeepspeedConfig:
    def test_bf16(
        self,
        training_config_deepspeed: TrainingConfig,
        h100_gpu_config: GPUConfig,
    ) -> None:
        config = build_deepspeed_config(
            training_config_deepspeed,
            h100_gpu_config,
            micro_batch_size=2,
            precision=PrecisionMode.BF16,
        )
        assert config["zero_optimization"]["stage"] == 2
        assert config["bf16"]["enabled"] is True
        assert config["fp16"]["enabled"] is False
        assert config["train_micro_batch_size_per_gpu"] == 2
        assert config["gradient_accumulation_steps"] == 16  # 32 // 2

    def test_fp16(
        self,
        training_config_deepspeed: TrainingConfig,
        h100_gpu_config: GPUConfig,
    ) -> None:
        config = build_deepspeed_config(
            training_config_deepspeed,
            h100_gpu_config,
            micro_batch_size=4,
            precision=PrecisionMode.FP16,
        )
        assert config["fp16"]["enabled"] is True
        assert config["bf16"]["enabled"] is False
        assert config["gradient_accumulation_steps"] == 8  # 32 // 4

    def test_gradient_accumulation(
        self,
        training_config_deepspeed: TrainingConfig,
        h100_gpu_config: GPUConfig,
    ) -> None:
        # effective_batch_size=32, micro_batch_size=8 → grad_accum=4
        config = build_deepspeed_config(
            training_config_deepspeed,
            h100_gpu_config,
            micro_batch_size=8,
            precision=PrecisionMode.BF16,
        )
        assert config["gradient_accumulation_steps"] == 4

    def test_min_gradient_accumulation(
        self,
        training_config_deepspeed: TrainingConfig,
        h100_gpu_config: GPUConfig,
    ) -> None:
        # micro_batch_size > effective_batch_size → grad_accum floors to 1
        config = build_deepspeed_config(
            training_config_deepspeed,
            h100_gpu_config,
            micro_batch_size=64,
            precision=PrecisionMode.BF16,
        )
        assert config["gradient_accumulation_steps"] == 1

    def test_fp4_falls_back_to_bf16(
        self,
        training_config_deepspeed: TrainingConfig,
        b300_gpu_config: GPUConfig,
    ) -> None:
        # FP4 is not a supported DeepSpeed precision — should silently use BF16
        config = build_deepspeed_config(
            training_config_deepspeed,
            b300_gpu_config,
            micro_batch_size=2,
            precision=PrecisionMode.FP4,
        )
        assert config["bf16"]["enabled"] is True
        assert config["fp16"]["enabled"] is False


# ─── Callbacks Tests ───


class TestThroughputCallback:
    def test_defaults(self) -> None:
        cb = ThroughputCallback()
        assert cb.get_mean_samples_per_second() == 0.0
        assert cb.get_mean_tokens_per_second() == 0.0

    def test_records_metrics(self) -> None:
        cb = ThroughputCallback()
        cb.samples_per_second.append(StepMetric(step=0, value=10.0))
        cb.samples_per_second.append(StepMetric(step=1, value=12.0))
        assert cb.get_mean_samples_per_second() == 11.0

    def test_step_lifecycle(self) -> None:
        cb = ThroughputCallback()
        state = MagicMock()
        state.global_step = 1

        cb.on_step_begin(args=None, state=state, control=None)
        # Simulate some work
        cb._step_samples = 16
        cb._step_tokens = 1024
        cb.on_step_end(args=None, state=state, control=None)

        assert len(cb.samples_per_second) == 1
        assert cb.samples_per_second[0].value > 0


class TestMemoryCallback:
    def test_defaults(self) -> None:
        cb = MemoryCallback()
        assert cb.get_peak_memory_gb() == 0.0

    def test_records_peak(self) -> None:
        cb = MemoryCallback()
        cb.peak_memory_gb.append(StepMetric(step=0, value=10.0))
        cb.peak_memory_gb.append(StepMetric(step=1, value=15.0))
        cb.peak_memory_gb.append(StepMetric(step=2, value=12.0))
        assert cb.get_peak_memory_gb() == 15.0

    def test_memory_breakdown_no_gpu(self) -> None:
        cb = MemoryCallback()
        cb._torch_available = False
        breakdown = cb.get_memory_breakdown()
        assert breakdown["model_gb"] == 0.0


class TestPowerCallback:
    def test_defaults(self) -> None:
        cb = PowerCallback()
        assert cb.get_mean_power_watts() == 0.0
        assert cb.get_peak_power_watts() == 0.0
        assert cb.get_power_summary() is None

    def test_step_without_reader(self) -> None:
        cb = PowerCallback()
        state = MagicMock()
        state.global_step = 1
        # Should not raise even without reader
        cb.on_step_end(args=None, state=state, control=None)
        assert len(cb.power_samples) == 0


# ─── Data Collator Tests ───


class TestMaskLabelsForPrompt:
    def test_basic(self) -> None:
        input_ids = [1, 2, 3, 4, 5]
        labels = mask_labels_for_prompt(input_ids, response_start_idx=3)
        assert labels == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 4, 5]

    def test_all_prompt(self) -> None:
        input_ids = [1, 2, 3]
        labels = mask_labels_for_prompt(input_ids, response_start_idx=3)
        assert labels == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]

    def test_no_prompt(self) -> None:
        input_ids = [1, 2, 3]
        labels = mask_labels_for_prompt(input_ids, response_start_idx=0)
        assert labels == [1, 2, 3]


class TestVLMDataCollator:
    def test_basic_collation(self) -> None:
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        collator = VLMDataCollator(tokenizer=tokenizer, max_length=10)

        features = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
            {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [4, 5]},
        ]

        batch = collator(features)
        assert batch["input_ids"].shape[0] == 2  # batch size
        assert batch["input_ids"].shape[1] >= 3  # padded to max
        assert batch["attention_mask"].shape == batch["input_ids"].shape
        assert batch["labels"].shape == batch["input_ids"].shape

    def test_padding(self) -> None:
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        collator = VLMDataCollator(tokenizer=tokenizer, max_length=100, pad_to_multiple_of=8)

        features = [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
                "labels": [1, 2, 3, 4, 5],
            },
        ]

        batch = collator(features)
        # 5 tokens → padded to 8 (nearest multiple of 8)
        assert batch["input_ids"].shape[1] == 8

    def test_truncation(self) -> None:
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        collator = VLMDataCollator(tokenizer=tokenizer, max_length=3, pad_to_multiple_of=None)

        features = [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
                "labels": [1, 2, 3, 4, 5],
            },
        ]

        batch = collator(features)
        assert batch["input_ids"].shape[1] == 3

    def test_label_padding_uses_ignore_index(self) -> None:
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        collator = VLMDataCollator(tokenizer=tokenizer, max_length=10, pad_to_multiple_of=None)

        features = [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [1, 2]},
            {"input_ids": [3, 4, 5], "attention_mask": [1, 1, 1], "labels": [3, 4, 5]},
        ]

        batch = collator(features)
        # First sample padded to length 3 → last label should be -100
        assert batch["labels"][0, 2].item() == IGNORE_INDEX

    def test_with_pixel_values(self) -> None:
        import torch

        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        collator = VLMDataCollator(tokenizer=tokenizer, max_length=10, pad_to_multiple_of=None)

        features = [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "labels": [1, 2, 3],
                "pixel_values": torch.randn(3, 224, 224),
                "image_sizes": (224, 224),
            },
            {
                "input_ids": [4, 5],
                "attention_mask": [1, 1],
                "labels": [4, 5],
                "pixel_values": torch.randn(3, 224, 224),
                "image_sizes": (224, 224),
            },
        ]

        batch = collator(features)
        assert "pixel_values" in batch
        assert batch["pixel_values"].shape[0] == 2
        assert "image_sizes" in batch

    def test_variable_size_images(self) -> None:
        import torch

        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        collator = VLMDataCollator(tokenizer=tokenizer, max_length=10, pad_to_multiple_of=None)

        features = [
            {
                "input_ids": [1, 2],
                "attention_mask": [1, 1],
                "labels": [1, 2],
                "pixel_values": torch.randn(3, 224, 224),
            },
            {
                "input_ids": [3, 4],
                "attention_mask": [1, 1],
                "labels": [3, 4],
                "pixel_values": torch.randn(3, 256, 192),
            },
        ]

        batch = collator(features)
        # Should be padded to max dimensions (256, 224)
        assert batch["pixel_values"].shape == (2, 3, 256, 224)

    def test_no_pad_token_fallback(self) -> None:
        tokenizer = MagicMock()
        tokenizer.pad_token_id = None

        collator = VLMDataCollator(tokenizer=tokenizer, max_length=10, pad_to_multiple_of=None)

        features = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
        ]

        # Should use 0 as fallback pad token
        batch = collator(features)
        assert batch["input_ids"].shape[0] == 1


# ─── Trainer Result Dataclass Tests ───


class TestTrainingConfigResult:
    def test_defaults(self) -> None:
        r = TrainingConfigResult(
            micro_batch_size=4,
            effective_batch_size=32,
            gradient_accumulation_steps=8,
            precision="bf16",
            run_id=0,
            metrics=TrainingMetrics(),
        )
        assert r.oom is False
        assert r.error is None
        assert r.steps_completed == 0
        assert r.wall_time_s == 0.0

    def test_oom_result(self) -> None:
        r = TrainingConfigResult(
            micro_batch_size=16,
            effective_batch_size=32,
            gradient_accumulation_steps=2,
            precision="bf16",
            run_id=0,
            metrics=TrainingMetrics(),
            oom=True,
            error="CUDA out of memory",
        )
        assert r.oom is True
        assert r.error is not None and "out of memory" in r.error


class TestConvergenceResult:
    def test_defaults(self) -> None:
        r = ConvergenceResult(precision="bf16", micro_batch_size=4)
        assert r.epochs_completed == 0
        assert r.final_train_loss == 0.0
        assert r.best_val_loss == float("inf")
        assert r.loss_curve == []

    def test_with_values(self) -> None:
        r = ConvergenceResult(
            precision="bf16",
            micro_batch_size=4,
            epochs_completed=3,
            final_train_loss=0.5,
            best_val_loss=0.4,
            best_val_epoch=2,
            loss_curve=[(0, 2.0), (100, 1.0), (200, 0.5)],
        )
        assert r.epochs_completed == 3
        assert len(r.loss_curve) == 3


class TestTrainingBenchmarkResult:
    def test_defaults(self) -> None:
        r = TrainingBenchmarkResult(
            model_name="test_model",
            gpu_type="h100_sxm",
        )
        assert r.configs == []
        assert r.convergence_run is None
        assert r.max_batch_sizes == {}
        assert r.total_wall_time_s == 0.0

    def test_with_configs(self) -> None:
        r = TrainingBenchmarkResult(
            model_name="test_model",
            gpu_type="h100_sxm",
            configs=[
                TrainingConfigResult(
                    micro_batch_size=4,
                    effective_batch_size=32,
                    gradient_accumulation_steps=8,
                    precision="bf16",
                    run_id=0,
                    metrics=TrainingMetrics(samples_per_second=10.0),
                ),
            ],
            max_batch_sizes={"bf16": 8},
        )
        assert len(r.configs) == 1
        assert r.configs[0].metrics.samples_per_second == 10.0
        assert r.max_batch_sizes["bf16"] == 8


# ─── Runner Tests ───


class TestTrainingBenchmarkRunner:
    def test_init(
        self,
        model_config: ModelConfig,
        h100_gpu_config: GPUConfig,
        training_config: TrainingConfig,
    ) -> None:
        runner = TrainingBenchmarkRunner(model_config, h100_gpu_config, training_config)
        assert runner._model_config == model_config
        assert runner._gpu_config == h100_gpu_config
        assert runner._training_config == training_config

    def test_get_lora_state_dict(self) -> None:
        import torch

        model = MagicMock()
        p1 = torch.tensor([1.0, 2.0])
        p1.requires_grad = True
        p2 = torch.tensor([3.0, 4.0])
        p2.requires_grad = False

        model.named_parameters.return_value = [
            ("lora.weight", p1),
            ("base.weight", p2),
        ]

        state = TrainingBenchmarkRunner._get_lora_state_dict(model)
        assert state is not None
        assert "lora.weight" in state
        assert "base.weight" not in state

    def test_reset_lora_weights(self) -> None:
        import torch

        model = MagicMock()
        param = torch.tensor([1.0, 2.0])
        param.requires_grad = True
        model.named_parameters.return_value = [("lora.weight", param)]

        initial_state = {"lora.weight": torch.tensor([5.0, 6.0])}
        TrainingBenchmarkRunner._reset_lora_weights(model, initial_state)

        assert torch.equal(param.data, torch.tensor([5.0, 6.0]))

    def test_cleanup_model(self) -> None:
        model = MagicMock()
        # Should not raise
        TrainingBenchmarkRunner._cleanup_model(model)


# ─── Module Export Tests ───


class TestModuleExports:
    def test_training_module_imports(self) -> None:
        from vlm_ocr_bench.training import (
            TrainingBenchmarkResult,
            TrainingBenchmarkRunner,
            TrainingMetrics,
            VLMDataCollator,
            build_deepspeed_config,
            build_peft_config,
        )

        assert TrainingBenchmarkRunner is not None
        assert TrainingBenchmarkResult is not None
        assert TrainingMetrics is not None
        assert VLMDataCollator is not None
        assert build_peft_config is not None
        assert build_deepspeed_config is not None
