"""Tests for the configuration system: schema, loader, and resolvers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlm_ocr_bench.config.loader import deep_merge, load_yaml, save_config_snapshot
from vlm_ocr_bench.config.resolvers import resolve_flash_attn_version, resolve_path
from vlm_ocr_bench.config.schema import (
    ExperimentConfig,
    FlashAttnVersion,
    GPUConfig,
    GPUType,
    InferenceConfig,
    LoRAConfig,
    ModelConfig,
    PrecisionMode,
    ProfilingConfig,
    QualityConfig,
    TrainingConfig,
)

# ─── Schema Tests ───


class TestEnums:
    """Test enum definitions."""

    def test_precision_modes(self) -> None:
        assert PrecisionMode.BF16.value == "bf16"
        assert PrecisionMode.FP8.value == "fp8"
        assert PrecisionMode.NVFP4.value == "nvfp4"

    def test_gpu_types(self) -> None:
        assert GPUType.H100_SXM.value == "h100_sxm"
        assert GPUType.B300_SXM.value == "b300_sxm"

    def test_flash_attn_versions(self) -> None:
        assert FlashAttnVersion.FA3.value == "flash_attn_3"
        assert FlashAttnVersion.AUTO.value == "auto"


class TestGPUConfig:
    """Test GPUConfig model."""

    def test_minimal(self) -> None:
        cfg = GPUConfig(
            gpu_type=GPUType.H100_SXM,
            precision_modes=[PrecisionMode.BF16, PrecisionMode.FP8],
        )
        assert cfg.gpu_type == GPUType.H100_SXM
        assert cfg.flash_attn == FlashAttnVersion.AUTO
        assert cfg.tensor_parallel == 1
        assert cfg.lock_clocks is True

    def test_tensor_parallel_bounds(self) -> None:
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            GPUConfig(
                gpu_type=GPUType.H100_SXM,
                precision_modes=[PrecisionMode.BF16],
                tensor_parallel=0,
            )


class TestModelConfig:
    """Test ModelConfig model."""

    def test_full(self) -> None:
        cfg = ModelConfig(
            name="test_model",
            hf_model_id="org/model",
            params_billion=3.0,
            tier="compact",
            adapter="TestAdapter",
            supported_resolutions=[1024, 2048],
            supports_flash_attn=[FlashAttnVersion.FA2, FlashAttnVersion.FA3],
        )
        assert cfg.max_output_tokens == 4096
        assert cfg.default_prompt_key == "document_parse_md"
        assert cfg.vision_token_estimate == "dynamic"
        assert cfg.requires_padding is False


class TestInferenceConfig:
    """Test InferenceConfig defaults."""

    def test_defaults(self) -> None:
        cfg = InferenceConfig()
        assert cfg.batch_sizes == [1, 2, 4, 8, 16, 32]
        assert cfg.warmup_requests == 50
        assert cfg.runs_per_config == 3

    def test_override(self) -> None:
        cfg = InferenceConfig(batch_sizes=[1], runs_per_config=1)
        assert cfg.batch_sizes == [1]
        assert cfg.runs_per_config == 1


class TestLoRAConfig:
    """Test LoRAConfig defaults."""

    def test_defaults(self) -> None:
        cfg = LoRAConfig()
        assert cfg.rank == 64
        assert cfg.alpha == 128
        assert cfg.target_modules == ["q_proj", "v_proj", "k_proj", "o_proj"]


class TestTrainingConfig:
    """Test TrainingConfig defaults."""

    def test_defaults(self) -> None:
        cfg = TrainingConfig()
        assert cfg.lora.rank == 64
        assert cfg.optimizer == "adamw_torch"
        assert cfg.dataset == "docmatix_50k"
        assert cfg.domain_dataset == "thai_fin_legal_5k"


class TestQualityConfig:
    """Test QualityConfig defaults."""

    def test_defaults(self) -> None:
        cfg = QualityConfig()
        assert "omnidocbench_v1.5" in cfg.benchmarks
        assert "edit_distance" in cfg.metrics
        assert cfg.per_document_type is True
        assert cfg.max_samples is None


class TestProfilingConfig:
    """Test ProfilingConfig defaults."""

    def test_defaults(self) -> None:
        cfg = ProfilingConfig()
        assert cfg.enabled is True
        assert cfg.backend == "dcgm"
        assert cfg.sample_interval_ms == 100


class TestExperimentConfig:
    """Test ExperimentConfig top-level model."""

    def test_minimal(self) -> None:
        cfg = ExperimentConfig(
            name="test",
            models=["model_a"],
            gpus=[GPUType.H100_SXM],
        )
        assert cfg.seed == 42
        assert cfg.output_dir == Path("results")
        assert cfg.phases == ["inference", "training", "quality"]

    def test_with_overrides(self) -> None:
        cfg = ExperimentConfig(
            name="test",
            models=["model_a"],
            gpus=[GPUType.H100_SXM],
            seed=123,
            inference=InferenceConfig(batch_sizes=[1, 2]),
        )
        assert cfg.seed == 123
        assert cfg.inference.batch_sizes == [1, 2]

    def test_invalid_missing_required(self) -> None:
        with pytest.raises(ValueError):
            ExperimentConfig(name="test", models=["a"])  # type: ignore[call-arg]


# ─── Loader Tests ───


class TestDeepMerge:
    """Test deep_merge function."""

    def test_simple_override(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self) -> None:
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 99, "z": 100}}
        result = deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 99, "z": 100}, "b": 3}

    def test_list_replacement(self) -> None:
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        result = deep_merge(base, override)
        assert result == {"items": [4, 5]}

    def test_no_mutation(self) -> None:
        base = {"a": {"x": 1}}
        override = {"a": {"y": 2}}
        deep_merge(base, override)
        assert base == {"a": {"x": 1}}
        assert override == {"a": {"y": 2}}

    def test_empty_base(self) -> None:
        result = deep_merge({}, {"a": 1})
        assert result == {"a": 1}

    def test_empty_override(self) -> None:
        result = deep_merge({"a": 1}, {})
        assert result == {"a": 1}


class TestLoadYaml:
    """Test YAML loading."""

    def test_load_valid(self, tmp_path: Path) -> None:
        f = tmp_path / "test.yaml"
        f.write_text("key: value\nnested:\n  a: 1\n")
        result = load_yaml(f)
        assert result == {"key": "value", "nested": {"a": 1}}

    def test_load_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.yaml"
        f.write_text("")
        result = load_yaml(f)
        assert result == {}

    def test_load_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_yaml(tmp_path / "nonexistent.yaml")

    def test_load_non_dict(self, tmp_path: Path) -> None:
        f = tmp_path / "list.yaml"
        f.write_text("- item1\n- item2\n")
        with pytest.raises(TypeError, match="Expected YAML mapping"):
            load_yaml(f)


class TestSaveConfigSnapshot:
    """Test config snapshot saving."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig(
            name="snapshot_test",
            models=["model_a"],
            gpus=[GPUType.H100_SXM],
        )
        snapshot_path = save_config_snapshot(cfg, tmp_path)
        assert snapshot_path.exists()
        with open(snapshot_path) as f:
            data = json.load(f)
        assert data["name"] == "snapshot_test"
        assert data["seed"] == 42

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        cfg = ExperimentConfig(
            name="test",
            models=["m"],
            gpus=[GPUType.H100_SXM],
        )
        nested = tmp_path / "a" / "b"
        save_config_snapshot(cfg, nested)
        assert (nested / "config_snapshot.json").exists()


class TestLoadExperimentConfig:
    """Test full experiment config loading with hierarchical merge."""

    def _create_config_tree(self, base: Path) -> Path:
        """Create a minimal configs/ tree for testing."""
        configs = base / "configs"
        (configs / "models").mkdir(parents=True)
        (configs / "hardware").mkdir(parents=True)
        (configs / "phases").mkdir(parents=True)
        (configs / "experiments").mkdir(parents=True)

        (configs / "defaults.yaml").write_text("seed: 42\noutput_dir: results\n")

        (configs / "hardware" / "h100_sxm.yaml").write_text(
            "gpu_type: h100_sxm\nprecision_modes: [bf16, fp8]\n"
            "flash_attn: flash_attn_3\nlock_clocks: true\n"
        )

        (configs / "models" / "test_model.yaml").write_text(
            "name: test_model\nhf_model_id: org/model\n"
            "params_billion: 3.0\ntier: compact\nadapter: TestAdapter\n"
            "supported_resolutions: [1024]\n"
        )

        (configs / "phases" / "inference.yaml").write_text(
            "batch_sizes: [1, 2]\nruns_per_config: 1\n"
        )

        exp_path = configs / "experiments" / "test.yaml"
        exp_path.write_text(
            "name: test_exp\nmodels: [test_model]\ngpus: [h100_sxm]\nphases: [inference]\n"
        )
        return exp_path

    def test_load_full(self, tmp_path: Path) -> None:
        from vlm_ocr_bench.config.loader import load_experiment_config

        exp_path = self._create_config_tree(tmp_path)
        cfg = load_experiment_config(exp_path)
        assert cfg.name == "test_exp"
        assert cfg.seed == 42
        assert cfg.models == ["test_model"]
        assert "h100_sxm" in cfg.gpu_configs
        assert cfg.gpu_configs["h100_sxm"].gpu_type == GPUType.H100_SXM
        assert "test_model" in cfg.model_configs
        assert cfg.model_configs["test_model"].name == "test_model"

    def test_cli_overrides(self, tmp_path: Path) -> None:
        from vlm_ocr_bench.config.loader import load_experiment_config

        exp_path = self._create_config_tree(tmp_path)
        cfg = load_experiment_config(exp_path, cli_overrides={"seed": 999})
        assert cfg.seed == 999

    def test_phase_config_merge(self, tmp_path: Path) -> None:
        from vlm_ocr_bench.config.loader import load_experiment_config

        exp_path = self._create_config_tree(tmp_path)
        cfg = load_experiment_config(exp_path)
        # Phase inference.yaml sets batch_sizes=[1,2], runs_per_config=1
        assert cfg.inference.batch_sizes == [1, 2]
        assert cfg.inference.runs_per_config == 1


# ─── Resolver Tests ───


class TestResolvers:
    """Test dynamic resolvers."""

    def test_resolve_flash_attn_h100(self) -> None:
        result = resolve_flash_attn_version(GPUType.H100_SXM)
        assert result == FlashAttnVersion.FA3

    def test_resolve_flash_attn_b300(self) -> None:
        result = resolve_flash_attn_version(GPUType.B300_SXM)
        assert result == FlashAttnVersion.FA4

    def test_resolve_flash_attn_unknown(self) -> None:
        result = resolve_flash_attn_version(None)
        assert result == FlashAttnVersion.FA2

    def test_resolve_path_expanduser(self) -> None:
        result = resolve_path("~/test")
        assert "~" not in str(result)
        assert result.is_absolute()

    def test_resolve_path_envvar(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_DIR", "/tmp/mydir")
        result = resolve_path("$TEST_DIR/sub")
        assert str(result) == "/tmp/mydir/sub"
