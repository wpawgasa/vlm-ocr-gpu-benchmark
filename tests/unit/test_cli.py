"""Unit tests for the CLI module (no GPU)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from vlm_ocr_bench.cli import app

runner = CliRunner()


# ─── info command ───


class TestInfoCommand:
    def test_info_runs(self) -> None:
        from vlm_ocr_bench.hardware.detector import GPUInfo

        mock_gpu = GPUInfo(
            name="Test GPU",
            gpu_type=None,
            vram_gb=80.0,
            compute_capability=(9, 0),
            driver_version="555.42",
            cuda_version="12.6",
            pytorch_version="2.4.0",
            vllm_version="0.6.0",
            flash_attn_version="2.5.0",
        )
        with patch("vlm_ocr_bench.hardware.detector.detect_gpu", return_value=mock_gpu):
            result = runner.invoke(app, ["info"])
            assert result.exit_code == 0
            assert "vlm-ocr-bench" in result.output


# ─── list-models command ───


class TestListModelsCommand:
    def test_list_models_runs(self) -> None:
        result = runner.invoke(app, ["list-models"])
        assert result.exit_code == 0
        # Should list at least one model
        assert "paddleocr" in result.output.lower() or "dots" in result.output.lower()


# ─── validate command ───


class TestValidateCommand:
    def test_validate_missing_file(self) -> None:
        result = runner.invoke(app, ["validate", "/nonexistent/config.yaml"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_validate_valid_config(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("name: test\nmodels: [dots_ocr_1.5_3b]\ngpus: [h100_sxm]\n")
            f.flush()
            result = runner.invoke(app, ["validate", f.name])
            assert result.exit_code == 0
            assert "valid" in result.output.lower()

    def test_validate_invalid_config(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("not_a_valid_field: true\n")
            f.flush()
            result = runner.invoke(app, ["validate", f.name])
            assert result.exit_code == 1


# ─── run command ───


class TestRunCommand:
    def test_run_missing_file(self) -> None:
        result = runner.invoke(app, ["run", "/nonexistent/config.yaml"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_run_dry_run(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(
                "name: test_dry\nmodels: [dots_ocr_1.5_3b]\ngpus: [h100_sxm]\nphases: [inference]\n"
            )
            f.flush()
            result = runner.invoke(app, ["run", f.name, "--dry-run"])
            assert result.exit_code == 0
            assert "dry run" in result.output.lower()

    def test_run_invalid_phase(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write("name: test\nmodels: [dots_ocr_1.5_3b]\ngpus: [h100_sxm]\n")
            f.flush()
            result = runner.invoke(app, ["run", f.name, "--phase", "invalid"])
            assert result.exit_code == 1
            assert "unknown phase" in result.output.lower()

    def test_run_with_phase_filter(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(
                "name: test\n"
                "models: [dots_ocr_1.5_3b]\n"
                "gpus: [h100_sxm]\n"
                "phases: [inference, training, quality]\n"
            )
            f.flush()
            result = runner.invoke(app, ["run", f.name, "--dry-run", "--phase", "inference"])
            assert result.exit_code == 0
            assert "inference" in result.output.lower()


# ─── analyze command ───


class TestAnalyzeCommand:
    def test_analyze_missing_dir(self) -> None:
        result = runner.invoke(app, ["analyze", "/nonexistent/results"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_analyze_basic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(app, ["analyze", tmpdir])
            assert result.exit_code == 0
            assert "report generated" in result.output.lower()
            report_path = Path(tmpdir) / "reports" / "report.md"
            assert report_path.exists()


# ─── download command ───


class TestDownloadCommand:
    def test_download_invalid_target(self) -> None:
        result = runner.invoke(app, ["download", "invalid"])
        assert result.exit_code == 1

    def test_download_models(self) -> None:
        result = runner.invoke(app, ["download", "models"])
        assert result.exit_code == 0

    def test_download_data(self) -> None:
        result = runner.invoke(app, ["download", "data"])
        assert result.exit_code == 0

    def test_download_all(self) -> None:
        result = runner.invoke(app, ["download", "all"])
        assert result.exit_code == 0


# ─── no-args help ───


class TestAppHelp:
    def test_no_args_shows_help(self) -> None:
        result = runner.invoke(app, [])
        # Typer returns exit code 0 or 2 for no_args_is_help
        assert result.exit_code in (0, 2)
        assert "gpu benchmarking" in result.output.lower() or "usage" in result.output.lower()

    def test_help_flag(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0


# ─── internal helpers ───


class TestRunPhases:
    def test_run_phases_skips_missing_gpu_config(self) -> None:
        """Phases should skip model x GPU combos without GPU config."""
        from vlm_ocr_bench.config.schema import ExperimentConfig

        config = ExperimentConfig(
            name="test",
            models=["dots_ocr_1.5_3b"],
            gpus=["h100_sxm"],  # type: ignore[list-item]
            gpu_configs={},  # no GPU configs
        )

        from vlm_ocr_bench.cli import _run_phases

        # Should not raise, just skip
        _run_phases(config, ["inference"])
