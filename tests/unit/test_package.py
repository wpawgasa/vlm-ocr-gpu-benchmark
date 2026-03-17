"""Tests for package setup and basic imports."""


def test_version() -> None:
    """Package version is accessible."""
    from vlm_ocr_bench import __version__

    assert __version__ == "1.0.0"


def test_subpackages_importable() -> None:
    """All subpackages can be imported."""
    import vlm_ocr_bench.analysis
    import vlm_ocr_bench.config
    import vlm_ocr_bench.data
    import vlm_ocr_bench.evaluation
    import vlm_ocr_bench.evaluation.benchmarks
    import vlm_ocr_bench.evaluation.metrics
    import vlm_ocr_bench.hardware
    import vlm_ocr_bench.inference
    import vlm_ocr_bench.models
    import vlm_ocr_bench.models.adapters
    import vlm_ocr_bench.profiling
    import vlm_ocr_bench.training
    import vlm_ocr_bench.utils  # noqa: F401


def test_cli_app_exists() -> None:
    """CLI app is importable."""
    from vlm_ocr_bench.cli import app

    assert app is not None
