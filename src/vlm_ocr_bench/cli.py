"""CLI entry point for vlm-ocr-bench."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import structlog
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="vlm-ocr-bench",
    help="GPU benchmarking framework for document OCR Vision-Language Models.",
    no_args_is_help=True,
)
console = Console()
logger = structlog.get_logger()


# ─── info ───


@app.command()
def info() -> None:
    """Print GPU and environment information."""
    from vlm_ocr_bench import __version__
    from vlm_ocr_bench.hardware.detector import detect_gpu

    console.print(f"[bold]vlm-ocr-bench[/bold] v{__version__}")
    console.print()

    gpu = detect_gpu()

    table = Table(title="Environment", show_header=False, border_style="dim")
    table.add_column("Key", style="bold")
    table.add_column("Value")
    table.add_row("GPU", gpu.name)
    table.add_row("GPU Type", str(gpu.gpu_type or "unknown"))
    table.add_row("VRAM", f"{gpu.vram_gb:.1f} GB")
    table.add_row("Compute Capability", gpu.compute_capability_str)
    table.add_row("FP8 Support", "Yes" if gpu.supports_fp8 else "No")
    table.add_row("FP4 Support", "Yes" if gpu.supports_fp4 else "No")
    table.add_row("Driver", gpu.driver_version)
    table.add_row("CUDA", gpu.cuda_version)
    table.add_row("PyTorch", gpu.pytorch_version)
    table.add_row("vLLM", gpu.vllm_version)
    table.add_row("Flash Attention", gpu.flash_attn_version)
    console.print(table)


# ─── list-models ───


@app.command(name="list-models")
def list_models() -> None:
    """List all registered OCR model configurations."""
    from vlm_ocr_bench.models.registry import MODEL_REGISTRY

    table = Table(title="Registered Models")
    table.add_column("Name", style="bold")
    table.add_column("HF Model ID")
    table.add_column("Params (B)", justify="right")
    table.add_column("Tier")
    table.add_column("Resolutions")
    table.add_column("Max Tokens", justify="right")

    for name, cfg in sorted(MODEL_REGISTRY.items()):
        table.add_row(
            name,
            cfg.hf_model_id,
            f"{cfg.params_billion:.1f}",
            cfg.tier,
            ", ".join(str(r) for r in cfg.supported_resolutions),
            str(cfg.max_output_tokens),
        )

    console.print(table)


# ─── validate ───


@app.command()
def validate(
    config: Annotated[Path, typer.Argument(help="Path to experiment YAML config")],
) -> None:
    """Validate an experiment config file without running it."""
    if not config.exists():
        console.print(f"[red]Config file not found:[/red] {config}")
        raise typer.Exit(code=1)

    try:
        from vlm_ocr_bench.config.loader import load_experiment_config

        exp_config = load_experiment_config(config)
        console.print(f"[green]Config valid:[/green] {exp_config.name}")
        console.print(f"  Models: {', '.join(exp_config.models)}")
        console.print(f"  GPUs: {', '.join(str(g) for g in exp_config.gpus)}")
        console.print(f"  Phases: {', '.join(exp_config.phases)}")
        console.print(f"  Seed: {exp_config.seed}")
        console.print(f"  Output: {exp_config.output_dir}")
    except Exception as exc:
        console.print(f"[red]Config validation failed:[/red] {exc}")
        raise typer.Exit(code=1) from None


# ─── run ───


@app.command()
def run(
    config: Annotated[Path, typer.Argument(help="Path to experiment YAML config")],
    phase: Annotated[
        str | None,
        typer.Option(help="Run only this phase (inference, training, quality)"),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(help="Run only this model (overrides config)"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate config and print plan without executing"),
    ] = False,
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Override output directory"),
    ] = None,
) -> None:
    """Run a benchmark experiment from a YAML config file."""
    if not config.exists():
        console.print(f"[red]Config file not found:[/red] {config}")
        raise typer.Exit(code=1)

    from vlm_ocr_bench.config.loader import (
        load_experiment_config,
        save_config_snapshot,
        save_config_yaml_copy,
    )

    # Build CLI overrides
    cli_overrides: dict[str, object] = {}
    if output_dir is not None:
        cli_overrides["output_dir"] = str(output_dir)
    if model is not None:
        cli_overrides["models"] = [model]

    try:
        exp_config = load_experiment_config(
            config,
            cli_overrides=cli_overrides,
        )
    except Exception as exc:
        console.print(f"[red]Config loading failed:[/red] {exc}")
        raise typer.Exit(code=1) from None

    # Determine phases to run
    phases_to_run = exp_config.phases
    if phase is not None:
        if phase not in ("inference", "training", "quality"):
            console.print(f"[red]Unknown phase:[/red] {phase}")
            console.print("Valid phases: inference, training, quality")
            raise typer.Exit(code=1)
        phases_to_run = [phase]

    # Print plan
    console.print(f"\n[bold]Experiment:[/bold] {exp_config.name}")
    console.print(f"[bold]Models:[/bold] {', '.join(exp_config.models)}")
    console.print(f"[bold]GPUs:[/bold] {', '.join(str(g) for g in exp_config.gpus)}")
    console.print(f"[bold]Phases:[/bold] {', '.join(phases_to_run)}")
    console.print(f"[bold]Output:[/bold] {exp_config.output_dir}")
    console.print()

    if dry_run:
        console.print("[yellow]Dry run — no benchmarks will be executed.[/yellow]")
        return

    # Save config snapshot
    results_dir = exp_config.output_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(exp_config, results_dir)
    save_config_yaml_copy(config, results_dir)

    # Execute phases
    total_start = time.monotonic()
    _run_phases(exp_config, phases_to_run)
    total_time = time.monotonic() - total_start

    console.print(f"\n[green]Experiment complete in {total_time:.1f}s[/green]")


def _run_phases(
    config: object,
    phases: list[str],
) -> None:
    """Dispatch to phase runners for each model x GPU combination."""
    from vlm_ocr_bench.config.schema import ExperimentConfig

    assert isinstance(config, ExperimentConfig)

    from vlm_ocr_bench.models.registry import get_model_config

    for phase_name in phases:
        console.rule(f"[bold]Phase: {phase_name}[/bold]")

        for model_name in config.models:
            # Get model config from experiment or registry
            if model_name in config.model_configs:
                model_cfg = config.model_configs[model_name]
            else:
                model_cfg = get_model_config(model_name)

            for gpu_type in config.gpus:
                gpu_key = str(gpu_type.value)
                if gpu_key not in config.gpu_configs:
                    console.print(
                        f"  [yellow]Skipping {model_name} on {gpu_key}: "
                        f"no GPU config found[/yellow]"
                    )
                    continue

                gpu_cfg = config.gpu_configs[gpu_key]
                console.print(f"  [bold]{model_name}[/bold] on [bold]{gpu_key}[/bold]")

                try:
                    _run_single_phase(phase_name, model_cfg, gpu_cfg, config)
                except Exception:
                    logger.error(
                        "phase_failed",
                        phase=phase_name,
                        model=model_name,
                        gpu=gpu_key,
                        exc_info=True,
                    )
                    console.print("    [red]Failed — see logs[/red]")


def _run_single_phase(
    phase: str,
    model_cfg: object,
    gpu_cfg: object,
    config: object,
) -> None:
    """Run a single phase for one model x GPU combination."""
    from vlm_ocr_bench.config.schema import (
        ExperimentConfig,
        GPUConfig,
        ModelConfig,
    )

    assert isinstance(model_cfg, ModelConfig)
    assert isinstance(gpu_cfg, GPUConfig)
    assert isinstance(config, ExperimentConfig)

    if phase == "inference":
        from vlm_ocr_bench.inference.runner import InferenceBenchmarkRunner

        inf_runner = InferenceBenchmarkRunner(model_cfg, gpu_cfg, config.inference)
        inf_result = inf_runner.run()
        console.print(
            f"    Inference: {len(inf_result.configs)} configs, {inf_result.total_wall_time_s:.1f}s"
        )

    elif phase == "training":
        from vlm_ocr_bench.training.trainer import TrainingBenchmarkRunner

        train_runner = TrainingBenchmarkRunner(model_cfg, gpu_cfg, config.training)
        train_result = train_runner.run()
        console.print(
            f"    Training: {len(train_result.configs)} configs, "
            f"{train_result.total_wall_time_s:.1f}s"
        )

    elif phase == "quality":
        from vlm_ocr_bench.evaluation.runner import QualityEvalRunner

        qual_runner = QualityEvalRunner(model_cfg, gpu_cfg, config.quality)
        qual_result = qual_runner.run()
        console.print(
            f"    Quality: {len(qual_result.benchmarks)} benchmarks, "
            f"{qual_result.total_wall_time_s:.1f}s"
        )

    else:
        console.print(f"    [yellow]Unknown phase: {phase}[/yellow]")


# ─── analyze ───


@app.command()
def analyze(
    results_dir: Annotated[
        Path,
        typer.Argument(help="Path to results directory"),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Output directory for report (default: results_dir/reports)"),
    ] = None,
) -> None:
    """Analyze benchmark results and generate a report."""
    if not results_dir.exists():
        console.print(f"[red]Results directory not found:[/red] {results_dir}")
        raise typer.Exit(code=1)

    from vlm_ocr_bench.analysis.report import ReportData, ReportGenerator

    report_dir = output_dir or (results_dir / "reports")

    data = ReportData(
        experiment_name=results_dir.name,
    )

    generator = ReportGenerator(data, output_dir=report_dir)
    report_path = generator.generate(filename="report.md")
    console.print(f"[green]Report generated:[/green] {report_path}")


# ─── download ───


@app.command()
def download(
    target: Annotated[
        str,
        typer.Argument(help="What to download: models, data, or all"),
    ] = "all",
) -> None:
    """Pre-download model weights and/or benchmark datasets."""
    valid_targets = ("models", "data", "all")
    if target not in valid_targets:
        console.print(f"[red]Invalid target:[/red] {target}")
        console.print(f"Valid targets: {', '.join(valid_targets)}")
        raise typer.Exit(code=1)

    if target in ("models", "all"):
        console.print("[bold]Downloading model weights...[/bold]")
        console.print(
            "[yellow]Model download requires HuggingFace access. "
            "Use `huggingface-cli login` first.[/yellow]"
        )

        from vlm_ocr_bench.models.registry import MODEL_REGISTRY

        for name, cfg in MODEL_REGISTRY.items():
            console.print(f"  {name}: {cfg.hf_model_id}")

    if target in ("data", "all"):
        console.print("[bold]Downloading benchmark datasets...[/bold]")

        from vlm_ocr_bench.data.datasets import DATASET_REGISTRY

        for name, spec in DATASET_REGISTRY.items():
            console.print(f"  {name}: {spec.hf_id or spec.source}")

    console.print(
        "\n[dim]Note: Actual downloads occur on first use during benchmarking. "
        "Pre-download via `huggingface-cli download <model_id>`.[/dim]"
    )


if __name__ == "__main__":
    app()
