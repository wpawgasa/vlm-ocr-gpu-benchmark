"""CLI entry point for vlm-ocr-bench."""

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    name="vlm-ocr-bench",
    help="GPU benchmarking framework for document OCR Vision-Language Models.",
)
console = Console()


@app.command()
def info() -> None:
    """Print GPU and environment information."""
    from vlm_ocr_bench import __version__

    console.print(f"[bold]vlm-ocr-bench[/bold] v{__version__}")
    console.print("GPU info: (not yet implemented)")


@app.command()
def run(
    config: Path = typer.Argument(..., help="Path to experiment YAML config"),  # noqa: B008
) -> None:
    """Run a benchmark experiment from a YAML config file."""
    if not config.exists():
        console.print(f"[red]Config file not found:[/red] {config}")
        raise typer.Exit(code=1)
    console.print(f"Running experiment from: {config}")
    console.print("[yellow]Not yet implemented[/yellow]")


if __name__ == "__main__":
    app()
