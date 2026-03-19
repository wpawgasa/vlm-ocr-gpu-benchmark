"""Pareto frontier — quality vs throughput tradeoff analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParetoPoint:
    """A single point in the quality-throughput space."""

    label: str
    throughput: float  # e.g., pages/s
    quality: float  # e.g., edit distance (higher = better)
    model_name: str = ""
    precision: str = ""
    gpu_type: str = ""
    is_pareto_optimal: bool = False


@dataclass
class ParetoResult:
    """Result of Pareto frontier analysis."""

    points: list[ParetoPoint] = field(default_factory=list)
    frontier: list[ParetoPoint] = field(default_factory=list)


def compute_pareto_frontier(
    points: list[ParetoPoint],
) -> ParetoResult:
    """Identify Pareto-optimal points (maximizing both quality and throughput).

    A point is Pareto-optimal if no other point has both higher quality
    AND higher throughput.

    Args:
        points: list of ParetoPoints to analyze

    Returns:
        ParetoResult with all points and the Pareto frontier marked.
    """
    if not points:
        return ParetoResult()

    # Sort by throughput descending
    sorted_pts = sorted(points, key=lambda p: p.throughput, reverse=True)

    frontier: list[ParetoPoint] = []
    max_quality = -float("inf")

    for pt in sorted_pts:
        if pt.quality >= max_quality:
            pt.is_pareto_optimal = True
            frontier.append(pt)
            max_quality = pt.quality

    # Sort frontier by throughput for plotting
    frontier.sort(key=lambda p: p.throughput)

    return ParetoResult(points=points, frontier=frontier)


def plot_pareto(
    result: ParetoResult,
    output_path: Path | str,
    x_label: str = "Throughput (pages/s)",
    y_label: str = "Quality (edit distance)",
    title: str = "Pareto Frontier — Quality vs Throughput",
) -> Path:
    """Generate a Pareto frontier plot.

    Args:
        result: ParetoResult from compute_pareto_frontier
        output_path: path to save the PNG
        x_label: X-axis label
        y_label: Y-axis label
        title: plot title

    Returns:
        Path to the saved plot file.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all non-frontier points
    non_frontier = [p for p in result.points if not p.is_pareto_optimal]
    if non_frontier:
        ax.scatter(
            [p.throughput for p in non_frontier],
            [p.quality for p in non_frontier],
            c="lightgray",
            s=60,
            alpha=0.7,
            label="Dominated",
            edgecolors="gray",
        )
        for p in non_frontier:
            ax.annotate(p.label, (p.throughput, p.quality), fontsize=7, alpha=0.6)

    # Plot frontier points
    if result.frontier:
        ax.scatter(
            [p.throughput for p in result.frontier],
            [p.quality for p in result.frontier],
            c="red",
            s=100,
            zorder=5,
            label="Pareto-optimal",
            edgecolors="black",
        )
        # Draw frontier line
        ax.plot(
            [p.throughput for p in result.frontier],
            [p.quality for p in result.frontier],
            "r--",
            alpha=0.5,
        )
        for p in result.frontier:
            ax.annotate(
                p.label,
                (p.throughput, p.quality),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=8,
                fontweight="bold",
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
