"""Visualization generators — matplotlib-based plot functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _save_figure(fig: Any, output_path: Path | str) -> Path:
    """Save a matplotlib figure and close it."""
    import matplotlib.pyplot as plt

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_throughput_comparison(
    models: list[str],
    h100_values: list[float],
    b300_values: list[float],
    output_path: Path | str,
    metric_label: str = "Pages/s",
    title: str = "Throughput Comparison — H100 vs B300",
    gpu_a_label: str = "H100 SXM",
    gpu_b_label: str = "B300 SXM",
) -> Path:
    """Generate grouped bar chart comparing throughput across GPUs.

    Args:
        models: model names for X-axis
        h100_values: throughput values for GPU A
        b300_values: throughput values for GPU B
        output_path: path to save the PNG
        metric_label: Y-axis label
        title: plot title
        gpu_a_label: label for the first GPU series
        gpu_b_label: label for the second GPU series

    Returns:
        Path to the saved plot.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width / 2, h100_values, width, label=gpu_a_label, color="#1f77b4")
    ax.bar(x + width / 2, b300_values, width, label=gpu_b_label, color="#ff7f0e")

    ax.set_xlabel("Model")
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    return _save_figure(fig, output_path)


def plot_latency_cdf(
    latencies: dict[str, list[float]],
    output_path: Path | str,
    title: str = "Latency CDF",
    x_label: str = "Latency (ms)",
) -> Path:
    """Generate CDF plot for latency distributions.

    Args:
        latencies: {label: [latency_values]} for each series
        output_path: path to save the PNG
        title: plot title
        x_label: X-axis label

    Returns:
        Path to the saved plot.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, values in latencies.items():
        if not values:
            continue
        sorted_vals = np.sort(values)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf, label=label, linewidth=2)

    # Mark p50, p95, p99 guidelines
    # Use a blended transform: x in axes coordinates (0=left edge), y in data coordinates.
    # This avoids reading ax.get_xlim() before data limits are established.
    from matplotlib.transforms import blended_transform_factory

    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for pct, style in [(0.5, ":"), (0.95, "--"), (0.99, "-.")]:
        ax.axhline(y=pct, color="gray", linestyle=style, alpha=0.4)
        ax.text(0.01, pct + 0.01, f"p{int(pct * 100)}", transform=trans, fontsize=8, color="gray")

    ax.set_xlabel(x_label)
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return _save_figure(fig, output_path)


def plot_power_efficiency(
    models: list[str],
    pages_per_wh: dict[str, list[float]],
    output_path: Path | str,
    title: str = "Power Efficiency — Pages per Watt-Hour",
) -> Path:
    """Generate power efficiency scatter/bar plot.

    Args:
        models: model names
        pages_per_wh: {gpu_label: [values_per_model]} for each GPU
        output_path: path to save the PNG
        title: plot title

    Returns:
        Path to the saved plot.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.35
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, (gpu_label, values) in enumerate(pages_per_wh.items()):
        offset = (i - len(pages_per_wh) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=gpu_label, color=colors[i % len(colors)])

    ax.set_xlabel("Model")
    ax.set_ylabel("Pages / Watt-Hour")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    return _save_figure(fig, output_path)


def plot_memory_usage(
    models: list[str],
    memory_data: dict[str, list[float]],
    output_path: Path | str,
    title: str = "Peak GPU Memory Usage",
) -> Path:
    """Generate memory usage bar chart.

    Args:
        models: model names
        memory_data: {gpu_label: [peak_memory_gb_per_model]}
        output_path: path to save the PNG
        title: plot title

    Returns:
        Path to the saved plot.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.35
    colors = ["#1f77b4", "#ff7f0e"]

    for i, (gpu_label, values) in enumerate(memory_data.items()):
        offset = (i - len(memory_data) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=gpu_label, color=colors[i % len(colors)])

    ax.set_xlabel("Model")
    ax.set_ylabel("Peak GPU Memory (GB)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    return _save_figure(fig, output_path)


def plot_training_convergence(
    loss_curves: dict[str, list[tuple[int, float]]],
    output_path: Path | str,
    title: str = "Training Convergence",
) -> Path:
    """Generate training loss convergence curves.

    Args:
        loss_curves: {label: [(step, loss), ...]} for each configuration
        output_path: path to save the PNG
        title: plot title

    Returns:
        Path to the saved plot.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, curve in loss_curves.items():
        if not curve:
            continue
        steps = [s for s, _ in curve]
        losses = [loss for _, loss in curve]
        ax.plot(steps, losses, label=label, linewidth=2)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return _save_figure(fig, output_path)


def plot_quality_heatmap(
    models: list[str],
    doc_types: list[str],
    scores: list[list[float]],
    output_path: Path | str,
    metric_name: str = "Edit Distance",
    title: str = "Quality Heatmap",
) -> Path:
    """Generate quality metric heatmap (model x doc_type).

    Args:
        models: row labels (model names)
        doc_types: column labels (document types)
        scores: 2D list [model_idx][doc_type_idx] of metric values
        output_path: path to save the PNG
        metric_name: name of the metric for colorbar
        title: plot title

    Returns:
        Path to the saved plot.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(8, len(doc_types) * 1.5), max(4, len(models) * 0.8)))
    data = np.array(scores)

    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, label=metric_name)

    ax.set_xticks(np.arange(len(doc_types)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(doc_types, rotation=45, ha="right")
    ax.set_yticklabels(models)
    ax.set_title(title)

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(doc_types)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)

    return _save_figure(fig, output_path)
