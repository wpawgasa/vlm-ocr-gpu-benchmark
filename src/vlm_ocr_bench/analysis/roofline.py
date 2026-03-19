"""Roofline model — compute arithmetic intensity, bottleneck, utilization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vlm_ocr_bench.config.schema import GPUType, PrecisionMode
from vlm_ocr_bench.hardware.profiles import HardwareProfile, get_profile


@dataclass
class RooflinePoint:
    """A single point on the roofline plot."""

    label: str
    arithmetic_intensity: float  # FLOP/byte
    measured_tflops: float
    peak_tflops: float
    memory_bandwidth_tb_s: float
    utilization_pct: float  # measured / peak * 100
    bottleneck: str  # "compute", "memory", or "balanced"


def _get_peak_tflops(profile: HardwareProfile, precision: PrecisionMode) -> float:
    """Get precision-specific peak TFLOPS for a GPU."""
    mapping = {
        PrecisionMode.BF16: profile.bf16_tflops,
        PrecisionMode.FP16: profile.fp16_tflops,
        PrecisionMode.FP8: profile.fp8_tflops,
        PrecisionMode.FP4: profile.fp4_tflops,
        PrecisionMode.NVFP4: profile.fp4_tflops,
    }
    return mapping.get(precision, profile.bf16_tflops)


def compute_roofline(
    gpu_type: GPUType,
    precision: PrecisionMode,
    measured_tflops: float,
    measured_bandwidth_tb_s: float,
    label: str = "",
) -> RooflinePoint:
    """Compute a roofline analysis point.

    Args:
        gpu_type: GPU type for peak specs
        precision: precision mode for peak TFLOPS lookup
        measured_tflops: measured compute throughput in TFLOPS
        measured_bandwidth_tb_s: measured memory bandwidth in TB/s
        label: descriptive label for this point (e.g., model name)

    Returns:
        RooflinePoint with arithmetic intensity, bottleneck, and utilization.
    """
    profile = get_profile(gpu_type)
    peak_tflops = _get_peak_tflops(profile, precision)
    mem_bw = profile.memory_bandwidth_tb_s

    # Arithmetic intensity = FLOP / byte
    # Ridge point = peak_tflops / mem_bw (TFLOP/s / TB/s = FLOP/byte)
    ridge_point = peak_tflops / mem_bw if mem_bw > 0 else 0.0

    if measured_bandwidth_tb_s > 0:
        arithmetic_intensity = measured_tflops / measured_bandwidth_tb_s
    else:
        # No bandwidth data — place well above ridge point to classify as compute-bound
        arithmetic_intensity = ridge_point * 2.0

    # Determine bottleneck
    if arithmetic_intensity < ridge_point * 0.9:
        bottleneck = "memory"
    elif arithmetic_intensity > ridge_point * 1.1:
        bottleneck = "compute"
    else:
        bottleneck = "balanced"

    utilization = (measured_tflops / peak_tflops * 100.0) if peak_tflops > 0 else 0.0

    return RooflinePoint(
        label=label,
        arithmetic_intensity=arithmetic_intensity,
        measured_tflops=measured_tflops,
        peak_tflops=peak_tflops,
        memory_bandwidth_tb_s=mem_bw,
        utilization_pct=utilization,
        bottleneck=bottleneck,
    )


def plot_roofline(
    gpu_type: GPUType,
    precision: PrecisionMode,
    points: list[RooflinePoint],
    output_path: Path | str,
) -> Path:
    """Generate an annotated roofline plot.

    Args:
        gpu_type: GPU type for theoretical peaks
        precision: precision mode for peak line
        points: list of measured RooflinePoints to plot
        output_path: path to save the PNG file

    Returns:
        Path to the saved plot file.
    """
    import matplotlib.pyplot as plt

    profile = get_profile(gpu_type)
    peak_tflops = _get_peak_tflops(profile, precision)
    mem_bw = profile.memory_bandwidth_tb_s
    ridge_point = peak_tflops / mem_bw if mem_bw > 0 else 1.0

    fig, ax = plt.subplots(figsize=(10, 6))

    # Roofline envelope
    x_range = np.logspace(-2, 4, 500)
    roofline = np.minimum(peak_tflops, x_range * mem_bw)
    ax.plot(x_range, roofline, "k-", linewidth=2, label="Roofline")

    # Ridge point
    ridge_label = f"Ridge ({ridge_point:.1f})"
    ax.axvline(x=ridge_point, color="gray", linestyle="--", alpha=0.5, label=ridge_label)

    # Plot measured points
    colors = {"compute": "red", "memory": "blue", "balanced": "green"}
    for pt in points:
        color = colors.get(pt.bottleneck, "gray")
        ax.scatter(
            pt.arithmetic_intensity,
            pt.measured_tflops,
            c=color,
            s=100,
            zorder=5,
            edgecolors="black",
        )
        ax.annotate(
            f"{pt.label}\n({pt.utilization_pct:.0f}%)",
            (pt.arithmetic_intensity, pt.measured_tflops),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=8,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Performance (TFLOPS)")
    ax.set_title(f"Roofline — {profile.name} ({precision.value})")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
