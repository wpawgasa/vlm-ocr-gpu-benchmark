"""Training metrics dataclasses and computation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainingMetrics:
    """Aggregated training metrics for a single configuration run."""

    # Throughput
    samples_per_second: float = 0.0
    tokens_per_second: float = 0.0

    # Memory (GB)
    peak_gpu_memory_gb: float = 0.0
    model_memory_gb: float = 0.0
    optimizer_memory_gb: float = 0.0
    activation_memory_gb: float = 0.0

    # Power efficiency
    mean_power_watts: float = 0.0
    peak_power_watts: float = 0.0
    energy_per_1k_samples_wh: float = 0.0

    # Stability
    loss_values: list[float] = field(default_factory=list)
    gradient_norm_values: list[float] = field(default_factory=list)

    # Cost estimates
    estimated_cost_per_epoch: float = 0.0
    estimated_time_per_epoch_hours: float = 0.0


def compute_throughput(
    num_samples: int,
    num_tokens: int,
    wall_time_s: float,
) -> tuple[float, float]:
    """Compute samples/second and tokens/second.

    Returns (samples_per_second, tokens_per_second).
    """
    if wall_time_s <= 0:
        return 0.0, 0.0
    return num_samples / wall_time_s, num_tokens / wall_time_s


def compute_power_efficiency(
    samples_per_second: float,
    mean_power_watts: float,
) -> float:
    """Compute energy per 1K samples in watt-hours.

    Returns energy_per_1k_samples_wh.
    """
    if mean_power_watts <= 0 or samples_per_second <= 0:
        return 0.0
    time_per_sample_s = 1.0 / samples_per_second
    energy_per_sample_wh = mean_power_watts * time_per_sample_s / 3600.0
    return energy_per_sample_wh * 1000.0


def compute_cost_estimate(
    samples_per_second: float,
    total_samples: int,
    cost_per_gpu_hour: float = 2.0,
) -> tuple[float, float]:
    """Estimate cost and time per epoch.

    Args:
        samples_per_second: Training throughput.
        total_samples: Number of samples in one epoch.
        cost_per_gpu_hour: Cloud GPU hourly rate (default $2/hr).

    Returns (estimated_cost_per_epoch, estimated_time_per_epoch_hours).
    """
    if samples_per_second <= 0:
        return 0.0, 0.0
    time_s = total_samples / samples_per_second
    time_hours = time_s / 3600.0
    cost = time_hours * cost_per_gpu_hour
    return cost, time_hours


def compute_training_metrics(
    num_samples: int,
    num_tokens: int,
    wall_time_s: float,
    peak_gpu_memory_gb: float = 0.0,
    model_memory_gb: float = 0.0,
    optimizer_memory_gb: float = 0.0,
    activation_memory_gb: float = 0.0,
    mean_power_watts: float = 0.0,
    peak_power_watts: float = 0.0,
    loss_values: list[float] | None = None,
    gradient_norm_values: list[float] | None = None,
    total_epoch_samples: int = 50000,
    cost_per_gpu_hour: float = 2.0,
) -> TrainingMetrics:
    """Compute all training metrics from raw measurements."""
    sps, tps = compute_throughput(num_samples, num_tokens, wall_time_s)
    energy = compute_power_efficiency(sps, mean_power_watts)
    cost, time_hours = compute_cost_estimate(sps, total_epoch_samples, cost_per_gpu_hour)

    return TrainingMetrics(
        samples_per_second=sps,
        tokens_per_second=tps,
        peak_gpu_memory_gb=peak_gpu_memory_gb,
        model_memory_gb=model_memory_gb,
        optimizer_memory_gb=optimizer_memory_gb,
        activation_memory_gb=activation_memory_gb,
        mean_power_watts=mean_power_watts,
        peak_power_watts=peak_power_watts,
        energy_per_1k_samples_wh=energy,
        loss_values=loss_values or [],
        gradient_norm_values=gradient_norm_values or [],
        estimated_cost_per_epoch=cost,
        estimated_time_per_epoch_hours=time_hours,
    )
