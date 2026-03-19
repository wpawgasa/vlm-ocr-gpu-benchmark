"""TCO (Total Cost of Ownership) projection for GPU deployments."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class TCOParams:
    """Configurable TCO parameters."""

    gpu_rental_rate_per_hour: float = 2.00  # $/hour/GPU (cloud rate)
    power_cost_per_kwh: float = 0.10  # $/kWh
    pages_per_day: int = 100_000  # throughput target
    operating_hours_per_day: float = 24.0
    projection_months: int = 12


@dataclass
class TCOResult:
    """TCO projection result."""

    # GPU requirements
    gpus_required: int = 1
    pages_per_second_per_gpu: float = 0.0

    # Monthly costs
    monthly_gpu_rental: float = 0.0
    monthly_power_cost: float = 0.0
    monthly_total: float = 0.0

    # Annual costs
    annual_gpu_rental: float = 0.0
    annual_power_cost: float = 0.0
    annual_total: float = 0.0

    # Per-unit costs
    cost_per_1k_pages: float = 0.0

    # Power
    total_power_kw: float = 0.0
    energy_per_day_kwh: float = 0.0

    # Projection
    projection_months: int = 12
    monthly_breakdown: list[dict[str, float]] = field(default_factory=list)


def compute_tco(
    pages_per_second: float,
    mean_power_watts: float,
    params: TCOParams | None = None,
) -> TCOResult:
    """Compute Total Cost of Ownership for a GPU deployment.

    Args:
        pages_per_second: measured throughput per GPU
        mean_power_watts: measured power draw per GPU in watts
        params: TCO parameters (uses defaults if None)

    Returns:
        TCOResult with cost breakdown and projections.
    """
    if params is None:
        params = TCOParams()

    result = TCOResult(projection_months=params.projection_months)
    result.pages_per_second_per_gpu = pages_per_second

    if pages_per_second <= 0:
        return result

    # GPUs required to meet throughput target
    pages_per_day_per_gpu = pages_per_second * params.operating_hours_per_day * 3600
    result.gpus_required = max(1, math.ceil(params.pages_per_day / pages_per_day_per_gpu))

    # Power
    result.total_power_kw = mean_power_watts * result.gpus_required / 1000.0
    result.energy_per_day_kwh = result.total_power_kw * params.operating_hours_per_day

    # Monthly costs (30.44 days/month average)
    days_per_month = 30.44
    hours_per_month = params.operating_hours_per_day * days_per_month

    result.monthly_gpu_rental = (
        params.gpu_rental_rate_per_hour * hours_per_month * result.gpus_required
    )
    result.monthly_power_cost = (
        result.energy_per_day_kwh * days_per_month * params.power_cost_per_kwh
    )
    result.monthly_total = result.monthly_gpu_rental + result.monthly_power_cost

    # Annual costs
    result.annual_gpu_rental = result.monthly_gpu_rental * 12
    result.annual_power_cost = result.monthly_power_cost * 12
    result.annual_total = result.monthly_total * 12

    # Cost per 1K pages
    total_pages_per_month = params.pages_per_day * days_per_month
    if total_pages_per_month > 0:
        result.cost_per_1k_pages = result.monthly_total / total_pages_per_month * 1000

    # Monthly projection
    cumulative = 0.0
    for month in range(1, params.projection_months + 1):
        cumulative += result.monthly_total
        result.monthly_breakdown.append(
            {
                "month": float(month),
                "monthly_cost": result.monthly_total,
                "cumulative_cost": cumulative,
            }
        )

    return result
