"""Inference metrics computation — latency percentiles, throughput, power efficiency."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class InferenceMetrics:
    """Aggregated inference metrics for a single configuration run."""

    # Latency (ms)
    ttft_p50: float = 0.0
    ttft_p95: float = 0.0
    ttft_p99: float = 0.0
    e2e_latency_p50: float = 0.0
    e2e_latency_p95: float = 0.0
    e2e_latency_p99: float = 0.0

    # Throughput
    pages_per_second: float = 0.0
    tokens_per_second: float = 0.0

    # Power efficiency
    mean_power_watts: float = 0.0
    peak_power_watts: float = 0.0
    energy_per_page_wh: float = 0.0
    pages_per_watt_hour: float = 0.0
    tokens_per_watt_hour: float = 0.0

    # Memory (GB)
    peak_gpu_memory_gb: float = 0.0
    kv_cache_memory_gb: float = 0.0

    # GPU utilization (0-1)
    mean_sm_occupancy: float = 0.0
    mean_tensor_util: float = 0.0
    mean_memory_util: float = 0.0

    # Throttle events
    throttle_events: int = 0

    # Raw per-request latencies
    per_request_latencies_ms: list[float] = field(default_factory=list)
    per_request_ttft_ms: list[float] = field(default_factory=list)


def compute_latency_percentiles(
    latencies_ms: list[float],
) -> tuple[float, float, float]:
    """Compute p50, p95, p99 from a list of latency values.

    Returns (p50, p95, p99) in milliseconds. Returns (0, 0, 0) if empty.
    """
    if not latencies_ms:
        return 0.0, 0.0, 0.0
    arr = np.array(latencies_ms)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    return p50, p95, p99


def compute_throughput(
    num_requests: int,
    total_output_tokens: int,
    wall_time_s: float,
) -> tuple[float, float]:
    """Compute pages/second and tokens/second.

    Returns (pages_per_second, tokens_per_second).
    """
    if wall_time_s <= 0:
        return 0.0, 0.0
    pages_per_second = num_requests / wall_time_s
    tokens_per_second = total_output_tokens / wall_time_s
    return pages_per_second, tokens_per_second


def compute_power_efficiency(
    pages_per_second: float,
    tokens_per_second: float,
    mean_power_watts: float,
) -> tuple[float, float, float]:
    """Compute power efficiency metrics.

    Returns (energy_per_page_wh, pages_per_watt_hour, tokens_per_watt_hour).
    """
    if mean_power_watts <= 0 or pages_per_second <= 0:
        return 0.0, 0.0, 0.0

    # Energy per page in watt-hours: power * time_per_page / 3600
    time_per_page_s = 1.0 / pages_per_second
    energy_per_page_wh = mean_power_watts * time_per_page_s / 3600.0

    pages_per_watt_hour = pages_per_second * 3600.0 / mean_power_watts
    tokens_per_watt_hour = tokens_per_second * 3600.0 / mean_power_watts

    return energy_per_page_wh, pages_per_watt_hour, tokens_per_watt_hour


def compute_metrics(
    per_request_latencies_ms: list[float],
    per_request_ttft_ms: list[float],
    total_output_tokens: int,
    wall_time_s: float,
    mean_power_watts: float = 0.0,
    peak_power_watts: float = 0.0,
    peak_gpu_memory_gb: float = 0.0,
    kv_cache_memory_gb: float = 0.0,
    mean_sm_occupancy: float = 0.0,
    mean_tensor_util: float = 0.0,
    mean_memory_util: float = 0.0,
    throttle_events: int = 0,
) -> InferenceMetrics:
    """Compute all inference metrics from raw measurements."""
    num_requests = len(per_request_latencies_ms)

    ttft_p50, ttft_p95, ttft_p99 = compute_latency_percentiles(per_request_ttft_ms)
    e2e_p50, e2e_p95, e2e_p99 = compute_latency_percentiles(per_request_latencies_ms)

    pages_per_second, tokens_per_second = compute_throughput(
        num_requests, total_output_tokens, wall_time_s
    )

    energy_per_page_wh, pages_per_watt_hour, tokens_per_watt_hour = compute_power_efficiency(
        pages_per_second, tokens_per_second, mean_power_watts
    )

    return InferenceMetrics(
        ttft_p50=ttft_p50,
        ttft_p95=ttft_p95,
        ttft_p99=ttft_p99,
        e2e_latency_p50=e2e_p50,
        e2e_latency_p95=e2e_p95,
        e2e_latency_p99=e2e_p99,
        pages_per_second=pages_per_second,
        tokens_per_second=tokens_per_second,
        mean_power_watts=mean_power_watts,
        peak_power_watts=peak_power_watts,
        energy_per_page_wh=energy_per_page_wh,
        pages_per_watt_hour=pages_per_watt_hour,
        tokens_per_watt_hour=tokens_per_watt_hour,
        peak_gpu_memory_gb=peak_gpu_memory_gb,
        kv_cache_memory_gb=kv_cache_memory_gb,
        mean_sm_occupancy=mean_sm_occupancy,
        mean_tensor_util=mean_tensor_util,
        mean_memory_util=mean_memory_util,
        throttle_events=throttle_events,
        per_request_latencies_ms=per_request_latencies_ms,
        per_request_ttft_ms=per_request_ttft_ms,
    )
