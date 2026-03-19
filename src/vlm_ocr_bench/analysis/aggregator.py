"""Result aggregation — group by config, compute mean/std/CI across runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import structlog

from vlm_ocr_bench.analysis.statistics import bootstrap_ci

logger = structlog.get_logger()


@dataclass
class AggregatedMetric:
    """A single metric aggregated across multiple runs."""

    name: str
    mean: float = 0.0
    std: float = 0.0
    ci_95_low: float = 0.0
    ci_95_high: float = 0.0
    n_runs: int = 0
    values: list[float] = field(default_factory=list)


@dataclass
class AggregatedConfig:
    """Aggregated results for a single (model, gpu, precision, batch_size, resolution) config."""

    model_name: str
    gpu_type: str
    precision: str
    batch_size: int
    resolution: int
    max_output_tokens: int = 0
    n_runs: int = 0
    metrics: dict[str, AggregatedMetric] = field(default_factory=dict)


def _group_key(row: dict[str, Any]) -> tuple[str, str, str, int, int, int]:
    """Create grouping key from a result row."""
    return (
        str(row.get("model_name", "")),
        str(row.get("gpu_type", "")),
        str(row.get("precision", "")),
        int(row.get("batch_size", 0)),
        int(row.get("resolution", 0)),
        int(row.get("max_output_tokens", 0)),
    )


_INFERENCE_METRIC_FIELDS = [
    "e2e_latency_p50",
    "e2e_latency_p95",
    "e2e_latency_p99",
    "ttft_p50",
    "ttft_p95",
    "ttft_p99",
    "pages_per_second",
    "tokens_per_second",
    "mean_power_watts",
    "peak_power_watts",
    "energy_per_page_wh",
    "pages_per_watt_hour",
    "tokens_per_watt_hour",
    "peak_gpu_memory_gb",
]

_TRAINING_METRIC_FIELDS = [
    "samples_per_second",
    "tokens_per_second",
    "peak_gpu_memory_gb",
    "mean_power_watts",
    "energy_per_1k_samples_wh",
]


def _aggregate_metric(name: str, values: list[float]) -> AggregatedMetric:
    """Compute mean, std, and 95% bootstrap CI for a list of values."""
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    mean_val = float(np.mean(arr)) if n > 0 else 0.0
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    if n >= 2:
        ci_low, ci_high = bootstrap_ci(values, n_bootstrap=5000)
    else:
        ci_low, ci_high = mean_val, mean_val

    return AggregatedMetric(
        name=name,
        mean=mean_val,
        std=std_val,
        ci_95_low=ci_low,
        ci_95_high=ci_high,
        n_runs=n,
        values=values,
    )


def aggregate_inference_results(
    results: list[dict[str, Any]],
) -> list[AggregatedConfig]:
    """Aggregate inference benchmark results across runs.

    Args:
        results: list of flattened result dicts, each containing
            model_name, gpu_type, precision, batch_size, resolution,
            max_output_tokens, and metric fields.

    Returns:
        List of AggregatedConfig, one per unique config group.
    """
    if not results:
        return []

    # Filter out OOM/error rows
    valid = [r for r in results if not r.get("oom", False) and not r.get("error")]

    # Group by config
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in valid:
        key = _group_key(row)
        groups.setdefault(key, []).append(row)

    aggregated: list[AggregatedConfig] = []
    for key, rows in groups.items():
        model, gpu, precision, batch_size, resolution, max_tokens = key
        config = AggregatedConfig(
            model_name=model,
            gpu_type=gpu,
            precision=precision,
            batch_size=batch_size,
            resolution=resolution,
            max_output_tokens=max_tokens,
            n_runs=len(rows),
        )

        for metric_name in _INFERENCE_METRIC_FIELDS:
            values = []
            for r in rows:
                if r.get(metric_name) is None and metric_name not in r:
                    logger.warning("missing_metric_field", metric=metric_name, config=key)
                values.append(float(r.get(metric_name, 0.0)))
            config.metrics[metric_name] = _aggregate_metric(metric_name, values)

        aggregated.append(config)

    logger.info(
        "inference_results_aggregated",
        n_configs=len(aggregated),
        n_input_rows=len(results),
    )
    return aggregated


def aggregate_training_results(
    results: list[dict[str, Any]],
) -> list[AggregatedConfig]:
    """Aggregate training benchmark results across runs.

    Args:
        results: list of flattened result dicts with training metrics.

    Returns:
        List of AggregatedConfig for training data.
    """
    if not results:
        return []

    valid = [r for r in results if not r.get("oom", False) and not r.get("error")]

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in valid:
        key = (
            str(row.get("model_name", "")),
            str(row.get("gpu_type", "")),
            str(row.get("precision", "")),
            int(row.get("micro_batch_size", row.get("batch_size", 0))),
            0,  # no resolution for training
            0,
        )
        groups.setdefault(key, []).append(row)

    aggregated: list[AggregatedConfig] = []
    for key, rows in groups.items():
        model, gpu, precision, batch_size, _, _ = key
        config = AggregatedConfig(
            model_name=model,
            gpu_type=gpu,
            precision=precision,
            batch_size=batch_size,
            resolution=0,
            n_runs=len(rows),
        )

        for metric_name in _TRAINING_METRIC_FIELDS:
            values = []
            for r in rows:
                if r.get(metric_name) is None and metric_name not in r:
                    logger.warning("missing_metric_field", metric=metric_name, config=key)
                values.append(float(r.get(metric_name, 0.0)))
            config.metrics[metric_name] = _aggregate_metric(metric_name, values)

        aggregated.append(config)

    logger.info(
        "training_results_aggregated",
        n_configs=len(aggregated),
        n_input_rows=len(results),
    )
    return aggregated


def flatten_inference_result(
    model_name: str,
    gpu_type: str,
    configs: list[Any],
) -> list[dict[str, Any]]:
    """Flatten SingleConfigResult list into dicts for aggregation.

    Args:
        model_name: model identifier
        gpu_type: GPU type string
        configs: list of SingleConfigResult dataclass instances

    Returns:
        List of flat dicts suitable for aggregation or DataFrame creation.
    """
    rows: list[dict[str, Any]] = []
    for cfg in configs:
        row: dict[str, Any] = {
            "model_name": model_name,
            "gpu_type": gpu_type,
            "precision": cfg.precision,
            "batch_size": cfg.batch_size,
            "resolution": cfg.resolution,
            "max_output_tokens": cfg.max_output_tokens,
            "run_id": cfg.run_id,
            "num_requests": cfg.num_requests,
            "wall_time_s": cfg.wall_time_s,
            "oom": cfg.oom,
            "error": cfg.error,
        }
        # Flatten metrics
        m = cfg.metrics
        for field_name in _INFERENCE_METRIC_FIELDS:
            row[field_name] = getattr(m, field_name, 0.0)
        rows.append(row)
    return rows


def results_to_dataframe(aggregated: list[AggregatedConfig]) -> pd.DataFrame:
    """Convert aggregated results to a pandas DataFrame."""
    rows: list[dict[str, Any]] = []
    for cfg in aggregated:
        row: dict[str, Any] = {
            "model_name": cfg.model_name,
            "gpu_type": cfg.gpu_type,
            "precision": cfg.precision,
            "batch_size": cfg.batch_size,
            "resolution": cfg.resolution,
            "n_runs": cfg.n_runs,
        }
        for metric_name, metric in cfg.metrics.items():
            row[f"{metric_name}_mean"] = metric.mean
            row[f"{metric_name}_std"] = metric.std
            row[f"{metric_name}_ci_low"] = metric.ci_95_low
            row[f"{metric_name}_ci_high"] = metric.ci_95_high
        rows.append(row)
    return pd.DataFrame(rows)
