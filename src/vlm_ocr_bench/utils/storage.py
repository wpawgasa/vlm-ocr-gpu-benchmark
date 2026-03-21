"""Result persistence — save benchmark results as Parquet + CSV."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

logger = structlog.get_logger()


def _timestamp() -> str:
    """Return a UTC timestamp string for filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _flatten_inference_result(result: Any) -> list[dict[str, Any]]:
    """Flatten an InferenceBenchmarkResult into a list of row dicts."""
    rows: list[dict[str, Any]] = []
    for cfg in result.configs:
        row: dict[str, Any] = {
            "model": result.model_name,
            "gpu": result.gpu_type,
            "resolution": cfg.resolution,
            "batch_size": cfg.batch_size,
            "precision": cfg.precision,
            "max_output_tokens": cfg.max_output_tokens,
            "run_id": cfg.run_id,
            "num_requests": cfg.num_requests,
            "wall_time_s": cfg.wall_time_s,
            "oom": cfg.oom,
            "error": cfg.error,
        }
        # Flatten metrics (exclude per-request lists)
        metrics = asdict(cfg.metrics)
        metrics.pop("per_request_latencies_ms", None)
        metrics.pop("per_request_ttft_ms", None)
        row.update(metrics)
        rows.append(row)
    return rows


def _flatten_training_result(result: Any) -> list[dict[str, Any]]:
    """Flatten a TrainingBenchmarkResult into a list of row dicts."""
    rows: list[dict[str, Any]] = []
    for cfg in result.configs:
        row: dict[str, Any] = {
            "model": result.model_name,
            "gpu": result.gpu_type,
        }
        cfg_dict = asdict(cfg)
        # Flatten metrics if nested
        metrics = cfg_dict.pop("metrics", {})
        row.update(cfg_dict)
        if isinstance(metrics, dict):
            row.update(metrics)
        rows.append(row)
    return rows


def _flatten_quality_result(result: Any) -> list[dict[str, Any]]:
    """Flatten a QualityEvalResult into a list of row dicts."""
    rows: list[dict[str, Any]] = []
    for bench in result.benchmarks:
        row: dict[str, Any] = {
            "model": result.model_name,
            "gpu": result.gpu_type,
        }
        bench_dict = asdict(bench)
        metrics = bench_dict.pop("metrics", {})
        row.update(bench_dict)
        if isinstance(metrics, dict):
            row.update(metrics)
        rows.append(row)
    return rows


_FLATTENERS = {
    "inference": _flatten_inference_result,
    "training": _flatten_training_result,
    "quality": _flatten_quality_result,
}


def save_phase_results(
    phase: str,
    result: Any,
    output_dir: Path,
) -> Path | None:
    """Save phase results as Parquet + CSV under output_dir/raw/{phase}/.

    Returns the path to the Parquet file, or None if there were no results.
    """
    flattener = _FLATTENERS.get(phase)
    if flattener is None:
        logger.warning("no_flattener_for_phase", phase=phase)
        return None

    rows = flattener(result)
    if not rows:
        logger.warning("no_results_to_save", phase=phase)
        return None

    df = pd.DataFrame(rows)

    # Determine model/gpu from the result for the filename
    model_name = getattr(result, "model_name", "unknown")
    gpu_type = getattr(result, "gpu_type", "unknown")
    ts = _timestamp()

    raw_dir = output_dir / "raw" / phase
    raw_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{model_name}_{gpu_type}_{ts}"
    parquet_path = raw_dir / f"{stem}.parquet"
    csv_path = raw_dir / f"{stem}.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    logger.info(
        "results_saved",
        phase=phase,
        parquet=str(parquet_path),
        csv=str(csv_path),
        rows=len(df),
    )

    # Also save per-request latencies as a separate JSON for detailed analysis
    if phase == "inference":
        _save_per_request_latencies(result, raw_dir, stem)

    return parquet_path


def _save_per_request_latencies(result: Any, raw_dir: Path, stem: str) -> None:
    """Save per-request latency data for detailed analysis."""
    latency_data: list[dict[str, Any]] = []
    for cfg in result.configs:
        for i, (lat, ttft) in enumerate(
            zip(cfg.metrics.per_request_latencies_ms, cfg.metrics.per_request_ttft_ms)
        ):
            latency_data.append(
                {
                    "resolution": cfg.resolution,
                    "batch_size": cfg.batch_size,
                    "precision": cfg.precision,
                    "max_output_tokens": cfg.max_output_tokens,
                    "run_id": cfg.run_id,
                    "request_idx": i,
                    "latency_ms": lat,
                    "ttft_ms": ttft,
                }
            )

    if latency_data:
        path = raw_dir / f"{stem}_per_request.json"
        with open(path, "w") as f:
            json.dump(latency_data, f, indent=2)
