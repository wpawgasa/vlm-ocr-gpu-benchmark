"""Inference benchmark runner — orchestrates the full Phase A sweep."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from vlm_ocr_bench.config.schema import (
    GPUConfig,
    InferenceConfig,
    ModelConfig,
    PrecisionMode,
)
from vlm_ocr_bench.inference.engine import EngineInfo, GenerationResult, VLLMEngine
from vlm_ocr_bench.inference.metrics import InferenceMetrics, compute_metrics
from vlm_ocr_bench.inference.vllm_config import build_vllm_engine_args, validate_vllm_config
from vlm_ocr_bench.inference.workload import build_workload_batch
from vlm_ocr_bench.models.registry import get_adapter

logger = structlog.get_logger()


@dataclass
class SingleConfigResult:
    """Result from a single (resolution, batch_size, precision, max_tokens, run_id) config."""

    resolution: int
    batch_size: int
    precision: str
    max_output_tokens: int
    run_id: int
    metrics: InferenceMetrics
    num_requests: int = 0
    wall_time_s: float = 0.0
    oom: bool = False
    error: str | None = None


@dataclass
class InferenceBenchmarkResult:
    """Complete result from a Phase A inference benchmark."""

    model_name: str
    gpu_type: str
    # Keyed by precision value (e.g. "bf16", "fp8") because a new engine is
    # initialised for each precision; storing all infos avoids overwriting.
    engine_infos: dict[str, EngineInfo] = field(default_factory=dict)
    configs: list[SingleConfigResult] = field(default_factory=list)
    total_wall_time_s: float = 0.0


class InferenceBenchmarkRunner:
    """Orchestrates the full inference benchmark sweep.

    For each (resolution, batch_size, precision, max_tokens) combo:
    1. Build vLLM engine
    2. Warmup
    3. Run measurement requests (multiple runs)
    4. Collect metrics
    5. Cooldown
    """

    def __init__(
        self,
        model_config: ModelConfig,
        gpu_config: GPUConfig,
        inference_config: InferenceConfig,
    ) -> None:
        self._model_config = model_config
        self._gpu_config = gpu_config
        self._inference_config = inference_config
        self._adapter = get_adapter(model_config.name)

    def run(self) -> InferenceBenchmarkResult:
        """Execute the full inference benchmark sweep."""
        total_start = time.monotonic()

        result = InferenceBenchmarkResult(
            model_name=self._model_config.name,
            gpu_type=self._gpu_config.gpu_type.value,
        )

        for precision in self._inference_config.precision_modes:
            # Validate config viability
            issues = validate_vllm_config(self._model_config, self._gpu_config, precision)
            if issues:
                logger.warning(
                    "skipping_precision",
                    precision=precision.value,
                    issues=issues,
                )
                continue

            # Build engine for this precision
            engine_args = build_vllm_engine_args(
                self._model_config,
                self._gpu_config,
                precision,
                self._adapter,
            )

            # Merge user-provided vllm_args
            for k, v in self._inference_config.vllm_args.items():
                engine_args[k] = v

            engine = VLLMEngine(engine_args)

            try:
                engine_info = engine.initialize()
                result.engine_infos[precision.value] = engine_info
            except Exception:
                logger.error(
                    "engine_init_failed",
                    model=self._model_config.name,
                    precision=precision.value,
                    exc_info=True,
                )
                continue

            try:
                self._run_precision_sweep(engine, precision, result)
            finally:
                engine.shutdown()

        result.total_wall_time_s = time.monotonic() - total_start
        logger.info(
            "inference_benchmark_complete",
            model=self._model_config.name,
            num_configs=len(result.configs),
            total_time_s=round(result.total_wall_time_s, 1),
        )
        return result

    def _run_precision_sweep(
        self,
        engine: VLLMEngine,
        precision: PrecisionMode,
        result: InferenceBenchmarkResult,
    ) -> None:
        """Run all resolution/batch_size/max_tokens combos for one precision."""
        cfg = self._inference_config

        for resolution in cfg.resolutions:
            # Skip resolutions not supported by this model
            if resolution not in self._model_config.supported_resolutions:
                logger.debug(
                    "skipping_unsupported_resolution",
                    resolution=resolution,
                    model=self._model_config.name,
                )
                continue

            for max_tokens in cfg.max_output_tokens_sweep:
                for batch_size in cfg.batch_sizes:
                    for run_id in range(cfg.runs_per_config):
                        config_result = self._run_single_config(
                            engine=engine,
                            resolution=resolution,
                            batch_size=batch_size,
                            precision=precision,
                            max_tokens=max_tokens,
                            run_id=run_id,
                        )
                        result.configs.append(config_result)

                    # Cooldown between batch size configs
                    if cfg.cooldown_seconds > 0:
                        logger.debug(
                            "cooldown",
                            seconds=cfg.cooldown_seconds,
                        )
                        time.sleep(cfg.cooldown_seconds)

    def _run_single_config(
        self,
        engine: VLLMEngine,
        resolution: int,
        batch_size: int,
        precision: PrecisionMode,
        max_tokens: int,
        run_id: int,
    ) -> SingleConfigResult:
        """Run a single config: warmup + measurement."""
        logger.info(
            "running_config",
            resolution=resolution,
            batch_size=batch_size,
            precision=precision.value,
            max_tokens=max_tokens,
            run_id=run_id,
        )

        try:
            return self._execute_config(
                engine, resolution, batch_size, precision, max_tokens, run_id
            )
        except Exception as exc:
            oom = "out of memory" in str(exc).lower()
            logger.warning(
                "config_failed",
                resolution=resolution,
                batch_size=batch_size,
                error=str(exc),
                oom=oom,
            )
            return SingleConfigResult(
                resolution=resolution,
                batch_size=batch_size,
                precision=precision.value,
                max_output_tokens=max_tokens,
                run_id=run_id,
                metrics=InferenceMetrics(),
                oom=oom,
                error=str(exc),
            )

    def _execute_config(
        self,
        engine: VLLMEngine,
        resolution: int,
        batch_size: int,
        precision: PrecisionMode,
        max_tokens: int,
        run_id: int,
    ) -> SingleConfigResult:
        """Execute warmup + measurement for a single config."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)

        # Build workload
        workload = build_workload_batch(
            adapter=self._adapter,
            batch_size=batch_size,
            resolution=resolution,
            requires_padding=self._model_config.requires_padding,
        )

        # Warmup: run_id resets to 0 for every (resolution, max_tokens, batch_size)
        # combination, so warmup fires once per unique config — not only globally.
        if run_id == 0 and self._inference_config.warmup_requests > 0:
            self._warmup(engine, workload, sampling_params)

        # Measurement
        all_results: list[GenerationResult] = []
        wall_start = time.monotonic()

        requests_done = 0
        while requests_done < self._inference_config.measurement_requests:
            batch_results = engine.generate_batch(workload, sampling_params)
            if not batch_results:
                logger.error(
                    "generate_batch_returned_empty",
                    requests_done=requests_done,
                    target=self._inference_config.measurement_requests,
                )
                break
            all_results.extend(batch_results)
            requests_done += len(batch_results)

        wall_time_s = time.monotonic() - wall_start

        # Collect per-request metrics
        latencies = [r.total_time_ms for r in all_results]
        ttfts = [r.ttft_ms for r in all_results]
        total_output_tokens = sum(r.num_output_tokens for r in all_results)

        # Get memory stats
        mem = engine.get_memory_stats()

        metrics = compute_metrics(
            per_request_latencies_ms=latencies,
            per_request_ttft_ms=ttfts,
            total_output_tokens=total_output_tokens,
            wall_time_s=wall_time_s,
            peak_gpu_memory_gb=mem.peak_allocated_gb,
        )

        return SingleConfigResult(
            resolution=resolution,
            batch_size=batch_size,
            precision=precision.value,
            max_output_tokens=max_tokens,
            run_id=run_id,
            metrics=metrics,
            num_requests=len(all_results),
            wall_time_s=wall_time_s,
        )

    def _warmup(
        self,
        engine: VLLMEngine,
        workload: list[dict[str, Any]],
        sampling_params: Any,
    ) -> None:
        """Run warmup requests (results discarded)."""
        warmup_count = self._inference_config.warmup_requests
        logger.info("warmup_start", num_requests=warmup_count)

        done = 0
        while done < warmup_count:
            results = engine.generate_batch(workload, sampling_params)
            if not results:
                logger.error("warmup_generate_batch_returned_empty")
                break
            done += len(workload)

        logger.info("warmup_complete", num_requests=done)
