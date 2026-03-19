"""Quality evaluation runner — orchestrates the full Phase C evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from vlm_ocr_bench.config.schema import (
    GPUConfig,
    ModelConfig,
    PrecisionMode,
    QualityConfig,
)
from vlm_ocr_bench.data.datasets import get_loader
from vlm_ocr_bench.evaluation.benchmarks.olmocr_bench import (
    evaluate_olmocr_bench,
)
from vlm_ocr_bench.evaluation.benchmarks.omnidocbench import (
    evaluate_omnidocbench,
)
from vlm_ocr_bench.evaluation.benchmarks.real5 import (
    evaluate_real5,
)
from vlm_ocr_bench.inference.engine import VLLMEngine
from vlm_ocr_bench.inference.vllm_config import build_vllm_engine_args, validate_vllm_config
from vlm_ocr_bench.models.registry import get_adapter

logger = structlog.get_logger()


@dataclass
class PrecisionDelta:
    """Quality degradation from BF16 baseline to a lower precision."""

    precision: str
    baseline_precision: str = "bf16"
    edit_distance_delta: float = 0.0
    bleu_delta: float = 0.0
    meteor_delta: float = 0.0


@dataclass
class PrecisionComparison:
    """Cross-precision quality comparison for a single benchmark."""

    benchmark: str
    baseline_precision: str = "bf16"
    deltas: list[PrecisionDelta] = field(default_factory=list)


@dataclass
class BenchmarkQuality:
    """Quality results for a single benchmark at a single precision."""

    benchmark: str
    precision: str
    edit_distance: float = 0.0
    bleu: float = 0.0
    meteor: float = 0.0
    num_samples: int = 0
    num_failed: int = 0
    details: Any = None  # benchmark-specific result object


@dataclass
class QualityEvalResult:
    """Complete result from a Phase C quality evaluation."""

    model_name: str
    gpu_type: str
    benchmarks: list[BenchmarkQuality] = field(default_factory=list)
    precision_comparisons: list[PrecisionComparison] = field(default_factory=list)
    total_wall_time_s: float = 0.0


class QualityEvalRunner:
    """Orchestrates the full quality evaluation sweep.

    Algorithm:
    1. For each benchmark:
       - Load dataset
       - For each precision (bf16, fp8, fp4):
         - Initialize inference engine
         - Run inference on all samples
         - Parse outputs via adapter
         - Compute all quality metrics
         - Breakdown by document type (if enabled)
       - Compare across precisions (degradation analysis)
    2. Generate quality comparison tables
    """

    def __init__(
        self,
        model_config: ModelConfig,
        gpu_config: GPUConfig,
        quality_config: QualityConfig,
    ) -> None:
        self._model_config = model_config
        self._gpu_config = gpu_config
        self._quality_config = quality_config
        self._adapter = get_adapter(model_config.name)

    def run(self) -> QualityEvalResult:
        """Execute the full quality evaluation sweep."""
        total_start = time.monotonic()

        result = QualityEvalResult(
            model_name=self._model_config.name,
            gpu_type=self._gpu_config.gpu_type.value,
        )

        for benchmark_name in self._quality_config.benchmarks:
            logger.info("evaluating_benchmark", benchmark=benchmark_name)

            try:
                self._evaluate_benchmark(benchmark_name, result)
            except Exception:
                logger.error(
                    "benchmark_evaluation_failed",
                    benchmark=benchmark_name,
                    exc_info=True,
                )

        # Generate precision comparisons
        result.precision_comparisons = self._compute_precision_comparisons(result.benchmarks)

        result.total_wall_time_s = time.monotonic() - total_start
        logger.info(
            "quality_evaluation_complete",
            model=self._model_config.name,
            num_benchmarks=len(self._quality_config.benchmarks),
            total_time_s=round(result.total_wall_time_s, 1),
        )
        return result

    def _evaluate_benchmark(
        self,
        benchmark_name: str,
        result: QualityEvalResult,
    ) -> None:
        """Evaluate a single benchmark across all precision modes."""
        # Load dataset
        loader = get_loader(benchmark_name)
        samples = loader.get_samples(max_samples=self._quality_config.max_samples)

        if not samples:
            logger.warning("no_samples_loaded", benchmark=benchmark_name)
            return

        # Build references
        references: dict[str, str] = {}
        if hasattr(loader, "get_ground_truth"):
            for sample in samples:
                try:
                    gt = loader.get_ground_truth(sample.sample_id)
                    references[sample.sample_id] = gt.markdown
                except KeyError:
                    pass
        else:
            for sample in samples:
                gt_text = sample.attributes.get(
                    "markdown",
                    sample.attributes.get("ground_truth", ""),
                )
                if gt_text:
                    references[sample.sample_id] = str(gt_text)

        if not references:
            logger.warning("no_references_found", benchmark=benchmark_name)
            return

        for precision in self._quality_config.precision_modes:
            logger.info(
                "evaluating_precision",
                benchmark=benchmark_name,
                precision=precision.value,
            )

            try:
                bq = self._evaluate_precision(benchmark_name, precision, samples, references)
                result.benchmarks.append(bq)
            except Exception:
                logger.error(
                    "precision_evaluation_failed",
                    benchmark=benchmark_name,
                    precision=precision.value,
                    exc_info=True,
                )

    def _evaluate_precision(
        self,
        benchmark_name: str,
        precision: PrecisionMode,
        samples: list[Any],
        references: dict[str, str],
    ) -> BenchmarkQuality:
        """Evaluate one benchmark at one precision level."""
        # Validate config
        issues = validate_vllm_config(self._model_config, self._gpu_config, precision)
        if issues:
            logger.warning(
                "skipping_precision",
                precision=precision.value,
                issues=issues,
            )
            return BenchmarkQuality(
                benchmark=benchmark_name,
                precision=precision.value,
                num_samples=0,
                num_failed=0,
            )

        # Build and initialize engine
        engine_args = build_vllm_engine_args(
            self._model_config,
            self._gpu_config,
            precision,
            self._adapter,
        )
        engine = VLLMEngine(engine_args)

        try:
            engine.initialize()
            predictions = self._run_inference(engine, samples)
        finally:
            engine.shutdown()

        # Evaluate predictions against references
        return self._score_predictions(benchmark_name, precision, predictions, references, samples)

    def _run_inference(
        self,
        engine: VLLMEngine,
        samples: list[Any],
    ) -> dict[str, str]:
        """Run inference on all samples and return predictions."""
        from vllm import SamplingParams

        predictions: dict[str, str] = {}
        sampling_params = SamplingParams(
            max_tokens=self._model_config.max_output_tokens,
            temperature=0.0,
        )

        for sample in samples:
            try:
                prompt_data = self._adapter.build_prompt(sample.image)
                prompt = prompt_data.get("prompt", prompt_data.get("messages", ""))
                images = prompt_data.get("images", [])

                workload = [
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"image": images[0]} if images else {},
                    }
                ]

                results = engine.generate_batch(workload, sampling_params)
                if results:
                    predictions[sample.sample_id] = results[0].output_text
            except Exception:
                logger.debug(
                    "inference_failed",
                    sample_id=sample.sample_id,
                    exc_info=True,
                )

        return predictions

    def _score_predictions(
        self,
        benchmark_name: str,
        precision: PrecisionMode,
        predictions: dict[str, str],
        references: dict[str, str],
        samples: list[Any],
    ) -> BenchmarkQuality:
        """Score predictions using the appropriate benchmark evaluator."""
        if benchmark_name == "omnidocbench_v1.5":
            detail = evaluate_omnidocbench(
                predictions,
                references,
                samples,
                per_document_type=self._quality_config.per_document_type,
            )
            return BenchmarkQuality(
                benchmark=benchmark_name,
                precision=precision.value,
                edit_distance=detail.overall_edit_distance,
                bleu=detail.overall_bleu,
                meteor=detail.overall_meteor,
                num_samples=detail.num_samples,
                num_failed=detail.num_failed,
                details=detail,
            )

        if benchmark_name == "olmocr_bench":
            detail_olm = evaluate_olmocr_bench(predictions, references)
            return BenchmarkQuality(
                benchmark=benchmark_name,
                precision=precision.value,
                edit_distance=detail_olm.overall_edit_distance,
                bleu=detail_olm.overall_bleu,
                meteor=detail_olm.overall_meteor,
                num_samples=detail_olm.num_samples,
                num_failed=detail_olm.num_failed,
                details=detail_olm,
            )

        if benchmark_name in ("real5_omnidocbench", "real5"):
            detail_r5 = evaluate_real5(predictions, references, samples)
            return BenchmarkQuality(
                benchmark=benchmark_name,
                precision=precision.value,
                edit_distance=detail_r5.overall_edit_distance,
                bleu=detail_r5.overall_bleu,
                meteor=detail_r5.overall_meteor,
                num_samples=detail_r5.num_samples,
                num_failed=detail_r5.num_failed,
                details=detail_r5,
            )

        # Generic fallback: use olmocr_bench evaluator
        detail_generic = evaluate_olmocr_bench(predictions, references)
        return BenchmarkQuality(
            benchmark=benchmark_name,
            precision=precision.value,
            edit_distance=detail_generic.overall_edit_distance,
            bleu=detail_generic.overall_bleu,
            meteor=detail_generic.overall_meteor,
            num_samples=detail_generic.num_samples,
            num_failed=detail_generic.num_failed,
            details=detail_generic,
        )

    def _compute_precision_comparisons(
        self,
        benchmarks: list[BenchmarkQuality],
    ) -> list[PrecisionComparison]:
        """Compare quality across precisions, using BF16 as baseline."""
        # Group by benchmark
        by_benchmark: dict[str, dict[str, BenchmarkQuality]] = {}
        for bq in benchmarks:
            by_benchmark.setdefault(bq.benchmark, {})[bq.precision] = bq

        comparisons: list[PrecisionComparison] = []
        for bench_name, precision_map in by_benchmark.items():
            baseline = precision_map.get("bf16")
            if baseline is None:
                continue

            comparison = PrecisionComparison(benchmark=bench_name)
            for prec, bq in precision_map.items():
                if prec == "bf16":
                    continue
                comparison.deltas.append(
                    PrecisionDelta(
                        precision=prec,
                        edit_distance_delta=bq.edit_distance - baseline.edit_distance,
                        bleu_delta=bq.bleu - baseline.bleu,
                        meteor_delta=bq.meteor - baseline.meteor,
                    )
                )
            comparisons.append(comparison)

        return comparisons
