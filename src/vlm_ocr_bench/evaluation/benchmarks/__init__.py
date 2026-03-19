"""Benchmark-specific evaluation pipelines."""

from vlm_ocr_bench.evaluation.benchmarks.olmocr_bench import (
    OlmOCRBenchResult,
    evaluate_olmocr_bench,
)
from vlm_ocr_bench.evaluation.benchmarks.omnidocbench import (
    OmniDocBenchResult,
    evaluate_omnidocbench,
)
from vlm_ocr_bench.evaluation.benchmarks.real5 import (
    Real5Result,
    evaluate_real5,
)

__all__ = [
    "OlmOCRBenchResult",
    "OmniDocBenchResult",
    "Real5Result",
    "evaluate_olmocr_bench",
    "evaluate_omnidocbench",
    "evaluate_real5",
]
