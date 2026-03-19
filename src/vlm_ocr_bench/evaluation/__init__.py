"""Quality evaluation (Phase C): OCR quality metrics and benchmark pipelines."""

from vlm_ocr_bench.evaluation.metrics import (
    block_level_edit_distance,
    compute_bleu,
    compute_cdm,
    compute_meteor,
    formula_structural_accuracy,
    normalized_edit_distance,
    table_structural_accuracy,
)
from vlm_ocr_bench.evaluation.parsers import Block, BlockType, parse_markdown
from vlm_ocr_bench.evaluation.runner import (
    BenchmarkQuality,
    PrecisionComparison,
    PrecisionDelta,
    QualityEvalResult,
    QualityEvalRunner,
)

__all__ = [
    "BenchmarkQuality",
    "Block",
    "BlockType",
    "PrecisionComparison",
    "PrecisionDelta",
    "QualityEvalResult",
    "QualityEvalRunner",
    "block_level_edit_distance",
    "compute_bleu",
    "compute_cdm",
    "compute_meteor",
    "formula_structural_accuracy",
    "normalized_edit_distance",
    "parse_markdown",
    "table_structural_accuracy",
]
