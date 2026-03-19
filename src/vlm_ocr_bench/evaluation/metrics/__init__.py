"""Quality evaluation metrics — edit distance, BLEU, METEOR, CDM, structural."""

from vlm_ocr_bench.evaluation.metrics.bleu import compute_bleu
from vlm_ocr_bench.evaluation.metrics.cdm import compute_cdm
from vlm_ocr_bench.evaluation.metrics.edit_distance import (
    block_level_edit_distance,
    normalized_edit_distance,
)
from vlm_ocr_bench.evaluation.metrics.meteor import compute_meteor
from vlm_ocr_bench.evaluation.metrics.structural import (
    formula_structural_accuracy,
    table_structural_accuracy,
)

__all__ = [
    "block_level_edit_distance",
    "compute_bleu",
    "compute_cdm",
    "compute_meteor",
    "formula_structural_accuracy",
    "normalized_edit_distance",
    "table_structural_accuracy",
]
