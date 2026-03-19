"""olmOCR-Bench evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from vlm_ocr_bench.evaluation.metrics.bleu import compute_bleu
from vlm_ocr_bench.evaluation.metrics.edit_distance import normalized_edit_distance
from vlm_ocr_bench.evaluation.metrics.meteor import compute_meteor

logger = structlog.get_logger()


@dataclass
class OlmOCRBenchResult:
    """Complete olmOCR-Bench evaluation result."""

    overall_edit_distance: float = 0.0
    overall_bleu: float = 0.0
    overall_meteor: float = 0.0
    num_samples: int = 0
    num_failed: int = 0
    per_sample_scores: list[dict[str, float]] = field(default_factory=list)


def evaluate_olmocr_bench(
    predictions: dict[str, str],
    references: dict[str, str],
    language: str = "en",
) -> OlmOCRBenchResult:
    """Run olmOCR-Bench evaluation pipeline.

    Args:
        predictions: {sample_id: predicted_text}
        references: {sample_id: reference_text}
        language: language for tokenization

    Returns:
        OlmOCRBenchResult with overall scores.
    """
    result = OlmOCRBenchResult()

    all_ed: list[float] = []
    all_bleu: list[float] = []
    all_meteor: list[float] = []

    for sample_id, ref_text in references.items():
        pred_text = predictions.get(sample_id, "")
        result.num_samples += 1

        if not pred_text.strip():
            result.num_failed += 1
            result.per_sample_scores.append(
                {
                    "sample_id_hash": hash(sample_id),
                    "edit_distance": 0.0,
                    "bleu": 0.0,
                    "meteor": 0.0,
                }
            )
            continue

        ed = normalized_edit_distance(pred_text, ref_text)
        bleu = compute_bleu(pred_text, ref_text, language)
        meteor = compute_meteor(pred_text, ref_text, language)

        all_ed.append(ed)
        all_bleu.append(bleu)
        all_meteor.append(meteor)

        result.per_sample_scores.append(
            {"sample_id_hash": hash(sample_id), "edit_distance": ed, "bleu": bleu, "meteor": meteor}
        )

    result.overall_edit_distance = sum(all_ed) / len(all_ed) if all_ed else 0.0
    result.overall_bleu = sum(all_bleu) / len(all_bleu) if all_bleu else 0.0
    result.overall_meteor = sum(all_meteor) / len(all_meteor) if all_meteor else 0.0

    logger.info(
        "olmocr_bench_evaluation_complete",
        num_samples=result.num_samples,
        num_failed=result.num_failed,
        overall_edit_distance=round(result.overall_edit_distance, 4),
    )
    return result
