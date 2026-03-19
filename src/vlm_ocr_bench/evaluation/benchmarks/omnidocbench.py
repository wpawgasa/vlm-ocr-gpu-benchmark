"""OmniDocBench evaluation pipeline — per-document-type and per-element-type scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from vlm_ocr_bench.evaluation.metrics.bleu import compute_bleu
from vlm_ocr_bench.evaluation.metrics.edit_distance import (
    block_level_edit_distance,
    normalized_edit_distance,
)
from vlm_ocr_bench.evaluation.metrics.meteor import compute_meteor
from vlm_ocr_bench.evaluation.metrics.structural import (
    formula_structural_accuracy,
    table_structural_accuracy,
)
from vlm_ocr_bench.evaluation.parsers import (
    extract_formulas,
    extract_tables,
    parse_markdown,
)

logger = structlog.get_logger()


@dataclass
class ElementScore:
    """Scores for a specific element type (text, table, formula, etc.)."""

    element_type: str
    edit_distance: float = 0.0
    bleu: float = 0.0
    meteor: float = 0.0
    count: int = 0


@dataclass
class DocTypeScore:
    """Aggregated scores for a document type."""

    doc_type: str
    edit_distance: float = 0.0
    bleu: float = 0.0
    meteor: float = 0.0
    num_samples: int = 0
    num_failed: int = 0
    element_scores: dict[str, ElementScore] = field(default_factory=dict)


@dataclass
class OmniDocBenchResult:
    """Complete OmniDocBench evaluation result."""

    overall_edit_distance: float = 0.0
    overall_bleu: float = 0.0
    overall_meteor: float = 0.0
    num_samples: int = 0
    num_failed: int = 0
    doc_type_scores: dict[str, DocTypeScore] = field(default_factory=dict)
    element_scores: dict[str, ElementScore] = field(default_factory=dict)


def evaluate_sample(
    prediction: str,
    reference: str,
    language: str = "en",
) -> dict[str, float]:
    """Evaluate a single prediction against its reference.

    Returns dict with edit_distance, bleu, meteor scores.
    """
    scores: dict[str, float] = {}
    scores["edit_distance"] = normalized_edit_distance(prediction, reference)
    scores["bleu"] = compute_bleu(prediction, reference, language)
    scores["meteor"] = compute_meteor(prediction, reference, language)

    # Block-level edit distance
    pred_blocks = [b.content for b in parse_markdown(prediction)]
    ref_blocks = [b.content for b in parse_markdown(reference)]
    scores["block_edit_distance"] = block_level_edit_distance(pred_blocks, ref_blocks)

    # Table structural accuracy (average across matched tables)
    pred_tables = extract_tables(prediction)
    ref_tables = extract_tables(reference)
    if pred_tables and ref_tables:
        table_scores = []
        for pt, rt in zip(pred_tables, ref_tables, strict=False):
            ts = table_structural_accuracy(pt, rt)
            table_scores.append(ts["cell_content_accuracy"])
        scores["table_accuracy"] = sum(table_scores) / len(table_scores)
    else:
        scores["table_accuracy"] = 0.0 if ref_tables else 1.0

    # Formula accuracy
    pred_formulas = extract_formulas(prediction)
    ref_formulas = extract_formulas(reference)
    if pred_formulas and ref_formulas:
        formula_scores = []
        for pf, rf in zip(pred_formulas, ref_formulas, strict=False):
            fs = formula_structural_accuracy(pf, rf)
            formula_scores.append(fs["cdm_score"])
        scores["formula_accuracy"] = sum(formula_scores) / len(formula_scores)
    else:
        scores["formula_accuracy"] = 0.0 if ref_formulas else 1.0

    return scores


def evaluate_omnidocbench(
    predictions: dict[str, str],
    references: dict[str, str],
    samples: list[Any],
    per_document_type: bool = True,
) -> OmniDocBenchResult:
    """Run full OmniDocBench evaluation pipeline.

    Args:
        predictions: {sample_id: predicted_text}
        references: {sample_id: reference_text}
        samples: list of DocSample for metadata (doc_type, language, etc.)
        per_document_type: whether to compute per-doc-type breakdown

    Returns:
        OmniDocBenchResult with overall and per-type scores.
    """
    result = OmniDocBenchResult()
    sample_map = {s.sample_id: s for s in samples}

    all_ed: list[float] = []
    all_bleu: list[float] = []
    all_meteor: list[float] = []
    doc_type_accum: dict[str, dict[str, list[float]]] = {}

    for sample_id, ref_text in references.items():
        pred_text = predictions.get(sample_id, "")
        sample = sample_map.get(sample_id)
        language = sample.language if sample else "en"
        doc_type = sample.doc_type if sample else "unknown"

        result.num_samples += 1

        if not pred_text.strip():
            result.num_failed += 1
            logger.debug("empty_prediction", sample_id=sample_id)
            continue

        scores = evaluate_sample(pred_text, ref_text, language)

        all_ed.append(scores["edit_distance"])
        all_bleu.append(scores["bleu"])
        all_meteor.append(scores["meteor"])

        if per_document_type:
            if doc_type not in doc_type_accum:
                doc_type_accum[doc_type] = {"ed": [], "bleu": [], "meteor": [], "failed": []}
            doc_type_accum[doc_type]["ed"].append(scores["edit_distance"])
            doc_type_accum[doc_type]["bleu"].append(scores["bleu"])
            doc_type_accum[doc_type]["meteor"].append(scores["meteor"])

    # Overall scores
    result.overall_edit_distance = sum(all_ed) / len(all_ed) if all_ed else 0.0
    result.overall_bleu = sum(all_bleu) / len(all_bleu) if all_bleu else 0.0
    result.overall_meteor = sum(all_meteor) / len(all_meteor) if all_meteor else 0.0

    # Per-document-type scores
    for doc_type, accum in doc_type_accum.items():
        n = len(accum["ed"])
        result.doc_type_scores[doc_type] = DocTypeScore(
            doc_type=doc_type,
            edit_distance=sum(accum["ed"]) / n if n else 0.0,
            bleu=sum(accum["bleu"]) / n if n else 0.0,
            meteor=sum(accum["meteor"]) / n if n else 0.0,
            num_samples=n,
        )

    logger.info(
        "omnidocbench_evaluation_complete",
        num_samples=result.num_samples,
        num_failed=result.num_failed,
        overall_edit_distance=round(result.overall_edit_distance, 4),
        overall_bleu=round(result.overall_bleu, 4),
    )
    return result
