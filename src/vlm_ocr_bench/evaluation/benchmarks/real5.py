"""Real5-OmniDocBench evaluation pipeline — per-scenario scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from vlm_ocr_bench.evaluation.metrics.bleu import compute_bleu
from vlm_ocr_bench.evaluation.metrics.edit_distance import normalized_edit_distance
from vlm_ocr_bench.evaluation.metrics.meteor import compute_meteor

logger = structlog.get_logger()

# Real5 scenarios
SCENARIOS = ["scanning", "warping", "screen_photo", "illumination", "skew"]


@dataclass
class ScenarioScore:
    """Scores for a specific capture scenario."""

    scenario: str
    edit_distance: float = 0.0
    bleu: float = 0.0
    meteor: float = 0.0
    num_samples: int = 0
    num_failed: int = 0


@dataclass
class Real5Result:
    """Complete Real5-OmniDocBench evaluation result."""

    overall_edit_distance: float = 0.0
    overall_bleu: float = 0.0
    overall_meteor: float = 0.0
    num_samples: int = 0
    num_failed: int = 0
    scenario_scores: dict[str, ScenarioScore] = field(default_factory=dict)


def _get_scenario(sample: Any | None) -> str:
    """Extract scenario from sample attributes."""
    if sample is None:
        return "unknown"
    return str(sample.attributes.get("scenario", sample.doc_type))


def evaluate_real5(
    predictions: dict[str, str],
    references: dict[str, str],
    samples: list[Any],
) -> Real5Result:
    """Run Real5-OmniDocBench evaluation pipeline.

    Scores are broken down by capture scenario (scanning, warping,
    screen_photo, illumination, skew).

    Args:
        predictions: {sample_id: predicted_text}
        references: {sample_id: reference_text}
        samples: list of Any for scenario metadata

    Returns:
        Real5Result with overall and per-scenario scores.
    """
    result = Real5Result()
    sample_map = {s.sample_id: s for s in samples}

    all_ed: list[float] = []
    all_bleu: list[float] = []
    all_meteor: list[float] = []
    scenario_accum: dict[str, dict[str, list[float]]] = {}

    for sample_id, ref_text in references.items():
        pred_text = predictions.get(sample_id, "")
        sample = sample_map.get(sample_id)
        language = sample.language if sample else "en"
        scenario = _get_scenario(sample)

        result.num_samples += 1

        if scenario not in scenario_accum:
            scenario_accum[scenario] = {"ed": [], "bleu": [], "meteor": [], "n_failed": []}

        if not pred_text.strip():
            result.num_failed += 1
            scenario_accum[scenario]["n_failed"].append(1.0)
            continue

        ed = normalized_edit_distance(pred_text, ref_text)
        bleu_score = compute_bleu(pred_text, ref_text, language)
        meteor_score = compute_meteor(pred_text, ref_text, language)

        all_ed.append(ed)
        all_bleu.append(bleu_score)
        all_meteor.append(meteor_score)

        scenario_accum[scenario]["ed"].append(ed)
        scenario_accum[scenario]["bleu"].append(bleu_score)
        scenario_accum[scenario]["meteor"].append(meteor_score)

    # Overall scores
    result.overall_edit_distance = sum(all_ed) / len(all_ed) if all_ed else 0.0
    result.overall_bleu = sum(all_bleu) / len(all_bleu) if all_bleu else 0.0
    result.overall_meteor = sum(all_meteor) / len(all_meteor) if all_meteor else 0.0

    # Per-scenario scores
    for scenario, accum in scenario_accum.items():
        n = len(accum["ed"])
        n_failed = len(accum["n_failed"])
        result.scenario_scores[scenario] = ScenarioScore(
            scenario=scenario,
            edit_distance=sum(accum["ed"]) / n if n else 0.0,
            bleu=sum(accum["bleu"]) / n if n else 0.0,
            meteor=sum(accum["meteor"]) / n if n else 0.0,
            num_samples=n + n_failed,
            num_failed=n_failed,
        )

    logger.info(
        "real5_evaluation_complete",
        num_samples=result.num_samples,
        num_failed=result.num_failed,
        num_scenarios=len(result.scenario_scores),
        overall_edit_distance=round(result.overall_edit_distance, 4),
    )
    return result
