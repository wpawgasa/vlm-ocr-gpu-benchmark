"""CDM (formula evaluation) — LaTeX token-level comparison fallback.

NOTE: `compute_cdm` is NOT the actual CDM (Character Detection Metric) from
the OmniDocBench paper. It is a token-level F1 fallback used when no external
CDM evaluator is available. Results produced by this function are NOT
comparable to published CDM baselines and should be labeled as
"token_level_f1" in any reporting context.
"""

from __future__ import annotations

import re

from vlm_ocr_bench.evaluation.metrics._utils import _normalize

__all__ = ["compute_cdm"]


def _tokenize_latex(latex: str) -> list[str]:
    """Tokenize LaTeX into meaningful tokens.

    Splits on commands (\\cmd), braces, operators, and whitespace.
    """
    latex = _normalize(latex).strip()
    if not latex:
        return []

    # Match LaTeX commands, numbers, letters, operators, braces
    pattern = r"\\[a-zA-Z]+|\\.|[0-9]+\.?[0-9]*|[a-zA-Z]|[{}()\[\]^_&=+\-*/|<>!,;:.]"
    return re.findall(pattern, latex)


def _token_level_f1(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    """Compute token-level F1 score between predicted and reference tokens."""
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_set: dict[str, int] = {}
    for t in pred_tokens:
        pred_set[t] = pred_set.get(t, 0) + 1

    ref_set: dict[str, int] = {}
    for t in ref_tokens:
        ref_set[t] = ref_set.get(t, 0) + 1

    # Count matching tokens (min of counts for each token type)
    matches = sum(min(pred_set.get(t, 0), c) for t, c in ref_set.items())

    precision = matches / len(pred_tokens) if pred_tokens else 0.0
    recall = matches / len(ref_tokens) if ref_tokens else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_cdm(
    prediction: str,
    reference: str,
) -> float:
    """Compute a token-level F1 score for LaTeX formula comparison.

    WARNING: This is a token-level F1 fallback, NOT the actual CDM
    (Character Detection Metric) from the OmniDocBench paper. Results
    are not comparable to published CDM baselines. Use an external CDM
    evaluator for benchmark-comparable scores.

    Returns a score in [0.0, 1.0] where 1.0 = perfect match.
    """
    pred_tokens = _tokenize_latex(prediction)
    ref_tokens = _tokenize_latex(reference)
    return _token_level_f1(pred_tokens, ref_tokens)
