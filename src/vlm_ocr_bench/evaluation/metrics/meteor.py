"""Language-aware METEOR score computation."""

from __future__ import annotations

from vlm_ocr_bench.evaluation.metrics._utils import _ensure_nltk_data, _normalize, _tokenize

__all__ = ["compute_meteor"]


def compute_meteor(
    prediction: str,
    reference: str,
    language: str = "en",
) -> float:
    """Compute METEOR score with language-aware tokenization.

    Returns a score in [0.0, 1.0] where 1.0 = perfect match.
    """
    _ensure_nltk_data()
    from nltk.translate.meteor_score import single_meteor_score

    pred_tokens = _tokenize(prediction, language)
    ref_tokens = _tokenize(reference, language)

    if not pred_tokens or not ref_tokens:
        return 0.0

    return float(single_meteor_score(ref_tokens, pred_tokens))
