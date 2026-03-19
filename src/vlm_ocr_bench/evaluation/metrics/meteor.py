"""Language-aware METEOR score computation."""

from __future__ import annotations

import unicodedata

from vlm_ocr_bench.evaluation.metrics.bleu import _ensure_nltk_data


def _normalize(text: str) -> str:
    """Unicode-normalize to NFKC."""
    return unicodedata.normalize("NFKC", text)


def _tokenize(text: str, language: str) -> list[str]:
    """Tokenize text based on language (same strategy as BLEU)."""
    text = _normalize(text)
    if not text.strip():
        return []

    if language == "th":
        from pythainlp.tokenize import word_tokenize as thai_tokenize

        return list(thai_tokenize(text, engine="newmm"))

    if language in ("zh", "cn", "chinese"):
        return list(text.replace(" ", ""))

    _ensure_nltk_data()
    from nltk.tokenize import word_tokenize

    return list(word_tokenize(text))


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
