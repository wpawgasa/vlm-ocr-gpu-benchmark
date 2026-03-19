"""Language-aware BLEU score computation."""

from __future__ import annotations

from vlm_ocr_bench.evaluation.metrics._utils import _ensure_nltk_data, _normalize, _tokenize

__all__ = ["compute_bleu"]


def compute_bleu(
    prediction: str,
    reference: str,
    language: str = "en",
    n_gram: int = 4,
) -> float:
    """Compute BLEU score with language-aware tokenization.

    Args:
        prediction: predicted text
        reference: reference text
        language: language code for tokenization
        n_gram: maximum n-gram order (default 4 for standard BLEU-4)

    Returns a score in [0.0, 1.0] where 1.0 = perfect match.
    """
    import sacrebleu

    pred_tokens = _tokenize(prediction, language)
    ref_tokens = _tokenize(reference, language)

    if not pred_tokens or not ref_tokens:
        return 0.0

    # sacrebleu expects detokenized strings for its internal tokenization,
    # but since we handle tokenization ourselves, join tokens with spaces
    pred_str = " ".join(pred_tokens)
    ref_str = " ".join(ref_tokens)

    bleu_metric = sacrebleu.BLEU(max_ngram_order=n_gram, tokenize="none")
    result = bleu_metric.sentence_score(pred_str, [ref_str])
    return float(result.score) / 100.0  # sacrebleu returns 0-100
