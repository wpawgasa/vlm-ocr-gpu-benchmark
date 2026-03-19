"""Language-aware BLEU score computation."""

from __future__ import annotations

import unicodedata


def _normalize(text: str) -> str:
    """Unicode-normalize to NFKC."""
    return unicodedata.normalize("NFKC", text)


def _tokenize(text: str, language: str) -> list[str]:
    """Tokenize text based on language.

    - Thai: pythainlp word segmentation
    - Chinese (zh): character-level tokenization
    - Others: nltk word_tokenize
    """
    text = _normalize(text)
    if not text.strip():
        return []

    if language == "th":
        from pythainlp.tokenize import word_tokenize as thai_tokenize

        return list(thai_tokenize(text, engine="newmm"))

    if language in ("zh", "cn", "chinese"):
        # Character-level tokenization for Chinese
        return list(text.replace(" ", ""))

    # Default: NLTK word tokenization
    from nltk.tokenize import word_tokenize

    return list(word_tokenize(text))


def compute_bleu(
    prediction: str,
    reference: str,
    language: str = "en",
    n_gram: int = 4,
) -> float:
    """Compute BLEU score with language-aware tokenization.

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

    bleu = sacrebleu.sentence_bleu(
        pred_str,
        [ref_str],
        tokenize="none",  # we already tokenized
    )
    return float(bleu.score) / 100.0  # sacrebleu returns 0-100
