"""Shared utilities for evaluation metrics."""

from __future__ import annotations

import unicodedata

_NLTK_READY = False


def _normalize(text: str) -> str:
    """Unicode-normalize to NFKC."""
    return unicodedata.normalize("NFKC", text)


def _ensure_nltk_data() -> None:
    """Download required NLTK data if not already present."""
    global _NLTK_READY
    if _NLTK_READY:
        return
    import nltk

    for resource in ("punkt_tab", "wordnet"):
        path = f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}"
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)
    _NLTK_READY = True


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
    _ensure_nltk_data()
    from nltk.tokenize import word_tokenize

    return list(word_tokenize(text))
