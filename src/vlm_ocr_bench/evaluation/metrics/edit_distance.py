"""Edit distance metrics — normalized and block-level with Hungarian matching."""

from __future__ import annotations

import editdistance
import numpy as np
from scipy.optimize import linear_sum_assignment

from vlm_ocr_bench.evaluation.metrics._utils import _normalize

__all__ = ["block_level_edit_distance", "normalized_edit_distance"]


def normalized_edit_distance(prediction: str, reference: str) -> float:
    """Compute normalized edit distance between two strings.

    Returns a score in [0.0, 1.0] where 1.0 = perfect match.
    Both strings are NFKC-normalized before comparison.
    """
    pred = _normalize(prediction)
    ref = _normalize(reference)

    if not pred and not ref:
        return 1.0
    max_len = max(len(pred), len(ref))
    if max_len == 0:
        return 1.0

    dist: int = editdistance.eval(pred, ref)
    return 1.0 - (dist / max_len)


def block_level_edit_distance(
    pred_blocks: list[str],
    ref_blocks: list[str],
    match_strategy: str = "hungarian",
) -> float:
    """Compute block-level edit distance using Hungarian matching.

    Aligns predicted blocks to reference blocks via optimal assignment,
    then averages normalized edit distance across matched pairs.
    Unmatched blocks (count difference) are penalized as zero-similarity.

    Returns a score in [0.0, 1.0] where 1.0 = perfect match.
    """
    if not pred_blocks and not ref_blocks:
        return 1.0
    if not pred_blocks or not ref_blocks:
        return 0.0

    n_pred = len(pred_blocks)
    n_ref = len(ref_blocks)

    # Build cost matrix: cost = 1 - normalized_edit_distance (so lower = better match)
    cost = np.zeros((n_pred, n_ref), dtype=np.float64)
    for i, pb in enumerate(pred_blocks):
        for j, rb in enumerate(ref_blocks):
            cost[i, j] = 1.0 - normalized_edit_distance(pb, rb)

    if match_strategy == "hungarian":
        row_ind, col_ind = linear_sum_assignment(cost)
    else:
        # Greedy fallback: match in order
        n_match = min(n_pred, n_ref)
        row_ind = np.arange(n_match)
        col_ind = np.arange(n_match)

    # Average similarity across matched pairs, with unmatched blocks scoring 0
    matched_scores = [1.0 - cost[r, c] for r, c in zip(row_ind, col_ind, strict=True)]
    n_unmatched = abs(n_pred - n_ref)
    # total_count includes unmatched blocks (each contributing 0 to the sum)
    total_count = len(matched_scores) + n_unmatched

    return sum(matched_scores) / total_count if total_count > 0 else 0.0
