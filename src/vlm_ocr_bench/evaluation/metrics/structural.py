"""Structural accuracy metrics for tables and formulas."""

from __future__ import annotations

import re
import unicodedata

from vlm_ocr_bench.evaluation.metrics.cdm import compute_cdm
from vlm_ocr_bench.evaluation.metrics.edit_distance import normalized_edit_distance


def _normalize(text: str) -> str:
    """Unicode-normalize to NFKC."""
    return unicodedata.normalize("NFKC", text)


def _parse_markdown_table(table_text: str) -> list[list[str]]:
    """Parse a Markdown table into a 2D list of cell strings.

    Skips separator rows (e.g., |---|---|).
    """
    rows: list[list[str]] = []
    for line in table_text.strip().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        # Skip separator rows
        if re.match(r"^\|[\s\-:|]+\|$", line):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if cells:
            rows.append(cells)
    return rows


def table_structural_accuracy(
    pred_table: str,
    ref_table: str,
    format: str = "markdown",
) -> dict[str, float]:
    """Compute structural accuracy metrics for a table.

    Returns dict with:
        row_count_match: 1.0 if row counts match, else 0.0
        col_count_match: 1.0 if column counts match, else 0.0
        cell_content_accuracy: mean normalized edit distance across aligned cells
        structure_f1: F1 combining row/col precision and recall
    """
    pred_rows = _parse_markdown_table(pred_table)
    ref_rows = _parse_markdown_table(ref_table)

    if not pred_rows and not ref_rows:
        return {
            "row_count_match": 1.0,
            "col_count_match": 1.0,
            "cell_content_accuracy": 1.0,
            "structure_f1": 1.0,
        }
    if not pred_rows or not ref_rows:
        return {
            "row_count_match": 0.0,
            "col_count_match": 0.0,
            "cell_content_accuracy": 0.0,
            "structure_f1": 0.0,
        }

    n_pred_rows = len(pred_rows)
    n_ref_rows = len(ref_rows)
    n_pred_cols = max(len(r) for r in pred_rows) if pred_rows else 0
    n_ref_cols = max(len(r) for r in ref_rows) if ref_rows else 0

    row_match = 1.0 if n_pred_rows == n_ref_rows else 0.0
    col_match = 1.0 if n_pred_cols == n_ref_cols else 0.0

    # Cell content accuracy: compare aligned cells
    cell_scores: list[float] = []
    for i in range(min(n_pred_rows, n_ref_rows)):
        pred_cells = pred_rows[i]
        ref_cells = ref_rows[i]
        for j in range(min(len(pred_cells), len(ref_cells))):
            cell_scores.append(normalized_edit_distance(pred_cells[j], ref_cells[j]))

    cell_accuracy = sum(cell_scores) / len(cell_scores) if cell_scores else 0.0

    # Structure F1: precision/recall on cell positions
    pred_cell_count = sum(len(r) for r in pred_rows)
    ref_cell_count = sum(len(r) for r in ref_rows)
    matched_cells = len(cell_scores)

    precision = matched_cells / pred_cell_count if pred_cell_count > 0 else 0.0
    recall = matched_cells / ref_cell_count if ref_cell_count > 0 else 0.0
    structure_f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    )

    return {
        "row_count_match": row_match,
        "col_count_match": col_match,
        "cell_content_accuracy": cell_accuracy,
        "structure_f1": structure_f1,
    }


def formula_structural_accuracy(
    pred_latex: str,
    ref_latex: str,
) -> dict[str, float]:
    """Compute structural accuracy metrics for a LaTeX formula.

    Returns dict with:
        exact_match: 1.0 if NFKC-normalized strings match exactly
        normalized_edit_distance: similarity score [0, 1]
        cdm_score: CDM token-level F1
        token_level_f1: same as cdm_score (fallback implementation)
    """
    pred_norm = _normalize(pred_latex).strip()
    ref_norm = _normalize(ref_latex).strip()

    exact = 1.0 if pred_norm == ref_norm else 0.0
    ned = normalized_edit_distance(pred_latex, ref_latex)
    cdm = compute_cdm(pred_latex, ref_latex)

    return {
        "exact_match": exact,
        "normalized_edit_distance": ned,
        "cdm_score": cdm,
        "token_level_f1": cdm,
    }
