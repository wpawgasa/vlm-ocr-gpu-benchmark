# Quality Evaluation Module (Phase C)

**Date**: 2026-03-19
**Branch**: `feature/quality-evaluation`

## Overview

Implements the Phase C quality evaluation module for measuring OCR quality across benchmarks and precision modes. Computes edit distance, BLEU, METEOR, CDM, and structural metrics with precision degradation analysis.

## Changes

### New Files

**Metrics** (`src/vlm_ocr_bench/evaluation/metrics/`):
- `edit_distance.py` — Normalized edit distance (NFKC) and block-level edit distance with Hungarian matching via `scipy.optimize.linear_sum_assignment`
- `bleu.py` — Language-aware BLEU using `sacrebleu` with Thai (pythainlp), Chinese (char-level), and English (NLTK) tokenization
- `meteor.py` — Language-aware METEOR using NLTK with same tokenization strategy
- `cdm.py` — CDM formula evaluation via LaTeX token-level F1 comparison fallback
- `structural.py` — Table structural accuracy (row/col match, cell content, structure F1) and formula structural accuracy (exact match, NED, CDM, token F1)

**Parsers** (`src/vlm_ocr_bench/evaluation/`):
- `parsers.py` — Markdown structure extraction into typed blocks (heading, paragraph, table, formula, code, list)

**Benchmarks** (`src/vlm_ocr_bench/evaluation/benchmarks/`):
- `omnidocbench.py` — OmniDocBench evaluation with per-document-type and per-element-type scoring
- `olmocr_bench.py` — olmOCR-Bench evaluation pipeline
- `real5.py` — Real5-OmniDocBench evaluation with per-scenario scoring (scanning, warping, screen_photo, illumination, skew)

**Runner** (`src/vlm_ocr_bench/evaluation/`):
- `runner.py` — `QualityEvalRunner` orchestrator with `QualityEvalResult`, `BenchmarkQuality`, `PrecisionComparison`, `PrecisionDelta` dataclasses

**Tests**:
- `tests/unit/test_evaluation.py` — 88 unit tests covering all metrics, parsers, benchmarks, runner dataclasses, and module exports

### Modified Files
- `CLAUDE.md` — Updated Phase C status to DONE

## Technical Details

- All string comparisons NFKC-normalized before comparison
- Edit distance score: 1.0 = perfect, 0.0 = totally wrong
- Thai BLEU uses pythainlp `newmm` segmentation
- Block-level edit distance uses Hungarian matching for optimal block alignment
- Precision comparison uses BF16 as baseline
- Runner supports all 4 benchmarks: omnidocbench_v1.5, olmocr_bench, real5_omnidocbench, thai_docs_500
- `num_failed` tracked for samples with empty/invalid model output

## Testing

- Unit tests: `tests/unit/test_evaluation.py` (88 tests)
- Coverage: 85% overall (runner GPU-dependent methods excluded)
- All 367 unit tests pass (existing + new)
- Linting: ruff clean, mypy clean
