# 08 — Quality Evaluation (Phase C)

## Overview

Measures OCR quality across benchmarks and precision modes. Computes edit distance, BLEU, METEOR, CDM, and structural metrics. Analyzes precision degradation.

## Checklist

### `evaluation/runner.py`

- [ ] Implement `QualityEvalRunner`:
  - `run() -> QualityEvalResult`
  - Algorithm:
    1. For each benchmark (omnidocbench_v1.5, olmocr_bench, real5_omnidocbench, thai_docs_500):
       - Load dataset
       - For each precision (bf16, fp8, fp4):
         - Initialize inference engine
         - Run inference on all samples
         - Parse outputs via adapter
         - Compute all quality metrics
         - Breakdown by document type (if enabled)
       - Compare across precisions (degradation analysis)
    2. Generate quality comparison tables
- [ ] Define `QualityEvalResult`, `BenchmarkQuality`, `PrecisionComparison`, `PrecisionDelta` dataclasses

### `evaluation/parsers.py`

- [ ] Implement Markdown structure extraction:
  - Split output into blocks (headings, paragraphs, tables, formulas)
  - Identify element types
  - Extract table content into structured format
  - Extract LaTeX formulas

### `evaluation/metrics/edit_distance.py`

- [ ] Implement `normalized_edit_distance(prediction, reference) -> float`
  - Unicode normalize (NFKC) both strings
  - Return `1.0 - (edit_dist / max(len(pred), len(ref)))` — 1.0 = perfect
- [ ] Implement `block_level_edit_distance(pred_blocks, ref_blocks, match_strategy="hungarian") -> float`
  - Hungarian matching to align predicted blocks to reference
  - Average edit distance across matched pairs

### `evaluation/metrics/bleu.py`

- [ ] Implement `compute_bleu(prediction, reference, language, n_gram=4) -> float`
  - Thai: use `pythainlp` word segmentation
  - Chinese: character-level tokenization
  - English/others: `nltk.word_tokenize`

### `evaluation/metrics/meteor.py`

- [ ] Implement `compute_meteor(prediction, reference, language) -> float`
  - Language-aware tokenization (same as BLEU)

### `evaluation/metrics/cdm.py`

- [ ] Implement CDM (formula evaluation) wrapper
  - Wrap external CDM evaluator if available
  - Fallback to LaTeX token-level comparison

### `evaluation/metrics/structural.py`

- [ ] Implement `table_structural_accuracy(pred_table, ref_table, format="markdown") -> dict`
  - Returns: row_count_match, col_count_match, cell_content_accuracy, structure_f1
- [ ] Implement `formula_structural_accuracy(pred_latex, ref_latex) -> dict`
  - Returns: exact_match, normalized_edit_distance, cdm_score, token_level_f1

### `evaluation/benchmarks/omnidocbench.py`

- [ ] Implement OmniDocBench evaluation pipeline
  - Per-document-type scoring
  - Per-element-type scoring (text, table, formula, title, figure)

### `evaluation/benchmarks/olmocr_bench.py`

- [ ] Implement olmOCR-Bench evaluation pipeline

### `evaluation/benchmarks/real5.py`

- [ ] Implement Real5-OmniDocBench evaluation pipeline
  - Per-scenario scoring (scanning, warping, screen_photo, illumination, skew)

## Key Rules

- Always unicode-normalize (NFKC) before comparing strings
- Edit distance score is 1.0 = perfect, 0.0 = totally wrong
- Thai BLEU requires pythainlp segmentation — don't use whitespace tokenization
- Track `num_failed` — samples where model output was empty/invalid
- Precision comparison uses BF16 as baseline
