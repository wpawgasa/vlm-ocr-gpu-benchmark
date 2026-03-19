# Analysis & Reporting Module

**Date**: 2026-03-19
**Branch**: `feature/analysis-reporting`

## Overview

Implements the analysis and reporting module for post-experiment aggregation, statistical testing, performance modeling, cost analysis, visualization, and Markdown report generation.

## Changes

### New Files

**Core Analysis** (`src/vlm_ocr_bench/analysis/`):
- `statistics.py` — Welch's t-test, bootstrap CI (10K iterations), Cohen's d effect size, speedup computation with significance testing
- `aggregator.py` — Result aggregation across runs (group by config, compute mean/std/CI), flatten results to dicts, convert to DataFrame
- `roofline.py` — Roofline model computation (arithmetic intensity, bottleneck detection, utilization %) and annotated plot generation
- `pareto.py` — Pareto frontier computation (quality vs throughput) with optimal point identification and plot generation
- `tco.py` — TCO projection (GPU requirements, monthly/annual cost breakdown, cost per 1K pages, monthly projections)
- `plots.py` — Visualization generators: throughput comparison bars, latency CDF, power efficiency, memory usage, training convergence, quality heatmap
- `report.py` — Markdown report generator with sections: Executive Summary, Inference Results, Training Results, Quality Analysis, Power Efficiency, TCO Analysis, Recommendations, Appendix

**Tests**:
- `tests/unit/test_analysis.py` — 65 unit tests

## Testing

- Unit tests: 65 tests in `tests/unit/test_analysis.py`
- All 436 unit tests pass
- ruff clean, mypy clean
