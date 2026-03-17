# 10 — Analysis & Reporting

## Overview

Post-experiment aggregation, statistical testing (Welch's t-test, bootstrap CI, Cohen's d), roofline model, Pareto frontier, TCO projection, and Markdown report generation.

## Checklist

### `analysis/aggregator.py`

- [ ] Implement result aggregation across runs:
  - Load raw results from `results/raw/{phase}/`
  - Group by (model, gpu, precision, batch_size, resolution)
  - Compute mean, std, CI for all numeric metrics
  - Output to `results/aggregated/`

### `analysis/statistics.py`

- [ ] Implement `welch_t_test(group_a, group_b) -> StatTestResult`
  - Returns: t_statistic, p_value, significant (p<0.05), cohens_d, ci_95_diff
- [ ] Implement `bootstrap_ci(data, statistic=np.mean, n_bootstrap=10000, ci=0.95) -> (float, float)`
- [ ] Implement `compute_speedup(h100_values, b300_values) -> SpeedupResult`
  - Returns: mean_speedup, ci_95, p_value, effect_size

### `analysis/roofline.py`

- [ ] Implement `compute_roofline(gpu_info, model_config, precision, measured_tflops, measured_bandwidth) -> RooflinePoint`
  - Compute arithmetic intensity (FLOP/byte)
  - Determine bottleneck: compute vs memory vs balanced
  - Compute utilization percentage vs theoretical peak
- [ ] Implement `plot_roofline(gpu_info, points, output_path)` — annotated roofline plot

### `analysis/pareto.py`

- [ ] Implement Pareto frontier computation:
  - Quality (e.g., edit distance) vs throughput (pages/s)
  - Identify Pareto-optimal model × config combinations
  - Generate Pareto frontier plot

### `analysis/tco.py`

- [ ] Define `TCOParams` dataclass (gpu_rental_rate, power_cost_per_kwh, pages_per_day, operating_hours, projection_months)
- [ ] Implement `compute_tco(inference_result, tco_params) -> TCOResult`
  - GPUs required to meet throughput target
  - Monthly/annual cost breakdown (GPU rental + power)
  - Cost per 1000 pages
- [ ] Define `TCOResult` dataclass

### `analysis/plots.py`

- [ ] Implement visualization generators (matplotlib + plotly):
  - Throughput comparison bar charts (H100 vs B300)
  - Latency CDF plots
  - Power efficiency scatter plots
  - Memory usage stacked bars
  - Training convergence curves
  - Quality metric heatmaps (model × doc_type)

### `analysis/report.py`

- [ ] Implement `ReportGenerator.generate()` producing Markdown with sections:
  1. Executive Summary (headline speedups, recommendations)
  2. Inference Results (tables, throughput plots, latency CDFs)
  3. Training Results (throughput, convergence, cost)
  4. Quality Analysis (precision degradation, per-doc-type)
  5. Power Efficiency (pages/Wh comparison)
  6. TCO Analysis (break-even, projections)
  7. Roofline Analysis (bottleneck identification)
  8. Pareto Frontier (quality-throughput tradeoff)
  9. Recommendations (model × GPU pairing guide)
  10. Appendix (full config snapshots, raw data links)

## Key Rules

- Always use Welch's t-test (not Student's) — variances are likely unequal
- Bootstrap CI with 10,000 iterations for robust confidence intervals
- Cohen's d thresholds: small=0.2, medium=0.5, large=0.8
- Roofline peaks must be precision-specific (FP8 peak ≠ BF16 peak)
- TCO uses cloud rental rates — make these configurable
