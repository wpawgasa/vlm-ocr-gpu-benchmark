"""Analysis & reporting: aggregation, statistics, roofline, Pareto, TCO, plots, reports."""

from vlm_ocr_bench.analysis.aggregator import (
    AggregatedConfig,
    AggregatedMetric,
    aggregate_inference_results,
    aggregate_training_results,
    flatten_inference_result,
    results_to_dataframe,
)
from vlm_ocr_bench.analysis.pareto import (
    ParetoPoint,
    ParetoResult,
    compute_pareto_frontier,
)
from vlm_ocr_bench.analysis.report import (
    ReportData,
    ReportGenerator,
    ReportSection,
)
from vlm_ocr_bench.analysis.roofline import (
    RooflinePoint,
    compute_roofline,
)
from vlm_ocr_bench.analysis.statistics import (
    SpeedupResult,
    StatTestResult,
    bootstrap_ci,
    compute_speedup,
    welch_t_test,
)
from vlm_ocr_bench.analysis.tco import (
    TCOParams,
    TCOResult,
    compute_tco,
)

__all__ = [
    "AggregatedConfig",
    "AggregatedMetric",
    "ParetoPoint",
    "ParetoResult",
    "ReportData",
    "ReportGenerator",
    "ReportSection",
    "RooflinePoint",
    "SpeedupResult",
    "StatTestResult",
    "TCOParams",
    "TCOResult",
    "aggregate_inference_results",
    "aggregate_training_results",
    "bootstrap_ci",
    "compute_pareto_frontier",
    "compute_roofline",
    "compute_speedup",
    "compute_tco",
    "flatten_inference_result",
    "results_to_dataframe",
    "welch_t_test",
]
