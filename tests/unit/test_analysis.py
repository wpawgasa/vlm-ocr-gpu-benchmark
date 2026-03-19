"""Unit tests for the analysis & reporting module (no GPU)."""

from __future__ import annotations

import tempfile
from pathlib import Path

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
from vlm_ocr_bench.config.schema import GPUType, PrecisionMode

# ─── Statistics Tests ───


class TestWelchTTest:
    def test_identical_groups(self) -> None:
        result = welch_t_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert isinstance(result, StatTestResult)
        assert result.p_value > 0.05
        assert result.significant is False
        assert result.cohens_d == 0.0

    def test_different_groups(self) -> None:
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [10.0, 11.0, 12.0, 13.0, 14.0]
        result = welch_t_test(a, b)
        assert result.p_value < 0.05
        assert result.significant is True
        assert abs(result.cohens_d) > 0.8  # large effect

    def test_insufficient_data(self) -> None:
        result = welch_t_test([1.0], [2.0])
        assert result.p_value == 1.0
        assert result.significant is False

    def test_ci_contains_true_diff(self) -> None:
        a = [10.0, 11.0, 12.0, 13.0, 14.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = welch_t_test(a, b)
        # True diff is 9.0, CI should contain it
        assert result.ci_95_diff[0] <= 9.0 <= result.ci_95_diff[1]

    def test_cohens_d_sign(self) -> None:
        a = [10.0, 11.0, 12.0]
        b = [1.0, 2.0, 3.0]
        result = welch_t_test(a, b)
        assert result.cohens_d > 0  # a > b

    def test_cohens_d_thresholds(self) -> None:
        # Small effect: d ≈ 0.2
        a = [10.0, 10.1, 10.2, 10.0, 10.1]
        b = [9.8, 9.9, 10.0, 9.8, 9.9]
        result = welch_t_test(a, b)
        assert result.cohens_d >= 0  # positive direction


class TestBootstrapCI:
    def test_basic(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        low, high = bootstrap_ci(data)
        assert low <= 3.0 <= high  # mean is 3.0

    def test_single_value(self) -> None:
        low, high = bootstrap_ci([5.0])
        assert low == 5.0
        assert high == 5.0

    def test_empty(self) -> None:
        low, high = bootstrap_ci([])
        assert low == 0.0
        assert high == 0.0

    def test_tight_ci_for_identical_values(self) -> None:
        data = [10.0] * 20
        low, high = bootstrap_ci(data)
        assert abs(low - 10.0) < 0.01
        assert abs(high - 10.0) < 0.01

    def test_median_statistic(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0, 100.0]  # outlier
        low, high = bootstrap_ci(data, statistic="median")
        assert low <= 3.0 <= high  # median is 3.0

    def test_reproducible_with_seed(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        r1 = bootstrap_ci(data, seed=42)
        r2 = bootstrap_ci(data, seed=42)
        assert r1 == r2

    def test_custom_ci(self) -> None:
        data = list(range(100))
        low_95, high_95 = bootstrap_ci([float(x) for x in data], ci=0.95)
        low_99, high_99 = bootstrap_ci([float(x) for x in data], ci=0.99)
        # 99% CI should be wider than 95% CI
        assert (high_99 - low_99) >= (high_95 - low_95)

    def test_invalid_statistic_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="statistic must be"):
            bootstrap_ci([1.0, 2.0, 3.0], statistic="mode")  # type: ignore[arg-type]


class TestComputeSpeedup:
    def test_double_speedup(self) -> None:
        baseline = [10.0, 10.0, 10.0]
        comparison = [20.0, 20.0, 20.0]
        result = compute_speedup(baseline, comparison)
        assert isinstance(result, SpeedupResult)
        assert abs(result.mean_speedup - 2.0) < 0.01

    def test_no_speedup(self) -> None:
        values = [10.0, 10.0, 10.0]
        result = compute_speedup(values, values)
        assert abs(result.mean_speedup - 1.0) < 0.01

    def test_zero_baseline(self) -> None:
        result = compute_speedup([0.0, 0.0], [10.0, 10.0])
        assert result.mean_speedup == 0.0

    def test_ci_covers_speedup(self) -> None:
        baseline = [10.0, 11.0, 12.0, 10.0, 11.0]
        comparison = [20.0, 21.0, 22.0, 20.0, 21.0]
        result = compute_speedup(baseline, comparison)
        assert result.ci_95[0] <= result.mean_speedup <= result.ci_95[1]

    def test_seed_reproducibility(self) -> None:
        baseline = [10.0, 11.0, 12.0, 10.5, 11.5]
        comparison = [20.0, 21.0, 22.0, 20.5, 21.5]
        r1 = compute_speedup(baseline, comparison, seed=7)
        r2 = compute_speedup(baseline, comparison, seed=7)
        assert r1.ci_95 == r2.ci_95


# ─── Aggregator Tests ───


def _make_inference_row(
    model: str = "test_model",
    gpu: str = "h100_sxm",
    precision: str = "bf16",
    batch_size: int = 1,
    resolution: int = 1024,
    pps: float = 10.0,
    latency_p50: float = 100.0,
    run_id: int = 0,
) -> dict[str, object]:
    return {
        "model_name": model,
        "gpu_type": gpu,
        "precision": precision,
        "batch_size": batch_size,
        "resolution": resolution,
        "max_output_tokens": 4096,
        "run_id": run_id,
        "pages_per_second": pps,
        "tokens_per_second": pps * 100,
        "e2e_latency_p50": latency_p50,
        "e2e_latency_p95": latency_p50 * 1.5,
        "e2e_latency_p99": latency_p50 * 2.0,
        "ttft_p50": latency_p50 * 0.1,
        "ttft_p95": latency_p50 * 0.15,
        "ttft_p99": latency_p50 * 0.2,
        "mean_power_watts": 500.0,
        "peak_power_watts": 600.0,
        "energy_per_page_wh": 0.05,
        "pages_per_watt_hour": 20.0,
        "tokens_per_watt_hour": 2000.0,
        "peak_gpu_memory_gb": 20.0,
    }


class TestAggregateInferenceResults:
    def test_basic_aggregation(self) -> None:
        rows = [
            _make_inference_row(pps=10.0, run_id=0),
            _make_inference_row(pps=12.0, run_id=1),
            _make_inference_row(pps=11.0, run_id=2),
        ]
        result = aggregate_inference_results(rows)
        assert len(result) == 1
        assert result[0].n_runs == 3
        pps = result[0].metrics["pages_per_second"]
        assert abs(pps.mean - 11.0) < 0.01
        assert pps.n_runs == 3

    def test_multiple_configs(self) -> None:
        rows = [
            _make_inference_row(batch_size=1, pps=10.0),
            _make_inference_row(batch_size=2, pps=20.0),
        ]
        result = aggregate_inference_results(rows)
        assert len(result) == 2

    def test_filters_oom(self) -> None:
        rows = [
            _make_inference_row(pps=10.0),
            {**_make_inference_row(pps=0.0), "oom": True},
        ]
        result = aggregate_inference_results(rows)
        assert len(result) == 1
        assert result[0].metrics["pages_per_second"].mean == 10.0

    def test_filters_errors(self) -> None:
        rows = [
            _make_inference_row(pps=10.0),
            {**_make_inference_row(pps=0.0), "error": "CUDA error"},
        ]
        result = aggregate_inference_results(rows)
        assert len(result) == 1

    def test_empty_input(self) -> None:
        assert aggregate_inference_results([]) == []

    def test_ci_computed(self) -> None:
        rows = [_make_inference_row(pps=float(i), run_id=i) for i in range(10, 20)]
        result = aggregate_inference_results(rows)
        pps = result[0].metrics["pages_per_second"]
        assert pps.ci_95_low <= pps.mean <= pps.ci_95_high


def _make_training_row(
    model: str = "test",
    gpu: str = "h100_sxm",
    precision: str = "bf16",
    micro_batch_size: int = 4,
    sps: float = 10.0,
) -> dict[str, object]:
    return {
        "model_name": model,
        "gpu_type": gpu,
        "precision": precision,
        "micro_batch_size": micro_batch_size,
        "samples_per_second": sps,
        "tokens_per_second": sps * 50,
        "peak_gpu_memory_gb": 20.0,
        "mean_power_watts": 400.0,
        "energy_per_1k_samples_wh": 13.9,
    }


class TestAggregateTrainingResults:
    def test_basic(self) -> None:
        rows = [_make_training_row()]
        result = aggregate_training_results(rows)
        assert len(result) == 1
        assert result[0].metrics["samples_per_second"].mean == 10.0

    def test_empty(self) -> None:
        assert aggregate_training_results([]) == []

    def test_filters_oom(self) -> None:
        rows = [
            _make_training_row(sps=10.0),
            {**_make_training_row(sps=0.0), "oom": True},
        ]
        result = aggregate_training_results(rows)
        assert len(result) == 1
        assert result[0].metrics["samples_per_second"].mean == 10.0

    def test_filters_errors(self) -> None:
        rows = [
            _make_training_row(sps=10.0),
            {**_make_training_row(sps=0.0), "error": "CUDA error"},
        ]
        result = aggregate_training_results(rows)
        assert len(result) == 1


class TestFlattenInferenceResult:
    def test_basic(self) -> None:
        from unittest.mock import MagicMock

        from vlm_ocr_bench.inference.metrics import InferenceMetrics

        cfg = MagicMock()
        cfg.precision = "bf16"
        cfg.batch_size = 1
        cfg.resolution = 1024
        cfg.max_output_tokens = 4096
        cfg.run_id = 0
        cfg.num_requests = 100
        cfg.wall_time_s = 10.0
        cfg.oom = False
        cfg.error = None
        cfg.metrics = InferenceMetrics(pages_per_second=10.0, tokens_per_second=1000.0)

        rows = flatten_inference_result("test_model", "h100_sxm", [cfg])
        assert len(rows) == 1
        assert rows[0]["model_name"] == "test_model"
        assert rows[0]["pages_per_second"] == 10.0


class TestResultsToDataframe:
    def test_basic(self) -> None:
        config = AggregatedConfig(
            model_name="test",
            gpu_type="h100_sxm",
            precision="bf16",
            batch_size=1,
            resolution=1024,
            n_runs=3,
            metrics={
                "pages_per_second": AggregatedMetric(
                    name="pages_per_second", mean=10.0, std=1.0, n_runs=3
                ),
            },
        )
        df = results_to_dataframe([config])
        assert len(df) == 1
        assert "pages_per_second_mean" in df.columns
        assert df.iloc[0]["pages_per_second_mean"] == 10.0

    def test_empty(self) -> None:
        df = results_to_dataframe([])
        assert len(df) == 0


class TestAggregatedMetric:
    def test_defaults(self) -> None:
        m = AggregatedMetric(name="test")
        assert m.mean == 0.0
        assert m.std == 0.0
        assert m.n_runs == 0
        assert m.values == []


class TestAggregatedConfig:
    def test_defaults(self) -> None:
        c = AggregatedConfig(
            model_name="test",
            gpu_type="h100_sxm",
            precision="bf16",
            batch_size=1,
            resolution=1024,
        )
        assert c.n_runs == 0
        assert c.metrics == {}


# ─── Roofline Tests ───


class TestComputeRoofline:
    def test_basic(self) -> None:
        point = compute_roofline(
            gpu_type=GPUType.H100_SXM,
            precision=PrecisionMode.BF16,
            measured_tflops=500.0,
            measured_bandwidth_tb_s=2.0,
            label="test_model",
        )
        assert isinstance(point, RooflinePoint)
        assert point.label == "test_model"
        assert point.measured_tflops == 500.0
        assert point.peak_tflops > 0
        assert point.utilization_pct > 0
        assert point.bottleneck in ("compute", "memory", "balanced")

    def test_memory_bound(self) -> None:
        # Low arithmetic intensity → memory bound
        point = compute_roofline(
            gpu_type=GPUType.H100_SXM,
            precision=PrecisionMode.BF16,
            measured_tflops=100.0,
            measured_bandwidth_tb_s=3.0,  # high BW usage
        )
        assert point.bottleneck == "memory"

    def test_compute_bound(self) -> None:
        # High arithmetic intensity → compute bound
        point = compute_roofline(
            gpu_type=GPUType.H100_SXM,
            precision=PrecisionMode.BF16,
            measured_tflops=800.0,
            measured_bandwidth_tb_s=0.1,  # low BW
        )
        assert point.bottleneck == "compute"

    def test_fp8_precision(self) -> None:
        point = compute_roofline(
            gpu_type=GPUType.H100_SXM,
            precision=PrecisionMode.FP8,
            measured_tflops=1000.0,
            measured_bandwidth_tb_s=2.0,
        )
        assert point.peak_tflops > 1900  # FP8 peak is ~1978.9

    def test_b300_gpu(self) -> None:
        point = compute_roofline(
            gpu_type=GPUType.B300_SXM,
            precision=PrecisionMode.BF16,
            measured_tflops=1000.0,
            measured_bandwidth_tb_s=4.0,
        )
        assert point.peak_tflops > 2000.0  # B300 BF16 peak should exceed H100's ~1979 TFLOPS


class TestRooflinePoint:
    def test_defaults(self) -> None:
        p = RooflinePoint(
            label="test",
            arithmetic_intensity=1.0,
            measured_tflops=100.0,
            peak_tflops=1000.0,
            memory_bandwidth_tb_s=3.0,
            utilization_pct=10.0,
            bottleneck="memory",
        )
        assert p.utilization_pct == 10.0


# ─── Pareto Tests ───


class TestComputeParetoFrontier:
    def test_basic(self) -> None:
        points = [
            ParetoPoint(label="A", throughput=10.0, quality=0.9),
            ParetoPoint(label="B", throughput=20.0, quality=0.8),
            ParetoPoint(label="C", throughput=5.0, quality=0.7),
        ]
        result = compute_pareto_frontier(points)
        assert isinstance(result, ParetoResult)
        assert len(result.frontier) == 2  # A and B dominate C
        frontier_labels = {p.label for p in result.frontier}
        assert "A" in frontier_labels
        assert "B" in frontier_labels

    def test_single_point(self) -> None:
        points = [ParetoPoint(label="A", throughput=10.0, quality=0.9)]
        result = compute_pareto_frontier(points)
        assert len(result.frontier) == 1
        assert result.frontier[0].is_pareto_optimal

    def test_all_pareto_optimal(self) -> None:
        # Each point dominates on one axis
        points = [
            ParetoPoint(label="A", throughput=10.0, quality=0.5),
            ParetoPoint(label="B", throughput=5.0, quality=0.9),
        ]
        result = compute_pareto_frontier(points)
        assert len(result.frontier) == 2

    def test_dominated_point(self) -> None:
        points = [
            ParetoPoint(label="A", throughput=10.0, quality=0.9),
            ParetoPoint(label="B", throughput=5.0, quality=0.5),  # dominated by A
        ]
        result = compute_pareto_frontier(points)
        assert len(result.frontier) == 1
        assert result.frontier[0].label == "A"

    def test_empty(self) -> None:
        result = compute_pareto_frontier([])
        assert result.frontier == []
        assert result.points == []

    def test_frontier_sorted_by_throughput(self) -> None:
        points = [
            ParetoPoint(label="A", throughput=20.0, quality=0.8),
            ParetoPoint(label="B", throughput=5.0, quality=0.95),
            ParetoPoint(label="C", throughput=10.0, quality=0.9),
        ]
        result = compute_pareto_frontier(points)
        throughputs = [p.throughput for p in result.frontier]
        assert throughputs == sorted(throughputs)


class TestParetoPoint:
    def test_defaults(self) -> None:
        p = ParetoPoint(label="test", throughput=10.0, quality=0.9)
        assert p.is_pareto_optimal is False
        assert p.model_name == ""


# ─── TCO Tests ───


class TestComputeTco:
    def test_basic(self) -> None:
        result = compute_tco(10.0, 500.0)
        assert isinstance(result, TCOResult)
        assert result.gpus_required >= 1
        assert result.monthly_total > 0
        assert result.projected_total > 0
        assert result.cost_per_1k_pages > 0
        assert len(result.monthly_breakdown) == 12

    def test_projected_total_uses_projection_months(self) -> None:
        params_6 = TCOParams(projection_months=6)
        params_12 = TCOParams(projection_months=12)
        r6 = compute_tco(10.0, 500.0, params_6)
        r12 = compute_tco(10.0, 500.0, params_12)
        assert abs(r12.projected_total - r6.projected_total * 2) < 0.01

    def test_zero_throughput(self) -> None:
        result = compute_tco(0.0, 500.0)
        assert result.gpus_required == 1
        assert result.monthly_total == 0.0

    def test_custom_params(self) -> None:
        params = TCOParams(
            gpu_rental_rate_per_hour=3.00,
            pages_per_day=50_000,
            projection_months=6,
        )
        result = compute_tco(10.0, 500.0, params)
        assert len(result.monthly_breakdown) == 6

    def test_high_throughput_needs_fewer_gpus(self) -> None:
        params = TCOParams(pages_per_day=100_000)
        slow = compute_tco(1.0, 500.0, params)
        fast = compute_tco(100.0, 500.0, params)
        assert fast.gpus_required <= slow.gpus_required

    def test_cumulative_cost_increases(self) -> None:
        result = compute_tco(10.0, 500.0)
        costs = [m["cumulative_cost"] for m in result.monthly_breakdown]
        assert costs == sorted(costs)

    def test_power_cost_proportional(self) -> None:
        low_power = compute_tco(10.0, 200.0)
        high_power = compute_tco(10.0, 800.0)
        assert high_power.monthly_power_cost > low_power.monthly_power_cost


class TestTCOParams:
    def test_defaults(self) -> None:
        p = TCOParams()
        assert p.gpu_rental_rate_per_hour == 2.00
        assert p.power_cost_per_kwh == 0.10
        assert p.pages_per_day == 100_000
        assert p.projection_months == 12


class TestTCOResult:
    def test_defaults(self) -> None:
        r = TCOResult()
        assert r.gpus_required == 1
        assert r.monthly_total == 0.0
        assert r.monthly_breakdown == []


# ─── Report Tests ───


class TestReportSection:
    def test_creation(self) -> None:
        s = ReportSection(title="Test Section", content="Some content.")
        assert s.title == "Test Section"
        assert s.level == 2


class TestReportData:
    def test_defaults(self) -> None:
        d = ReportData()
        assert d.experiment_name == ""
        assert d.inference_results == []
        assert d.quality_results == []


class TestReportGenerator:
    def test_generate_empty_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data = ReportData(
                experiment_name="test_experiment",
                gpu_types=["h100_sxm"],
                model_names=["test_model"],
            )
            gen = ReportGenerator(data, output_dir=tmpdir)
            path = gen.generate()
            assert path.exists()
            content = path.read_text()
            assert "test_experiment" in content
            assert "Table of Contents" in content

    def test_generate_with_inference_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AggregatedConfig(
                model_name="dots_ocr",
                gpu_type="h100_sxm",
                precision="bf16",
                batch_size=1,
                resolution=1024,
                n_runs=3,
                metrics={
                    "pages_per_second": AggregatedMetric(
                        name="pages_per_second", mean=15.5, std=1.0, n_runs=3
                    ),
                    "tokens_per_second": AggregatedMetric(
                        name="tokens_per_second", mean=1500.0, std=100.0, n_runs=3
                    ),
                    "e2e_latency_p50": AggregatedMetric(
                        name="e2e_latency_p50", mean=65.0, std=5.0, n_runs=3
                    ),
                    "pages_per_watt_hour": AggregatedMetric(
                        name="pages_per_watt_hour", mean=20.0, std=2.0, n_runs=3
                    ),
                    "energy_per_page_wh": AggregatedMetric(
                        name="energy_per_page_wh", mean=0.05, std=0.005, n_runs=3
                    ),
                    "mean_power_watts": AggregatedMetric(
                        name="mean_power_watts", mean=500.0, std=20.0, n_runs=3
                    ),
                    "peak_gpu_memory_gb": AggregatedMetric(
                        name="peak_gpu_memory_gb", mean=20.0, std=1.0, n_runs=3
                    ),
                },
            )
            data = ReportData(
                experiment_name="test",
                gpu_types=["h100_sxm"],
                model_names=["dots_ocr"],
                inference_results=[config],
            )
            gen = ReportGenerator(data, output_dir=tmpdir)
            path = gen.generate()
            content = path.read_text()
            assert "Inference Results" in content
            assert "dots_ocr" in content
            assert "15.5" in content

    def test_generate_with_quality_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data = ReportData(
                experiment_name="test",
                quality_results=[
                    {
                        "model_name": "test_model",
                        "gpu_type": "h100_sxm",
                        "precision": "bf16",
                        "benchmark": "omnidocbench",
                        "edit_distance": 0.85,
                        "bleu": 0.72,
                        "meteor": 0.78,
                    },
                ],
            )
            gen = ReportGenerator(data, output_dir=tmpdir)
            path = gen.generate()
            content = path.read_text()
            assert "Quality Analysis" in content
            assert "0.850" in content

    def test_generate_with_tco(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data = ReportData(
                experiment_name="test",
                tco_results={
                    "H100 BF16": {
                        "gpus_required": 2,
                        "monthly_total": 3000.0,
                        "projected_total": 36000.0,
                        "cost_per_1k_pages": 0.05,
                    },
                },
            )
            gen = ReportGenerator(data, output_dir=tmpdir)
            path = gen.generate()
            content = path.read_text()
            assert "TCO Analysis" in content
            assert "H100 BF16" in content


# ─── Plot Tests (just verify no errors, don't check visuals) ───


class TestPlots:
    def test_throughput_comparison(self) -> None:
        from vlm_ocr_bench.analysis.plots import plot_throughput_comparison

        with tempfile.TemporaryDirectory() as tmpdir:
            path = plot_throughput_comparison(
                models=["model_a", "model_b"],
                h100_values=[10.0, 15.0],
                b300_values=[20.0, 30.0],
                output_path=Path(tmpdir) / "throughput.png",
            )
            assert path.exists()

    def test_latency_cdf(self) -> None:
        from vlm_ocr_bench.analysis.plots import plot_latency_cdf

        with tempfile.TemporaryDirectory() as tmpdir:
            path = plot_latency_cdf(
                latencies={
                    "H100 BF16": [10.0, 20.0, 30.0, 40.0, 50.0],
                    "B300 BF16": [5.0, 10.0, 15.0, 20.0, 25.0],
                },
                output_path=Path(tmpdir) / "latency_cdf.png",
            )
            assert path.exists()

    def test_power_efficiency(self) -> None:
        from vlm_ocr_bench.analysis.plots import plot_power_efficiency

        with tempfile.TemporaryDirectory() as tmpdir:
            path = plot_power_efficiency(
                models=["model_a", "model_b"],
                pages_per_wh={"H100": [10.0, 15.0], "B300": [20.0, 30.0]},
                output_path=Path(tmpdir) / "power.png",
            )
            assert path.exists()

    def test_memory_usage(self) -> None:
        from vlm_ocr_bench.analysis.plots import plot_memory_usage

        with tempfile.TemporaryDirectory() as tmpdir:
            path = plot_memory_usage(
                models=["model_a", "model_b"],
                memory_data={"H100": [20.0, 25.0], "B300": [15.0, 20.0]},
                output_path=Path(tmpdir) / "memory.png",
            )
            assert path.exists()

    def test_training_convergence(self) -> None:
        from vlm_ocr_bench.analysis.plots import plot_training_convergence

        with tempfile.TemporaryDirectory() as tmpdir:
            path = plot_training_convergence(
                loss_curves={
                    "Config A": [(0, 2.0), (100, 1.5), (200, 1.0)],
                    "Config B": [(0, 2.5), (100, 1.8), (200, 1.2)],
                },
                output_path=Path(tmpdir) / "convergence.png",
            )
            assert path.exists()

    def test_quality_heatmap(self) -> None:
        from vlm_ocr_bench.analysis.plots import plot_quality_heatmap

        with tempfile.TemporaryDirectory() as tmpdir:
            path = plot_quality_heatmap(
                models=["model_a", "model_b"],
                doc_types=["academic", "financial"],
                scores=[[0.9, 0.8], [0.85, 0.75]],
                output_path=Path(tmpdir) / "heatmap.png",
            )
            assert path.exists()

    def test_empty_latency_cdf(self) -> None:
        from vlm_ocr_bench.analysis.plots import plot_latency_cdf

        with tempfile.TemporaryDirectory() as tmpdir:
            path = plot_latency_cdf(
                latencies={"empty": []},
                output_path=Path(tmpdir) / "empty_cdf.png",
            )
            assert path.exists()

    def test_throughput_comparison_custom_labels(self) -> None:
        from vlm_ocr_bench.analysis.plots import plot_throughput_comparison

        with tempfile.TemporaryDirectory() as tmpdir:
            path = plot_throughput_comparison(
                models=["model_a"],
                h100_values=[10.0],
                b300_values=[20.0],
                output_path=Path(tmpdir) / "throughput_custom.png",
                gpu_a_label="GPU A",
                gpu_b_label="GPU B",
            )
            assert path.exists()


class TestPlotRoofline:
    def test_basic(self) -> None:
        from vlm_ocr_bench.analysis.roofline import plot_roofline

        points = [
            RooflinePoint(
                label="model_a",
                arithmetic_intensity=10.0,
                measured_tflops=500.0,
                peak_tflops=1979.0,
                memory_bandwidth_tb_s=3.35,
                utilization_pct=25.3,
                bottleneck="compute",
            )
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = plot_roofline(
                gpu_type=GPUType.H100_SXM,
                precision=PrecisionMode.BF16,
                points=points,
                output_path=Path(tmpdir) / "roofline.png",
            )
            assert path.exists()

    def test_empty_points(self) -> None:
        from vlm_ocr_bench.analysis.roofline import plot_roofline

        with tempfile.TemporaryDirectory() as tmpdir:
            path = plot_roofline(
                gpu_type=GPUType.H100_SXM,
                precision=PrecisionMode.BF16,
                points=[],
                output_path=Path(tmpdir) / "roofline_empty.png",
            )
            assert path.exists()


class TestPlotPareto:
    def test_basic(self) -> None:
        from vlm_ocr_bench.analysis.pareto import plot_pareto

        points = [
            ParetoPoint(label="A", throughput=10.0, quality=0.9),
            ParetoPoint(label="B", throughput=20.0, quality=0.8),
            ParetoPoint(label="C", throughput=5.0, quality=0.7),
        ]
        result = compute_pareto_frontier(points)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = plot_pareto(result, output_path=Path(tmpdir) / "pareto.png")
            assert path.exists()

    def test_empty(self) -> None:
        from vlm_ocr_bench.analysis.pareto import plot_pareto

        result = compute_pareto_frontier([])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = plot_pareto(result, output_path=Path(tmpdir) / "pareto_empty.png")
            assert path.exists()


# ─── Module Export Tests ───


class TestModuleExports:
    def test_analysis_module_imports(self) -> None:
        from vlm_ocr_bench.analysis import (
            ReportGenerator,
            aggregate_inference_results,
            bootstrap_ci,
            compute_pareto_frontier,
            compute_roofline,
            compute_speedup,
            compute_tco,
            flatten_inference_result,
            plot_pareto,
            plot_roofline,
            results_to_dataframe,
            welch_t_test,
        )

        assert welch_t_test is not None
        assert bootstrap_ci is not None
        assert compute_speedup is not None
        assert aggregate_inference_results is not None
        assert compute_roofline is not None
        assert compute_pareto_frontier is not None
        assert compute_tco is not None
        assert ReportGenerator is not None
        assert results_to_dataframe is not None
        assert flatten_inference_result is not None
        assert plot_roofline is not None
        assert plot_pareto is not None
