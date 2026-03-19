"""Statistical testing — Welch's t-test, bootstrap CI, Cohen's d, speedup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
from scipy import stats

_FloatArray: TypeAlias = np.ndarray[Any, np.dtype[np.float64]]


@dataclass
class StatTestResult:
    """Result from a statistical significance test."""

    t_statistic: float
    p_value: float
    significant: bool  # p < 0.05
    cohens_d: float
    ci_95_diff: tuple[float, float]


@dataclass
class SpeedupResult:
    """Result from a speedup comparison between two GPU groups."""

    mean_speedup: float
    ci_95: tuple[float, float]
    p_value: float
    effect_size: float  # Cohen's d


def _cohens_d(group_a: _FloatArray, group_b: _FloatArray) -> float:
    """Compute Cohen's d effect size (pooled standard deviation)."""
    n_a, n_b = len(group_a), len(group_b)
    if n_a + n_b < 3:
        return 0.0
    var_a = float(np.var(group_a, ddof=1)) if n_a > 1 else 0.0
    var_b = float(np.var(group_b, ddof=1)) if n_b > 1 else 0.0
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group_a) - np.mean(group_b)) / pooled_std)


def welch_t_test(
    group_a: list[float],
    group_b: list[float],
) -> StatTestResult:
    """Perform Welch's t-test (unequal variance) between two groups.

    Args:
        group_a: measurements from group A
        group_b: measurements from group B

    Returns:
        StatTestResult with t-statistic, p-value, significance, Cohen's d,
        and 95% CI for the difference in means.
    """
    a = np.array(group_a, dtype=np.float64)
    b = np.array(group_b, dtype=np.float64)

    if len(a) < 2 or len(b) < 2:
        return StatTestResult(
            t_statistic=0.0,
            p_value=1.0,
            significant=False,
            cohens_d=0.0,
            ci_95_diff=(0.0, 0.0),
        )

    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    d = _cohens_d(a, b)

    # 95% CI for difference in means
    diff_mean = float(np.mean(a) - np.mean(b))
    se = np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
    # Welch-Satterthwaite degrees of freedom
    df_num = (np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b)) ** 2
    df_den = (np.var(a, ddof=1) / len(a)) ** 2 / (len(a) - 1) + (
        np.var(b, ddof=1) / len(b)
    ) ** 2 / (len(b) - 1)
    df = df_num / df_den if df_den > 0 else 1.0
    t_crit = float(stats.t.ppf(0.975, df))
    ci_low = diff_mean - t_crit * float(se)
    ci_high = diff_mean + t_crit * float(se)

    return StatTestResult(
        t_statistic=float(t_stat),
        p_value=float(p_val),
        significant=float(p_val) < 0.05,
        cohens_d=d,
        ci_95_diff=(ci_low, ci_high),
    )


def bootstrap_ci(
    data: list[float],
    statistic: Literal["mean", "median"] = "mean",
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval.

    Args:
        data: observed values
        statistic: "mean" or "median"
        n_bootstrap: number of bootstrap iterations
        ci: confidence level (default 0.95)
        seed: random seed for reproducibility

    Returns:
        (lower_bound, upper_bound) of the confidence interval.

    Raises:
        ValueError: if statistic is not "mean" or "median".
    """
    if statistic not in ("mean", "median"):
        raise ValueError(f"statistic must be 'mean' or 'median', got {statistic!r}")

    arr = np.array(data, dtype=np.float64)
    if len(arr) < 2:
        val = float(arr[0]) if len(arr) == 1 else 0.0
        return (val, val)

    rng = np.random.default_rng(seed)

    # Vectorised bootstrap: draw all samples at once for efficiency
    samples = rng.choice(arr, size=(n_bootstrap, len(arr)), replace=True)
    if statistic == "mean":
        boot_stats = np.mean(samples, axis=1)
    else:
        boot_stats = np.median(samples, axis=1)

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_stats, 100 * alpha))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha)))
    return (lower, upper)


def compute_speedup(
    baseline_values: list[float],
    comparison_values: list[float],
    seed: int = 42,
) -> SpeedupResult:
    """Compute speedup of comparison over baseline with statistical significance.

    Speedup = mean(comparison) / mean(baseline).
    A speedup > 1.0 means comparison is faster.

    Args:
        baseline_values: measurements from baseline (e.g., H100)
        comparison_values: measurements from comparison (e.g., B300)
        seed: random seed for bootstrap reproducibility

    Returns:
        SpeedupResult with mean speedup, 95% CI, p-value, effect size.
    """
    base = np.array(baseline_values, dtype=np.float64)
    comp = np.array(comparison_values, dtype=np.float64)

    base_mean = float(np.mean(base)) if len(base) > 0 else 0.0
    comp_mean = float(np.mean(comp)) if len(comp) > 0 else 0.0

    if base_mean == 0:
        return SpeedupResult(mean_speedup=0.0, ci_95=(0.0, 0.0), p_value=1.0, effect_size=0.0)

    mean_speedup = comp_mean / base_mean

    # Bootstrap CI for the speedup ratio (vectorised)
    if len(base) >= 2 and len(comp) >= 2:
        rng = np.random.default_rng(seed)
        n_boot = 10000
        b_samples = rng.choice(base, size=(n_boot, len(base)), replace=True)
        c_samples = rng.choice(comp, size=(n_boot, len(comp)), replace=True)
        b_means = np.mean(b_samples, axis=1)
        c_means = np.mean(c_samples, axis=1)
        boot_speedups = np.where(b_means > 0, c_means / b_means, 0.0)
        ci_low = float(np.percentile(boot_speedups, 2.5))
        ci_high = float(np.percentile(boot_speedups, 97.5))
    else:
        ci_low, ci_high = mean_speedup, mean_speedup

    # Welch's t-test for significance
    test = welch_t_test(baseline_values, comparison_values)

    return SpeedupResult(
        mean_speedup=mean_speedup,
        ci_95=(ci_low, ci_high),
        p_value=test.p_value,
        effect_size=test.cohens_d,
    )
