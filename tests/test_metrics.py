"""
Tests for src/metrics.py — performance metrics.

Verifies:
  - Tracking error formula correctness
  - Normalised tracking error properties
  - Weight violation statistics
  - Wealth comparison metrics
  - Cross-path summary statistics
  - Edge cases (perfect tracking, constant paths)
"""

import numpy as np
import pytest

from src.metrics import (
    tracking_error,
    normalised_tracking_error,
    weight_violations,
    wealth_comparison,
    cross_path_summary,
    compute_granularity_convergence_rate,
    tracking_error_vs_granularity,
)

PI_MINUS = 0.62
PI_PLUS = 0.72


# ---------------------------------------------------------------------------
# tracking_error tests
# ---------------------------------------------------------------------------

def test_tracking_error_zero_for_perfect_match():
    """TE = 0 when actual == target."""
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    assert tracking_error(arr, arr.copy()) == 0.0


def test_tracking_error_known_value():
    """TE = sqrt(mean([0, 1, 0, 1])) = sqrt(0.5) for unit alternating errors."""
    actual = np.array([1.0, 1.0, 1.0, 1.0])
    target = np.array([1.0, 2.0, 1.0, 2.0])
    te = tracking_error(actual, target)
    expected = np.sqrt(0.5)
    assert abs(te - expected) < 1e-10


def test_tracking_error_nonnegative():
    """TE is always non-negative."""
    actual = np.random.default_rng(0).standard_normal(100)
    target = np.random.default_rng(1).standard_normal(100)
    te = tracking_error(actual, target)
    assert te >= 0.0


def test_tracking_error_symmetric():
    """TE(a, b) == TE(b, a)."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 1.0, 2.0])
    assert abs(tracking_error(a, b) - tracking_error(b, a)) < 1e-12


def test_tracking_error_single_element():
    """TE with single element equals absolute difference."""
    te = tracking_error(np.array([3.0]), np.array([5.0]))
    assert abs(te - 2.0) < 1e-12


# ---------------------------------------------------------------------------
# normalised_tracking_error tests
# ---------------------------------------------------------------------------

def test_normalised_te_zero_for_perfect_match():
    arr = np.array([5.0, 10.0, 7.5])
    assert normalised_tracking_error(arr, arr.copy()) == 0.0


def test_normalised_te_dimensionless():
    """NTE should be dimensionless — doubling both arrays shouldn't change it."""
    a = np.array([1.0, 1.5, 2.0])
    b = np.array([1.2, 1.4, 1.9])
    nte1 = normalised_tracking_error(a, b)
    nte2 = normalised_tracking_error(2 * a, 2 * b)
    assert abs(nte1 - nte2) < 1e-12


def test_normalised_te_nan_for_zero_target():
    """NTE is nan when mean absolute target is zero."""
    result = normalised_tracking_error(np.array([0.1, 0.2]), np.array([0.0, 0.0]))
    assert np.isnan(result)


# ---------------------------------------------------------------------------
# weight_violations tests
# ---------------------------------------------------------------------------

def test_no_violations():
    """All weights inside zone → violation_rate = 0."""
    weights = np.linspace(PI_MINUS + 0.01, PI_PLUS - 0.01, 100)
    viol = weight_violations(weights, PI_MINUS, PI_PLUS)
    assert viol["violation_rate"] == 0.0
    assert viol["max_below_pi_minus"] == 0.0
    assert viol["max_above_pi_plus"] == 0.0


def test_all_violations_below():
    """All weights below pi_minus → violation_rate = 1."""
    weights = np.full(50, PI_MINUS - 0.05)
    viol = weight_violations(weights, PI_MINUS, PI_PLUS)
    assert viol["violation_rate"] == 1.0
    assert viol["max_below_pi_minus"] > 0.0
    assert viol["max_above_pi_plus"] == 0.0


def test_all_violations_above():
    """All weights above pi_plus → violation_rate = 1."""
    weights = np.full(50, PI_PLUS + 0.05)
    viol = weight_violations(weights, PI_MINUS, PI_PLUS)
    assert viol["violation_rate"] == 1.0
    assert viol["max_above_pi_plus"] > 0.0


def test_partial_violations():
    """Mixed weights → violation_rate is fraction outside zone."""
    weights = np.array([0.60, 0.65, 0.67, 0.70, 0.75])  # [0.60, 0.75] outside [0.62, 0.72]
    viol = weight_violations(weights, PI_MINUS, PI_PLUS)
    assert 0.0 < viol["violation_rate"] < 1.0


def test_violation_stats_bounded():
    """mean and std of weights should be finite."""
    weights = np.random.default_rng(42).uniform(0.5, 0.8, 200)
    viol = weight_violations(weights, PI_MINUS, PI_PLUS)
    assert np.isfinite(viol["mean_weight"])
    assert np.isfinite(viol["std_weight"])


# ---------------------------------------------------------------------------
# wealth_comparison tests
# ---------------------------------------------------------------------------

def test_wealth_comparison_equal_paths():
    """When LP and Merton have identical wealth, ratio = 1 everywhere."""
    W = np.exp(np.cumsum(np.random.default_rng(0).normal(0.0004, 0.01, 100)))
    wc = wealth_comparison(W, W.copy())
    assert abs(wc["terminal_relative_wealth"] - 1.0) < 1e-10
    assert abs(wc["mean_relative_wealth"] - 1.0) < 1e-10


def test_wealth_comparison_keys():
    """Wealth comparison returns all expected keys."""
    W1 = np.ones(50) * 1000.0
    W2 = np.ones(50) * 900.0
    wc = wealth_comparison(W1, W2)
    for key in ["terminal_relative_wealth", "mean_relative_wealth",
                "log_relative_wealth_sharpe", "lp_terminal_wealth",
                "merton_terminal_wealth"]:
        assert key in wc


def test_wealth_comparison_terminal_ratio():
    """terminal_relative_wealth = lp_wealth[-1] / merton_wealth[-1]."""
    lp = np.array([1000.0, 1010.0, 1025.0])
    merton = np.array([1000.0, 1005.0, 990.0])
    wc = wealth_comparison(lp, merton)
    expected = 1025.0 / 990.0
    assert abs(wc["terminal_relative_wealth"] - expected) < 1e-10


# ---------------------------------------------------------------------------
# cross_path_summary tests
# ---------------------------------------------------------------------------

def test_cross_path_summary_keys():
    """Summary should contain all expected keys."""
    lp = np.random.default_rng(0).lognormal(0, 0.05, 100)
    m = np.random.default_rng(1).lognormal(0, 0.05, 100)
    summary = cross_path_summary(lp * 1000, m * 1000)
    for key in ["mean_terminal_ratio", "std_terminal_ratio", "median_terminal_ratio",
                "t_stat_vs_1", "p_value_vs_1", "fraction_lp_outperforms"]:
        assert key in summary


def test_cross_path_fraction_outperforms_bounded():
    """Fraction outperforming must be in [0, 1]."""
    lp = np.ones(100) * 1100.0
    m = np.ones(100) * 1000.0
    summary = cross_path_summary(lp, m)
    assert summary["fraction_lp_outperforms"] == 1.0


def test_cross_path_p_value_bounded():
    """p-value must be in [0, 1]."""
    rng = np.random.default_rng(99)
    lp = rng.normal(1000, 50, 200)
    m = rng.normal(1000, 50, 200)
    summary = cross_path_summary(np.abs(lp), np.abs(m))
    assert 0.0 <= summary["p_value_vs_1"] <= 1.0


# ---------------------------------------------------------------------------
# granularity convergence tests
# ---------------------------------------------------------------------------

def test_convergence_rate_negative():
    """For a decreasing TE sequence, convergence rate should be negative."""
    n_steps = [50, 100, 200, 400]
    te_vals = [0.10, 0.07, 0.05, 0.035]   # decreasing
    rate = compute_granularity_convergence_rate(n_steps, te_vals)
    assert rate < 0.0


def test_convergence_rate_power_law():
    """Known power-law TE = C * n^{-0.5}: rate should be close to -0.5."""
    n_steps = [10, 40, 160, 640, 2560]
    te_vals = [1.0 / np.sqrt(n) for n in n_steps]
    rate = compute_granularity_convergence_rate(n_steps, te_vals)
    assert abs(rate - (-0.5)) < 0.01


def test_tracking_error_vs_granularity_format():
    """Output dict has correct keys and matching lengths."""
    amm_X = [np.ones(n) * 5.0 for n in [50, 100, 200]]
    target_X = [np.ones(n) * 5.2 for n in [50, 100, 200]]
    n_steps_list = [50, 100, 200]
    result = tracking_error_vs_granularity(amm_X, target_X, n_steps_list)
    assert "n_steps" in result and "tracking_error" in result
    assert len(result["n_steps"]) == 3
    assert len(result["tracking_error"]) == 3
