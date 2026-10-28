"""
Performance Metrics for AMM vs. Merton-Optimal Strategy Comparison.

Implements:
  - Tracking Error between AMM inventory and target schedule
  - Weight boundary violation statistics
  - Relative wealth metrics (LP vs. Merton)
  - Path-level and cross-path summary statistics
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Tracking Error
# ---------------------------------------------------------------------------

def tracking_error(
    actual_inventory: np.ndarray,
    target_inventory: np.ndarray,
) -> float:
    """
    Root-mean-square tracking error between actual and target inventory.

        TE = sqrt(mean((X_t - q*(p_t))^2))

    Parameters
    ----------
    actual_inventory : Actual LP risky-asset holdings at each step.
    target_inventory : Target (Davis-Norman optimal) inventory at each step.

    Returns
    -------
    float
        RMSE tracking error.
    """
    diff = actual_inventory - target_inventory
    return float(np.sqrt(np.mean(diff**2)))


def normalised_tracking_error(
    actual_inventory: np.ndarray,
    target_inventory: np.ndarray,
) -> float:
    """
    Tracking error normalised by the mean absolute target inventory.

        NTE = TE / mean(|q*(p_t)|)

    Parameters
    ----------
    actual_inventory : Actual LP risky-asset holdings.
    target_inventory : Target inventory schedule.

    Returns
    -------
    float
        Normalised tracking error (dimensionless).
    """
    te = tracking_error(actual_inventory, target_inventory)
    mean_abs_target = float(np.mean(np.abs(target_inventory)))
    if mean_abs_target < 1e-10:
        return float("nan")
    return te / mean_abs_target


def tracking_error_vs_granularity(
    amm_X_list: list[np.ndarray],
    target_X_list: list[np.ndarray],
    n_steps_list: list[int],
) -> dict[str, list]:
    """
    Compute tracking error for multiple path resolutions (granularity study).

    Tests the paper's hypothesis that 'tracking error should decrease with
    path granularity'.

    Parameters
    ----------
    amm_X_list     : List of AMM inventory arrays (one per granularity level).
    target_X_list  : Corresponding target inventory arrays.
    n_steps_list   : Number of steps at each granularity level.

    Returns
    -------
    dict with keys 'n_steps' and 'tracking_error'.
    """
    te_vals = [
        tracking_error(ax, tx)
        for ax, tx in zip(amm_X_list, target_X_list)
    ]
    return {"n_steps": n_steps_list, "tracking_error": te_vals}


# ---------------------------------------------------------------------------
# Weight boundary violations
# ---------------------------------------------------------------------------

def weight_violations(
    weights: np.ndarray,
    pi_minus: float,
    pi_plus: float,
) -> dict[str, float]:
    """
    Compute statistics on portfolio-weight boundary violations.

    The paper expects: 'The AMM pool's inventory should stay strictly within
    the [pi_minus, pi_plus] weight range.'

    Parameters
    ----------
    weights  : Array of portfolio weights at each step.
    pi_minus : Lower weight boundary.
    pi_plus  : Upper weight boundary.

    Returns
    -------
    dict with keys:
      'violation_rate'       : Fraction of steps outside [pi_minus, pi_plus].
      'mean_weight'          : Mean portfolio weight.
      'std_weight'           : Std of portfolio weight.
      'max_below_pi_minus'   : Worst violation below lower boundary.
      'max_above_pi_plus'    : Worst violation above upper boundary.
    """
    below = np.maximum(0.0, pi_minus - weights)
    above = np.maximum(0.0, weights - pi_plus)
    violated = (below > 0) | (above > 0)
    return {
        "violation_rate": float(np.mean(violated)),
        "mean_weight": float(np.mean(weights)),
        "std_weight": float(np.std(weights)),
        "max_below_pi_minus": float(np.max(below)),
        "max_above_pi_plus": float(np.max(above)),
    }


# ---------------------------------------------------------------------------
# Wealth comparison
# ---------------------------------------------------------------------------

def wealth_comparison(
    lp_wealth: np.ndarray,
    merton_wealth: np.ndarray,
) -> dict[str, float]:
    """
    Compare LP wealth trajectory against Merton-optimal strategy wealth.

    Computes:
      - Terminal relative wealth: W_LP(T) / W_Merton(T)
      - Mean relative wealth over the path
      - Sharpe-like ratio of relative wealth increments

    Parameters
    ----------
    lp_wealth     : LP total wealth at each step.
    merton_wealth : Merton-agent total wealth at each step.

    Returns
    -------
    dict with comparison metrics.
    """
    relative = lp_wealth / merton_wealth
    terminal_ratio = float(relative[-1])
    mean_ratio = float(np.mean(relative))
    log_rel = np.log(relative[1:] / relative[:-1])
    sharpe = float(np.mean(log_rel) / np.std(log_rel)) if np.std(log_rel) > 0 else 0.0

    return {
        "terminal_relative_wealth": terminal_ratio,
        "mean_relative_wealth": mean_ratio,
        "log_relative_wealth_sharpe": sharpe,
        "lp_terminal_wealth": float(lp_wealth[-1]),
        "merton_terminal_wealth": float(merton_wealth[-1]),
    }


def cross_path_summary(
    all_terminal_lp: np.ndarray,
    all_terminal_merton: np.ndarray,
) -> dict[str, float]:
    """
    Summarise LP vs. Merton wealth across multiple simulated paths.

    Parameters
    ----------
    all_terminal_lp     : Terminal LP wealth for each path. Shape (n_paths,).
    all_terminal_merton : Terminal Merton wealth for each path. Shape (n_paths,).

    Returns
    -------
    dict with aggregate statistics.
    """
    ratios = all_terminal_lp / all_terminal_merton
    t_stat, p_value = stats.ttest_1samp(ratios, 1.0)
    return {
        "mean_terminal_ratio": float(np.mean(ratios)),
        "std_terminal_ratio": float(np.std(ratios)),
        "median_terminal_ratio": float(np.median(ratios)),
        "t_stat_vs_1": float(t_stat),
        "p_value_vs_1": float(p_value),
        "fraction_lp_outperforms": float(np.mean(ratios > 1.0)),
    }


# ---------------------------------------------------------------------------
# Granularity convergence helper
# ---------------------------------------------------------------------------

def compute_granularity_convergence_rate(
    n_steps_list: list[int],
    te_list: list[float],
) -> float:
    """
    Estimate the convergence rate of tracking error with granularity.

    Fits TE ~ C * n_steps^alpha by log-log OLS.

    Parameters
    ----------
    n_steps_list : List of step counts (increasing).
    te_list      : Corresponding tracking errors.

    Returns
    -------
    float
        Estimated exponent alpha (negative implies decreasing TE with steps).
    """
    log_n = np.log(n_steps_list)
    log_te = np.log(te_list)
    slope, intercept, r, p, se = stats.linregress(log_n, log_te)
    return float(slope)
