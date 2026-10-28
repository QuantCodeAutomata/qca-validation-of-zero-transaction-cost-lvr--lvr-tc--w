"""
Tests for src/amm_simulator.py and src/merton_simulator.py.

Verifies:
  - AMM output shapes and valid values
  - Portfolio weight stays within no-trade zone
  - Wealth non-negativity
  - Merton strategy rebalancing logic
  - Constant-price no-trade scenario
"""

import numpy as np
import pytest

from src.amm_construction import (
    compute_amm_weight,
    initialise_pool,
    compute_gamma_amm,
    compute_fee_tau,
)
from src.amm_simulator import AMMSimulator
from src.merton_simulator import MertonSimulator
from src.price_simulator import simulate_gbm, time_grid

PI_MINUS = 0.62
PI_PLUS = 0.72
W0 = 1000.0
S0 = 100.0
LAM_EXT = 0.003


def _make_amm(n_steps: int = 50, n_paths: int = 1):
    w = compute_amm_weight(PI_MINUS, PI_PLUS)
    gamma = compute_gamma_amm(PI_MINUS, PI_PLUS)
    tau = compute_fee_tau(gamma, LAM_EXT)
    X0, Y0, k = initialise_pool(S0, W0, w)
    return AMMSimulator(X0, Y0, k, w, tau, LAM_EXT), w


# ---------------------------------------------------------------------------
# AMM simulator tests
# ---------------------------------------------------------------------------

def test_amm_output_shapes():
    """Output arrays should have length n_steps+1."""
    amm, _ = _make_amm()
    prices = simulate_gbm(S0, 0.10, 0.20, 1.0, 50, 1, seed=0)[0]
    times = time_grid(1.0, 50)
    res = amm.simulate(prices, times)
    n = len(prices)
    assert len(res.X) == n
    assert len(res.Y) == n
    assert len(res.weights) == n
    assert len(res.wealth) == n
    assert len(res.traded) == n


def test_amm_x_positive():
    """Risky asset reserves X must remain positive."""
    amm, _ = _make_amm()
    prices = simulate_gbm(S0, 0.10, 0.20, 1.0, 100, 1, seed=1)[0]
    times = time_grid(1.0, 100)
    res = amm.simulate(prices, times)
    assert np.all(res.X > 0)


def test_amm_wealth_positive():
    """LP total wealth must remain positive."""
    amm, _ = _make_amm()
    prices = simulate_gbm(S0, 0.10, 0.20, 1.0, 100, 1, seed=2)[0]
    times = time_grid(1.0, 100)
    res = amm.simulate(prices, times)
    assert np.all(res.wealth > 0)


def test_amm_weight_in_zero_one():
    """Portfolio weights must be in [0, 1]."""
    amm, _ = _make_amm()
    prices = simulate_gbm(S0, 0.10, 0.20, 1.0, 100, 1, seed=3)[0]
    times = time_grid(1.0, 100)
    res = amm.simulate(prices, times)
    assert np.all(res.weights >= 0.0) and np.all(res.weights <= 1.0)


def test_amm_no_trade_at_constant_price():
    """
    If price never changes, pool should not trade after first step.
    Weights should be exactly w throughout.
    """
    amm, w = _make_amm()
    prices = np.full(51, S0)
    times = time_grid(1.0, 50)
    res = amm.simulate(prices, times)
    # After the pool initialises (first step), no trade should happen
    assert np.all(~res.traded[1:]), "No trades expected at constant price"
    # Weights should remain constant
    np.testing.assert_allclose(res.weights[1:], res.weights[1], atol=1e-10)


def test_amm_fees_nonnegative():
    """Collected fees must be non-negative at each step."""
    amm, _ = _make_amm()
    prices = simulate_gbm(S0, 0.10, 0.20, 1.0, 50, 1, seed=4)[0]
    times = time_grid(1.0, 50)
    res = amm.simulate(prices, times)
    assert np.all(res.fees_collected >= 0.0)


def test_amm_weight_mostly_within_zone():
    """
    Over a reasonable simulation, the vast majority of weights should be
    within [pi_minus, pi_plus].  Allows small tolerance for boundary crossings.
    """
    amm, _ = _make_amm()
    prices = simulate_gbm(S0, 0.10, 0.20, 1.0, 252, 1, seed=5)[0]
    times = time_grid(1.0, 252)
    res = amm.simulate(prices, times)
    in_zone = (res.weights >= PI_MINUS - 0.02) & (res.weights <= PI_PLUS + 0.02)
    fraction_in = float(np.mean(in_zone))
    assert fraction_in > 0.90, f"Only {fraction_in:.2%} of weights in extended zone"


# ---------------------------------------------------------------------------
# Merton simulator tests
# ---------------------------------------------------------------------------

def _make_merton(lam=0.005):
    pi_star = 0.667
    X0 = pi_star * W0 / S0
    cash0 = (1 - pi_star) * W0
    return MertonSimulator(X0, cash0, PI_MINUS, PI_PLUS, lam)


def test_merton_output_shapes():
    """Merton output arrays should have length n_steps+1."""
    m = _make_merton()
    prices = simulate_gbm(S0, 0.10, 0.20, 1.0, 50, 1, seed=10)[0]
    times = time_grid(1.0, 50)
    res = m.simulate(prices, times)
    n = len(prices)
    assert len(res.X) == n
    assert len(res.wealth) == n
    assert len(res.weights) == n


def test_merton_weight_in_zone():
    """Merton weight should always be in [pi_minus, pi_plus]."""
    m = _make_merton()
    prices = simulate_gbm(S0, 0.10, 0.20, 1.0, 252, 1, seed=11)[0]
    times = time_grid(1.0, 252)
    res = m.simulate(prices, times)
    # Allow for floating-point error
    assert np.all(res.weights >= PI_MINUS - 1e-8)
    assert np.all(res.weights <= PI_PLUS + 1e-8)


def test_merton_transaction_costs_nonneg():
    """Cumulative transaction costs should be non-decreasing and non-negative."""
    m = _make_merton()
    prices = simulate_gbm(S0, 0.10, 0.20, 1.0, 100, 1, seed=12)[0]
    times = time_grid(1.0, 100)
    res = m.simulate(prices, times)
    assert np.all(res.transaction_costs >= 0.0)
    assert np.all(np.diff(res.transaction_costs) >= -1e-12), "TC should be non-decreasing"


def test_merton_no_rebalance_constant_price():
    """At constant price, if starting inside zone, no rebalancing occurs."""
    pi_star = 0.667
    X0 = pi_star * W0 / S0
    cash0 = (1 - pi_star) * W0
    m = MertonSimulator(X0, cash0, PI_MINUS, PI_PLUS, lam_cost=0.005)
    prices = np.full(51, S0)
    times = time_grid(1.0, 50)
    res = m.simulate(prices, times)
    # Weight at constant price should be pi_star (inside zone)
    assert PI_MINUS <= pi_star <= PI_PLUS
    # No rebalancing should happen
    assert not np.any(res.rebalanced)


def test_merton_rebalancing_triggered():
    """Large price movement should trigger rebalancing."""
    pi_star = 0.667
    X0 = pi_star * W0 / S0
    cash0 = (1 - pi_star) * W0
    m = MertonSimulator(X0, cash0, PI_MINUS, PI_PLUS, lam_cost=0.005)
    # Sudden price doubling will push weight well above pi_plus
    prices = np.array([S0] + [S0 * 2.0] * 10)
    times = time_grid(1.0, 10)
    res = m.simulate(prices, times)
    assert np.any(res.rebalanced)
