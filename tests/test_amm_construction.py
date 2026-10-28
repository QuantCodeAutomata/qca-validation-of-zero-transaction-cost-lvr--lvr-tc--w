"""
Tests for src/amm_construction.py — AMM reserve curve construction.

Verifies:
  - gamma_amm formula (paper Eq. 1)
  - tau formula (paper Eq. 2)
  - Reserve curve mathematical properties
  - Portfolio weight derivation
  - Pool initialisation consistency
  - Edge cases
"""

import numpy as np
import pytest

from src.amm_construction import (
    compute_gamma_amm,
    compute_fee_tau,
    compute_amm_weight,
    initialise_pool,
    marginal_price,
    portfolio_weight,
    reserve_Y,
    target_inventory_schedule,
    X_from_price,
)

# Representative no-trade zone from Davis-Norman solver
PI_MINUS = 0.62
PI_PLUS = 0.72
PI_STAR = 0.667
W0 = 1000.0
S0 = 100.0
LAM_EXT = 0.003


# ---------------------------------------------------------------------------
# gamma_amm tests
# ---------------------------------------------------------------------------

def test_gamma_amm_formula():
    """Paper formula: gamma = sqrt((pi_minus*(1-pi_plus)) / (pi_plus*(1-pi_minus)))."""
    gamma = compute_gamma_amm(PI_MINUS, PI_PLUS)
    expected = np.sqrt((PI_MINUS * (1 - PI_PLUS)) / (PI_PLUS * (1 - PI_MINUS)))
    assert abs(gamma - expected) < 1e-12


def test_gamma_amm_in_unit_interval():
    """gamma_amm must lie in (0, 1)."""
    gamma = compute_gamma_amm(PI_MINUS, PI_PLUS)
    assert 0.0 < gamma < 1.0


def test_gamma_amm_increases_with_wider_zone():
    """Wider zone → smaller gamma_amm (more asymmetry between boundaries)."""
    gamma_narrow = compute_gamma_amm(0.64, 0.70)
    gamma_wide = compute_gamma_amm(0.55, 0.80)
    # Wider zone → larger relative spread → smaller gamma
    assert gamma_wide < gamma_narrow


def test_gamma_amm_symmetric_around_half():
    """When pi_minus + pi_plus = 1, gamma_amm should capture that symmetry."""
    # Both boundaries equidistant from 0.5: pi_minus = 0.3, pi_plus = 0.7
    gamma = compute_gamma_amm(0.3, 0.7)
    expected = np.sqrt((0.3 * 0.3) / (0.7 * 0.7))
    assert abs(gamma - expected) < 1e-10


def test_gamma_amm_raises_invalid_ordering():
    with pytest.raises(AssertionError):
        compute_gamma_amm(PI_PLUS, PI_MINUS)  # wrong order


# ---------------------------------------------------------------------------
# tau tests
# ---------------------------------------------------------------------------

def test_tau_formula():
    """Paper formula: tau = 1 - gamma_amm / (1 - external_lambda)."""
    gamma = compute_gamma_amm(PI_MINUS, PI_PLUS)
    tau = compute_fee_tau(gamma, LAM_EXT)
    expected = 1.0 - gamma / (1.0 - LAM_EXT)
    assert abs(tau - expected) < 1e-12


def test_tau_no_external_fee():
    """With zero external fee: tau = 1 - gamma_amm."""
    gamma = compute_gamma_amm(PI_MINUS, PI_PLUS)
    tau = compute_fee_tau(gamma, 0.0)
    assert abs(tau - (1.0 - gamma)) < 1e-12


def test_tau_nonnegative():
    """tau should be non-negative for reasonable parameters."""
    gamma = compute_gamma_amm(PI_MINUS, PI_PLUS)
    tau = compute_fee_tau(gamma, LAM_EXT)
    assert tau >= 0.0


def test_tau_less_than_one():
    """tau must be strictly less than 1."""
    gamma = compute_gamma_amm(PI_MINUS, PI_PLUS)
    tau = compute_fee_tau(gamma, LAM_EXT)
    assert tau < 1.0


# ---------------------------------------------------------------------------
# AMM weight tests
# ---------------------------------------------------------------------------

def test_amm_weight_in_unit_interval():
    """AMM weight w must be in (0, 1)."""
    w = compute_amm_weight(PI_MINUS, PI_PLUS)
    assert 0.0 < w < 1.0


def test_amm_weight_between_boundaries():
    """AMM weight w should lie between pi_minus and pi_plus."""
    w = compute_amm_weight(PI_MINUS, PI_PLUS)
    assert PI_MINUS < w < PI_PLUS


def test_amm_weight_symmetric_case():
    """For symmetric zone around 0.5, w = 0.5."""
    w = compute_amm_weight(0.3, 0.7)
    assert abs(w - 0.5) < 1e-10


# ---------------------------------------------------------------------------
# Reserve curve tests
# ---------------------------------------------------------------------------

def test_reserve_Y_positive():
    """Y must be positive for positive X and k."""
    w = compute_amm_weight(PI_MINUS, PI_PLUS)
    X0, Y0, k = initialise_pool(S0, W0, w)
    assert Y0 > 0.0
    assert reserve_Y(X0, k, w) > 0.0


def test_marginal_price_positive():
    """Marginal price must be positive."""
    w = compute_amm_weight(PI_MINUS, PI_PLUS)
    X0, Y0, k = initialise_pool(S0, W0, w)
    p = marginal_price(X0, k, w)
    assert p > 0.0


def test_reserve_curve_price_round_trip():
    """X_from_price(marginal_price(X)) == X."""
    w = compute_amm_weight(PI_MINUS, PI_PLUS)
    X0, Y0, k = initialise_pool(S0, W0, w)
    p = marginal_price(X0, k, w)
    X_back = X_from_price(p, k, w)
    assert abs(X_back - X0) / X0 < 1e-8


def test_reserve_curve_decreasing():
    """f(X) is decreasing (more X → less Y for CFMM)."""
    w = compute_amm_weight(PI_MINUS, PI_PLUS)
    X0, _, k = initialise_pool(S0, W0, w)
    X_values = np.array([X0 * 0.5, X0, X0 * 2.0])
    Y_values = reserve_Y(X_values, k, w)
    assert Y_values[0] > Y_values[1] > Y_values[2]


def test_portfolio_weight_at_init():
    """At initialisation, weight should equal w."""
    w = compute_amm_weight(PI_MINUS, PI_PLUS)
    X0, Y0, k = initialise_pool(S0, W0, w)
    p = marginal_price(X0, k, w)
    pi = portfolio_weight(X0, k, w, p)
    assert abs(pi - w) < 1e-8


def test_initialise_pool_wealth():
    """Pool total wealth should equal W0 at initialisation."""
    w = compute_amm_weight(PI_MINUS, PI_PLUS)
    X0, Y0, k = initialise_pool(S0, W0, w)
    total = S0 * X0 + Y0
    assert abs(total - W0) / W0 < 1e-10


def test_invariant_preserved():
    """
    The geometric-mean invariant X^w * Y^(1-w) should equal k^(1-w) at init.
    """
    w = compute_amm_weight(PI_MINUS, PI_PLUS)
    X0, Y0, k = initialise_pool(S0, W0, w)
    invariant = X0**w * Y0**(1 - w)
    expected = k**(1 - w)
    assert abs(invariant - expected) / expected < 1e-8


# ---------------------------------------------------------------------------
# Target inventory schedule tests
# ---------------------------------------------------------------------------

def test_target_inventory_all_positive():
    """Target inventory should be positive (holding shares)."""
    prices = np.linspace(90, 110, 50)
    q = target_inventory_schedule(prices, W0, PI_MINUS, PI_PLUS, PI_STAR)
    assert np.all(q > 0)


def test_target_inventory_weight_in_zone():
    """LP weight at target inventory should be within [pi_minus, pi_plus]."""
    prices = np.linspace(90, 110, 100)
    q = target_inventory_schedule(prices, W0, PI_MINUS, PI_PLUS, PI_STAR)
    # Re-construct cash holdings
    cash = W0 - prices[0] * (PI_STAR * W0 / prices[0])
    for i, (p, x) in enumerate(zip(prices, q)):
        W = p * x + max(cash, 0)
        if W > 0:
            pi = p * x / W
            assert PI_MINUS - 0.01 <= pi <= PI_PLUS + 0.01, (
                f"Weight {pi:.4f} violates zone [{PI_MINUS}, {PI_PLUS}] at step {i}"
            )
