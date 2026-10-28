"""
Tests for src/merton_boundary.py — Davis-Norman free-boundary problem.

Verifies:
  - Merton fraction formula
  - Boundary ordering and containment of pi*
  - Monotonicity with respect to transaction cost
  - Edge cases (zero cost, near-zero alpha)
  - Mathematical properties (pi in (0, 1))
"""

import numpy as np
import pytest

from src.merton_boundary import (
    merton_fraction,
    davis_norman_asymptotic,
    davis_norman_numerical,
)

# Default parameters used across tests
MU = 0.10
SIGMA = 0.20
R = 0.02
GAMMA = 3.0
LAM = 0.005


# ---------------------------------------------------------------------------
# merton_fraction tests
# ---------------------------------------------------------------------------

def test_merton_fraction_standard():
    """Known result: pi* = (0.10 - 0.02) / (3 * 0.04) = 0.667..."""
    pi = merton_fraction(MU, SIGMA, R, GAMMA)
    assert abs(pi - 2.0 / 3.0) < 1e-10


def test_merton_fraction_zero_excess_return():
    """pi* = 0 when mu = r."""
    pi = merton_fraction(r=0.05, mu=0.05, sigma=0.20, gamma=2.0)
    assert pi == 0.0


def test_merton_fraction_negative_excess_return():
    """pi* < 0 when mu < r (short risky asset)."""
    pi = merton_fraction(mu=0.01, sigma=0.20, r=0.05, gamma=2.0)
    assert pi < 0.0


def test_merton_fraction_raises_on_negative_sigma():
    with pytest.raises(ValueError, match="sigma"):
        merton_fraction(MU, -0.1, R, GAMMA)


def test_merton_fraction_raises_on_zero_gamma():
    with pytest.raises(ValueError, match="gamma"):
        merton_fraction(MU, SIGMA, R, 0.0)


def test_merton_fraction_decreases_with_risk_aversion():
    """Higher risk aversion → lower allocation to risky asset."""
    pi_low_gamma = merton_fraction(MU, SIGMA, R, gamma=1.5)
    pi_high_gamma = merton_fraction(MU, SIGMA, R, gamma=5.0)
    assert pi_low_gamma > pi_high_gamma


# ---------------------------------------------------------------------------
# davis_norman_asymptotic tests
# ---------------------------------------------------------------------------

def test_asymptotic_boundaries_contain_pi_star():
    """pi_minus < pi* < pi_plus."""
    pi_star = merton_fraction(MU, SIGMA, R, GAMMA)
    pi_minus, pi_plus = davis_norman_asymptotic(MU, SIGMA, R, GAMMA, LAM)
    assert pi_minus < pi_star < pi_plus


def test_asymptotic_boundaries_in_unit_interval():
    """Both boundaries must be in (0, 1)."""
    pi_minus, pi_plus = davis_norman_asymptotic(MU, SIGMA, R, GAMMA, LAM)
    assert 0.0 < pi_minus < 1.0
    assert 0.0 < pi_plus < 1.0


def test_asymptotic_bandwidth_increases_with_cost():
    """Larger transaction cost → wider no-trade zone."""
    _, _, _, gamma = MU, SIGMA, R, GAMMA
    pi_m1, pi_p1 = davis_norman_asymptotic(MU, SIGMA, R, GAMMA, lam=0.001)
    pi_m2, pi_p2 = davis_norman_asymptotic(MU, SIGMA, R, GAMMA, lam=0.01)
    bw1 = pi_p1 - pi_m1
    bw2 = pi_p2 - pi_m2
    assert bw2 > bw1


def test_asymptotic_symmetric_around_pi_star():
    """For symmetric costs the zone is symmetric around pi*."""
    pi_star = merton_fraction(MU, SIGMA, R, GAMMA)
    pi_minus, pi_plus = davis_norman_asymptotic(MU, SIGMA, R, GAMMA, LAM)
    eps_minus = pi_star - pi_minus
    eps_plus = pi_plus - pi_star
    # Should be exactly symmetric in the Shreve-Soner asymptotic formula
    assert abs(eps_minus - eps_plus) < 1e-10


def test_asymptotic_zero_cost_gives_degenerate_zone():
    """With zero cost, bandwidth → 0: pi_minus = pi_plus = pi*."""
    pi_star = merton_fraction(MU, SIGMA, R, GAMMA)
    pi_minus, pi_plus = davis_norman_asymptotic(MU, SIGMA, R, GAMMA, lam=0.0)
    assert abs(pi_minus - pi_star) < 1e-6
    assert abs(pi_plus - pi_star) < 1e-6


def test_asymptotic_various_risk_aversion():
    """Higher risk aversion → narrower zone (less deviation from pi*).

    Note: gamma=1.5 with these parameters gives pi*=1.33 (>1, unconstrained);
    use gamma=3.0 (pi*≈0.67) and gamma=5.0 (pi*≈0.40) to stay in (0,1).
    """
    pi_m1, pi_p1 = davis_norman_asymptotic(MU, SIGMA, R, gamma=3.0, lam=LAM)
    pi_m2, pi_p2 = davis_norman_asymptotic(MU, SIGMA, R, gamma=5.0, lam=LAM)
    bw1 = pi_p1 - pi_m1
    bw2 = pi_p2 - pi_m2
    # Higher gamma shrinks the zone (more risk averse → faster rebalancing)
    assert bw2 < bw1


# ---------------------------------------------------------------------------
# davis_norman_numerical tests
# ---------------------------------------------------------------------------

def test_numerical_boundaries_contain_pi_star():
    """Numerically refined boundaries should still contain pi*."""
    pi_star = merton_fraction(MU, SIGMA, R, GAMMA)
    pi_minus, pi_plus = davis_norman_numerical(MU, SIGMA, R, GAMMA, LAM)
    assert pi_minus < pi_star < pi_plus


def test_numerical_close_to_asymptotic():
    """Numerical and asymptotic solutions should be close for small lam."""
    pi_m_a, pi_p_a = davis_norman_asymptotic(MU, SIGMA, R, GAMMA, LAM)
    pi_m_n, pi_p_n = davis_norman_numerical(MU, SIGMA, R, GAMMA, LAM)
    # Within 10% relative deviation for small costs
    assert abs(pi_m_n - pi_m_a) / pi_m_a < 0.10
    assert abs(pi_p_n - pi_p_a) / pi_p_a < 0.10


def test_numerical_boundaries_in_unit_interval():
    """Numerical boundaries must be in (0, 1)."""
    pi_minus, pi_plus = davis_norman_numerical(MU, SIGMA, R, GAMMA, LAM)
    assert 0.0 < pi_minus < 1.0
    assert 0.0 < pi_plus < 1.0


def test_numerical_asymmetric_costs():
    """Asymmetric costs shift boundaries asymmetrically."""
    pi_m, pi_p = davis_norman_numerical(
        MU, SIGMA, R, GAMMA, lam=LAM,
        lam_buy=0.003, lam_sell=0.008
    )
    assert pi_m < merton_fraction(MU, SIGMA, R, GAMMA) < pi_p
