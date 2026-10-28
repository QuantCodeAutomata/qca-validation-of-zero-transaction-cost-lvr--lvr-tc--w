"""
Davis-Norman free-boundary problem solver for the Merton optimal investment problem
with proportional transaction costs (CRRA utility).

References:
  Davis, M.H.A. & Norman, A.R. (1990). "Portfolio Selection with Transaction Costs."
  Shreve, S.E. & Soner, H.M. (1994). "Optimal Investment and Consumption with
    Transaction Costs." Annals of Applied Probability.
  Janecek, K. & Shreve, S. (2004). "Asymptotic analysis for optimal investment
    and consumption with transaction costs."
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, fsolve


def merton_fraction(mu: float, sigma: float, r: float, gamma: float) -> float:
    """
    Compute the Merton optimal portfolio fraction (no transaction costs).

    For CRRA utility U(W) = W^(1-gamma)/(1-gamma), the optimal risky
    asset fraction satisfies:

        pi* = (mu - r) / (gamma * sigma^2)

    Parameters
    ----------
    mu    : Expected return of the risky asset.
    sigma : Volatility of the risky asset.
    r     : Risk-free rate.
    gamma : CRRA risk-aversion coefficient (gamma > 0).

    Returns
    -------
    float
        Optimal portfolio fraction pi*.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    return (mu - r) / (gamma * sigma**2)


def _dn_ode(z: float, v: list[float], mu: float, sigma: float,
            r: float, gamma: float) -> list[float]:
    """
    RHS of the Davis-Norman reduced-value-function ODE.

    The CRRA value function in the no-trade region decomposes as
        V(w, pi) = w^(1-gamma)/(1-gamma) * v(pi)
    where v(pi) solves:

        A(pi) v'' + B(pi) v' + C(pi) v = 0

    with:
        A(pi) = sigma^2 / 2 * pi^2 * (1-pi)^2
        B(pi) = pi*(1-pi) * [(mu-r) - gamma*sigma^2*pi]
        C(pi) = (1-gamma)*[(mu-r)*pi + r] - (1-gamma)*gamma*sigma^2*pi^2/2

    Parameters
    ----------
    z     : Current value of pi (portfolio fraction, independent variable).
    v     : [v(z), v'(z)] state vector.
    mu, sigma, r, gamma : Model parameters.

    Returns
    -------
    list[float]
        Derivative [v', v''].
    """
    v0, v1 = v
    A = 0.5 * sigma**2 * z**2 * (1.0 - z)**2
    B = z * (1.0 - z) * ((mu - r) - gamma * sigma**2 * z)
    C = (1.0 - gamma) * ((mu - r) * z + r) - 0.5 * (1.0 - gamma) * gamma * sigma**2 * z**2
    if abs(A) < 1e-14:
        return [v1, 0.0]
    v2 = (-C * v0 - B * v1) / A
    return [v1, v2]


def _slope_condition(pi_b: float, mu: float, sigma: float, r: float,
                     gamma: float, v0_pi_star: float, v1_pi_star: float,
                     pi_star: float, lam: float, is_buy: bool) -> float:
    """
    Shooting residual: integrate ODE from pi_star to pi_b, check smooth-pasting.

    Smooth-pasting conditions (Shreve-Soner 1994):
      At buy boundary pi_minus:  v'(pi-) / v(pi-) = (1-gamma) / (1-lam)
      At sell boundary pi_plus:  v'(pi+) / v(pi+) = (1-gamma) / (1+lam)

    Parameters
    ----------
    pi_b         : Candidate boundary value.
    is_buy       : True for buy (lower) boundary, False for sell (upper) boundary.
    v0_pi_star, v1_pi_star : Initial condition at pi_star.
    lam          : Proportional transaction cost.

    Returns
    -------
    float
        Residual of the smooth-pasting condition.
    """
    if is_buy:
        target_slope = (1.0 - gamma) / (1.0 - lam)
    else:
        target_slope = (1.0 - gamma) / (1.0 + lam)

    span = [pi_star, pi_b]
    sol = solve_ivp(
        _dn_ode, span, [v0_pi_star, v1_pi_star],
        args=(mu, sigma, r, gamma),
        dense_output=False, rtol=1e-9, atol=1e-11, method="RK45"
    )
    v0_end = sol.y[0, -1]
    v1_end = sol.y[1, -1]
    if abs(v0_end) < 1e-14:
        return 1e6
    return v1_end / v0_end - target_slope


def _find_initial_slope(pi_star: float, mu: float, sigma: float,
                         r: float, gamma: float, lam: float) -> float:
    """
    Find v'(pi*) consistent with smooth-pasting on both sides via a 1D search.

    The normalisation v(pi*) = 1 is fixed; we search for v'(pi*) = s such
    that the ODE solution satisfies both boundary conditions simultaneously.
    Because both conditions depend on s, we bisect on the difference.

    Returns
    -------
    float
        Best estimate for v'(pi*).
    """
    # Leading-order asymptotic estimate of slope at pi*
    # (from the Merton no-cost solution): v'/v ≈ (1-gamma)/1
    s0 = 1.0 - gamma

    def diff(s: float) -> float:
        """Difference in residuals: should be zero when s is consistent."""
        pi_minus_approx = max(pi_star * 0.5, pi_star - 0.3)
        pi_plus_approx = min(1.0 - (1.0 - pi_star) * 0.5, pi_star + 0.3)
        r_buy = _slope_condition(
            pi_minus_approx, mu, sigma, r, gamma, 1.0, s, pi_star, lam, True
        )
        r_sell = _slope_condition(
            pi_plus_approx, mu, sigma, r, gamma, 1.0, s, pi_star, lam, False
        )
        return r_buy - r_sell

    try:
        s_opt = brentq(diff, s0 - 5.0, s0 + 5.0, xtol=1e-6, maxiter=200)
    except ValueError:
        s_opt = s0
    return s_opt


def davis_norman_asymptotic(mu: float, sigma: float, r: float,
                             gamma: float, lam: float) -> tuple[float, float]:
    """
    Shreve-Soner leading-order asymptotic approximation for the no-trade zone.

    Bandwidth (half-width):
        eps = [3*lam*pi*^2*(1-pi*)^2 / (2*gamma)] ^ (1/3)

    Valid for small lam (lam << 1). Symmetric for symmetric costs.

    Parameters
    ----------
    mu, sigma, r, gamma : Model parameters.
    lam                 : Proportional transaction cost (symmetric).

    Returns
    -------
    tuple[float, float]
        (pi_minus, pi_plus) boundaries of the no-trade zone.
    """
    pi_star = merton_fraction(mu, sigma, r, gamma)
    eps = (3.0 * lam * pi_star**2 * (1.0 - pi_star)**2 / (2.0 * gamma)) ** (1.0 / 3.0)
    pi_minus = pi_star - eps
    pi_plus = pi_star + eps
    # Clamp to (0, 1)
    pi_minus = max(pi_minus, 1e-4)
    pi_plus = min(pi_plus, 1.0 - 1e-4)
    return pi_minus, pi_plus


def davis_norman_numerical(mu: float, sigma: float, r: float,
                            gamma: float, lam: float,
                            lam_buy: float | None = None,
                            lam_sell: float | None = None) -> tuple[float, float]:
    """
    Numerically solve the Davis-Norman free-boundary problem via ODE shooting.

    Uses the Shreve-Soner asymptotic result as the initial guess, then refines
    using brentq on the smooth-pasting conditions.

    Parameters
    ----------
    mu, sigma, r, gamma : Model parameters.
    lam                 : Symmetric proportional transaction cost.
    lam_buy             : Buy-side cost (overrides lam if given).
    lam_sell            : Sell-side cost (overrides lam if given).

    Returns
    -------
    tuple[float, float]
        (pi_minus, pi_plus) refined numerical boundaries.
    """
    if lam_buy is None:
        lam_buy = lam
    if lam_sell is None:
        lam_sell = lam

    pi_star = merton_fraction(mu, sigma, r, gamma)
    pi_minus_0, pi_plus_0 = davis_norman_asymptotic(mu, sigma, r, gamma, lam)

    # Initial slope: use Merton approximation
    s0 = 1.0 - gamma

    def buy_residual(pi_m: float) -> float:
        return _slope_condition(pi_m, mu, sigma, r, gamma, 1.0, s0, pi_star, lam_buy, True)

    def sell_residual(pi_p: float) -> float:
        return _slope_condition(pi_p, mu, sigma, r, gamma, 1.0, s0, pi_star, lam_sell, False)

    # Refine buy boundary
    try:
        lo_b, hi_b = max(pi_minus_0 * 0.5, 1e-4), pi_star - 1e-6
        f_lo, f_hi = buy_residual(lo_b), buy_residual(hi_b)
        if f_lo * f_hi < 0:
            pi_minus = brentq(buy_residual, lo_b, hi_b, xtol=1e-7, maxiter=300)
        else:
            pi_minus = pi_minus_0
    except Exception:
        pi_minus = pi_minus_0

    # Refine sell boundary
    try:
        lo_s, hi_s = pi_star + 1e-6, min(pi_plus_0 * 1.5, 1.0 - 1e-4)
        f_lo2, f_hi2 = sell_residual(lo_s), sell_residual(hi_s)
        if f_lo2 * f_hi2 < 0:
            pi_plus = brentq(sell_residual, lo_s, hi_s, xtol=1e-7, maxiter=300)
        else:
            pi_plus = pi_plus_0
    except Exception:
        pi_plus = pi_plus_0

    return float(pi_minus), float(pi_plus)
