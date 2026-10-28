"""
AMM Reserve Curve Construction for implementing Davis-Norman optimal strategies.

Constructs an Automated Market Maker (AMM) with a geometric-mean reserve curve
and fee parameter tau, such that the LP's inventory schedule matches a target
Davis-Norman no-trade zone strategy.

Methodology (from paper):
  1. gamma_amm = sqrt((pi_minus * (1-pi_plus)) / (pi_plus * (1-pi_minus)))
  2. tau       = 1 - gamma_amm / (1 - external_lambda)
  3. AMM weight w derived from pi_minus, pi_plus.
  4. Reserve curve: Y = f(X) = k * X^(-(1-w)/w)   [geometric-mean CFMM]
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Parameter derivation (paper Equations 1-2)
# ---------------------------------------------------------------------------

def compute_gamma_amm(pi_minus: float, pi_plus: float) -> float:
    """
    Compute the AMM fee-geometry parameter gamma_amm.

    Paper formula:
        gamma_amm = sqrt((pi_minus * (1 - pi_plus)) / (pi_plus * (1 - pi_minus)))

    This encodes how much the no-trade zone boundaries deviate from the
    centre weight and determines the effective fee needed.

    Parameters
    ----------
    pi_minus : Lower weight boundary (buy boundary).
    pi_plus  : Upper weight boundary (sell boundary).

    Returns
    -------
    float
        gamma_amm ∈ (0, 1).
    """
    assert 0.0 < pi_minus < pi_plus < 1.0, "Need 0 < pi_minus < pi_plus < 1"
    return float(np.sqrt((pi_minus * (1.0 - pi_plus)) / (pi_plus * (1.0 - pi_minus))))


def compute_fee_tau(gamma_amm: float, external_lambda: float) -> float:
    """
    Compute the AMM pool fee tau from gamma_amm and external lambda.

    Paper formula:
        tau = 1 - gamma_amm / (1 - lambda)

    tau is the fraction of each trade retained as fee by the pool; together
    with the external slippage/brokerage fee lambda it enforces the no-trade zone.

    Parameters
    ----------
    gamma_amm       : AMM geometry parameter (output of compute_gamma_amm).
    external_lambda : External fee / slippage (e.g., broker commission), ≥ 0.

    Returns
    -------
    float
        Pool fee tau ∈ [0, 1).
    """
    assert 0.0 <= external_lambda < 1.0, "external_lambda must be in [0, 1)"
    tau = 1.0 - gamma_amm / (1.0 - external_lambda)
    return float(tau)


def compute_amm_weight(pi_minus: float, pi_plus: float) -> float:
    """
    Derive the geometric-mean AMM weight w from the no-trade zone boundaries.

    The weight w satisfies the consistency condition obtained by requiring
    that the CFMM boundary prices match pi_minus and pi_plus:

        (1 - w) / w = sqrt[(1 - pi_minus)(1 - pi_plus) / (pi_minus * pi_plus)]

    Parameters
    ----------
    pi_minus : Lower portfolio-weight boundary.
    pi_plus  : Upper portfolio-weight boundary.

    Returns
    -------
    float
        Geometric-mean weight w ∈ (pi_minus, pi_plus).
    """
    ratio = np.sqrt((1.0 - pi_minus) * (1.0 - pi_plus) / (pi_minus * pi_plus))
    w = 1.0 / (1.0 + ratio)
    return float(w)


# ---------------------------------------------------------------------------
# Reserve curve: geometric-mean CFMM  Y = k * X^{-w/(1-w)}
# ---------------------------------------------------------------------------

def reserve_Y(X: np.ndarray | float, k: float, w: float) -> np.ndarray | float:
    """
    Geometric-mean reserve curve: numeraire reserves as function of risky reserves.

        Y = f(X) = k * X^{-w/(1-w)}

    Satisfies the Balancer-style invariant  X^w * Y^{1-w} = k^{1-w}.
    The LP's portfolio weight of the risky asset equals w (constant).

    Parameters
    ----------
    X : Risky asset reserves (>0).
    k : Invariant constant (>0).
    w : Weight of the risky asset ∈ (0, 1); equals the LP's portfolio weight.

    Returns
    -------
    ndarray or float
        Numeraire reserves Y.
    """
    exponent = -w / (1.0 - w)
    return k * np.asarray(X) ** exponent


def marginal_price(X: np.ndarray | float, k: float, w: float) -> np.ndarray | float:
    """
    Marginal price of risky asset implied by the reserve curve: p = -f'(X).

        p = w/(1-w) * Y/X = w/(1-w) * k * X^{-1/(1-w)}

    Parameters
    ----------
    X : Risky asset reserves (>0).
    k : Invariant constant.
    w : Weight of the risky asset.

    Returns
    -------
    ndarray or float
        Marginal price p > 0.
    """
    X = np.asarray(X)
    return w / (1.0 - w) * k * X ** (-1.0 / (1.0 - w))


def X_from_price(p: np.ndarray | float, k: float, w: float) -> np.ndarray | float:
    """
    Invert marginal_price: given external price p, compute equilibrium X.

        X(p) = [w/(1-w) * k / p]^{1-w}

    Parameters
    ----------
    p : External price (>0).
    k : Invariant constant.
    w : Weight of the risky asset.

    Returns
    -------
    ndarray or float
        Equilibrium risky asset reserves X.
    """
    p = np.asarray(p, dtype=float)
    return (w / (1.0 - w) * k / p) ** (1.0 - w)


def portfolio_weight(X: float, k: float, w: float, p: float) -> float:
    """
    LP's risky-asset portfolio weight at reserves (X, f(X)) and price p.

        pi = p * X / (p * X + f(X))

    Parameters
    ----------
    X : Risky asset reserves.
    k : Invariant constant.
    w : AMM weight.
    p : Current market price.

    Returns
    -------
    float
        Portfolio weight pi ∈ (0, 1).
    """
    Y = float(reserve_Y(X, k, w))
    return p * X / (p * X + Y)


# ---------------------------------------------------------------------------
# Target inventory schedule  q(p)  (no-trade zone)
# ---------------------------------------------------------------------------

def target_inventory_schedule(
    prices: np.ndarray,
    W0: float,
    pi_minus: float,
    pi_plus: float,
    pi_star: float,
    lam_cost: float = 0.0,
) -> np.ndarray:
    """
    Compute the Davis-Norman target inventory q(p) for a no-trade zone strategy.

    The investor:
      - Holds q* = W_t * pi* / p shares when weight is inside [pi_minus, pi_plus].
      - Rebalances to the nearest boundary when weight exits the zone.

    For the purpose of constructing the reserve curve we assume the investor
    starts at pi*, so q(p) = W0 * pi* / p (wealth stays ≈ W0 for small moves).

    Parameters
    ----------
    prices    : Array of prices.
    W0        : Initial wealth.
    pi_minus  : Lower weight boundary.
    pi_plus   : Upper weight boundary.
    pi_star   : Merton optimal fraction.
    lam_cost  : Proportional transaction cost (reduces wealth on rebalancing).

    Returns
    -------
    np.ndarray
        Target inventory q(p) for each price.
    """
    # Wealth-normalised inventory: X = W * pi_star / p
    # Clamped so that weight stays in [pi_minus, pi_plus]
    W = W0
    q = np.empty_like(prices)
    X = W * pi_star / prices[0]  # initial shares
    Y = W - prices[0] * X         # initial cash

    for i, p in enumerate(prices):
        current_weight = p * X / (p * X + Y)
        if current_weight < pi_minus:
            # Buy to pi_minus
            target_shares = pi_minus * (p * X + Y) / p
            delta = target_shares - X
            cost = p * delta * (1.0 + lam_cost)
            Y -= cost
            X = target_shares
        elif current_weight > pi_plus:
            # Sell to pi_plus
            target_shares = pi_plus * (p * X + Y) / p
            delta = X - target_shares
            proceeds = p * delta * (1.0 - lam_cost)
            Y += proceeds
            X = target_shares
        q[i] = X

    return q


# ---------------------------------------------------------------------------
# AMM initialisation helper
# ---------------------------------------------------------------------------

def initialise_pool(
    S0: float, W0: float, w: float
) -> tuple[float, float, float]:
    """
    Initialise the AMM pool at price S0 with total LP wealth W0.

    Given target weight w, the reserves satisfy:
        S0 * X0 / (S0 * X0 + Y0) = w

    We choose X0 = w * W0 / S0, Y0 = (1 - w) * W0.
    The invariant k is then derived from the curve equation.

    Parameters
    ----------
    S0 : Initial price.
    W0 : Initial LP wealth.
    w  : Geometric-mean weight.

    Returns
    -------
    tuple[float, float, float]
        (X0, Y0, k) initial reserves and invariant.
    """
    X0 = w * W0 / S0
    Y0 = (1.0 - w) * W0
    # k such that Y0 = k * X0^{-w/(1-w)}  =>  k = Y0 * X0^{w/(1-w)}
    k = Y0 * X0 ** (w / (1.0 - w))
    return X0, Y0, k
