"""
AMM Dynamics Simulator.

Simulates the evolution of a geometric-mean CFMM (Constant Function Market Maker)
pool over a discrete price path, driven by rational arbitrageurs.

Arbitrage mechanics:
  - At each time step the external price S_t is compared to the pool's
    current marginal price p_pool = (1-w)/w * Y/X.
  - With fee tau, an arbitrageur profits only if:
       S_t > p_pool / (1 - tau)   [buy risky from pool]  or
       S_t < p_pool * (1 - tau)   [sell risky to pool].
  - When profitable, the arbitrageur drives the pool to the new equilibrium
    (X_new, Y_new) such that the post-fee marginal price equals S_t.

The fee tau together with any external lambda creates an effective no-trade zone
in portfolio-weight space: [pi_minus, pi_plus].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.amm_construction import (
    marginal_price,
    portfolio_weight,
    reserve_Y,
    X_from_price,
)


@dataclass
class AMMState:
    """Snapshot of pool state at a single time step."""

    t: float
    price: float       # external price S_t
    X: float           # risky asset reserves
    Y: float           # numeraire reserves
    weight: float      # LP portfolio weight  pi = p*X/(p*X+Y)
    wealth: float      # LP total wealth  p*X + Y
    traded: bool       # whether arbitrage occurred this step


@dataclass
class AMMResult:
    """Full trajectory returned by AMMSimulator.simulate()."""

    times: np.ndarray
    prices: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    weights: np.ndarray
    wealth: np.ndarray
    traded: np.ndarray       # bool array
    fees_collected: np.ndarray  # numeraire value of fees collected per step


class AMMSimulator:
    """
    Discrete-time AMM pool simulator with geometric-mean reserve curve.

    The pool implements the 'retained-fee utility-indifference' construction:
      - Reserve curve: Y = k * X^{-(1-w)/w}
      - Pool fee fraction: tau  (fraction retained per trade)
      - External slippage / broker fee: external_lambda

    Parameters
    ----------
    X0              : Initial risky-asset reserves.
    Y0              : Initial numeraire reserves.
    k               : Reserve-curve invariant.
    w               : Geometric-mean weight of risky asset.
    tau             : Pool fee fraction ∈ [0, 1).
    external_lambda : External fee on arbitrage trades.
    """

    def __init__(
        self,
        X0: float,
        Y0: float,
        k: float,
        w: float,
        tau: float,
        external_lambda: float = 0.0,
    ) -> None:
        self.X0 = X0
        self.Y0 = Y0
        self.k = k
        self.w = w
        self.tau = tau
        self.external_lambda = external_lambda

    def _pool_price(self, X: float) -> float:
        """Pool marginal price at reserves X."""
        return float(marginal_price(X, self.k, self.w))

    def _arb_step(self, X: float, Y: float, S_ext: float) -> tuple[float, float, bool, float]:
        """
        Perform one arbitrage step.

        If the external price S_ext is outside the no-arb band around the pool
        price, rebalance the pool to the new equilibrium.

        Returns
        -------
        (X_new, Y_new, traded, fee_value)
        """
        p_pool = self._pool_price(X)
        effective_fee = 1.0 - (1.0 - self.tau) * (1.0 - self.external_lambda)
        p_low = p_pool * (1.0 - effective_fee)
        p_high = p_pool / (1.0 - effective_fee)

        if p_low <= S_ext <= p_high:
            return X, Y, False, 0.0

        # Arbitrageur drives pool price to S_ext (post-fee).
        # On a buy (S_ext > p_pool): trader sends numeraire, receives risky.
        #   The new pool marginal price (net of tau) should equal S_ext.
        #   Pool receives dy_in, sends out dx_out = dx_in such that
        #   (1-tau)*dy_in/dx_out = S_ext  =>  pool trades at S_ext/(1-tau).
        # Similarly for a sell.

        # Equilibrium X such that pool's gross price = S_ext
        X_new = float(X_from_price(S_ext, self.k, self.w))
        Y_new = float(reserve_Y(X_new, self.k, self.w))

        # Fee collected (in numeraire)
        if S_ext > p_pool:
            # Buy: numeraire flows in, risky out; fee retained from numeraire
            dy_gross = Y_new - Y   # Y_new > Y => dy_gross > 0  (pool gains numeraire)
            fee = self.tau * abs(dy_gross)
        else:
            # Sell: risky flows in, numeraire out; fee retained from risky (priced in num)
            dx_gross = X_new - X   # X_new < X => dx_gross < 0 ... wait
            # Actually on sell: X_new > X (pool gets more risky), Y_new < Y
            dx_gross = abs(X_new - X)
            fee = self.tau * S_ext * dx_gross

        return X_new, Y_new, True, fee

    def simulate(
        self, price_path: np.ndarray, times: np.ndarray
    ) -> AMMResult:
        """
        Simulate the AMM over a price path.

        Parameters
        ----------
        price_path : Array of shape (T+1,) with external prices.
        times      : Corresponding time grid of shape (T+1,).

        Returns
        -------
        AMMResult
            Full trajectory of pool state.
        """
        n = len(price_path)
        X_arr = np.empty(n)
        Y_arr = np.empty(n)
        weights_arr = np.empty(n)
        wealth_arr = np.empty(n)
        traded_arr = np.zeros(n, dtype=bool)
        fees_arr = np.zeros(n)

        X, Y = self.X0, self.Y0

        for i, (t, S) in enumerate(zip(times, price_path)):
            X, Y, traded, fee = self._arb_step(X, Y, S)
            W = S * X + Y
            pi = S * X / W if W > 0 else 0.0

            X_arr[i] = X
            Y_arr[i] = Y
            weights_arr[i] = pi
            wealth_arr[i] = W
            traded_arr[i] = traded
            fees_arr[i] = fee

        return AMMResult(
            times=times,
            prices=price_path,
            X=X_arr,
            Y=Y_arr,
            weights=weights_arr,
            wealth=wealth_arr,
            traded=traded_arr,
            fees_collected=fees_arr,
        )
