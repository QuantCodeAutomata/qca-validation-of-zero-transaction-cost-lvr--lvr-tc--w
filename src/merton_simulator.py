"""
Merton-Optimal (Davis-Norman) Strategy Simulator.

Implements the Davis-Norman no-trade zone optimal rebalancing strategy
as a benchmark against the AMM LP position.

Strategy:
  - Maintain portfolio weight pi in the no-trade zone [pi_minus, pi_plus].
  - When weight exits the zone, rebalance to the nearest boundary.
  - Proportional transaction cost lam_cost applied on each trade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MertonResult:
    """Full trajectory returned by MertonSimulator.simulate()."""

    times: np.ndarray
    prices: np.ndarray
    X: np.ndarray           # risky asset holdings (shares)
    cash: np.ndarray        # numeraire (bond) holdings
    weights: np.ndarray     # portfolio weight pi = p*X / (p*X + cash)
    wealth: np.ndarray      # total wealth p*X + cash
    rebalanced: np.ndarray  # bool: rebalancing occurred at this step
    transaction_costs: np.ndarray  # cumulative transaction cost paid


class MertonSimulator:
    """
    Discrete-time simulator for the Davis-Norman no-trade zone strategy.

    Parameters
    ----------
    X0        : Initial risky-asset holdings (shares).
    cash0     : Initial cash (numeraire) holdings.
    pi_minus  : Lower weight boundary (buy if below this).
    pi_plus   : Upper weight boundary (sell if above this).
    lam_cost  : Proportional transaction cost (symmetric).
    r         : Risk-free rate (for cash accrual per step).
    """

    def __init__(
        self,
        X0: float,
        cash0: float,
        pi_minus: float,
        pi_plus: float,
        lam_cost: float,
        r: float = 0.0,
    ) -> None:
        self.X0 = X0
        self.cash0 = cash0
        self.pi_minus = pi_minus
        self.pi_plus = pi_plus
        self.lam_cost = lam_cost
        self.r = r

    def _rebalance(
        self, X: float, cash: float, p: float, dt: float
    ) -> tuple[float, float, bool, float]:
        """
        Apply the no-trade zone rebalancing rule at price p.

        Returns
        -------
        (X_new, cash_new, rebalanced, cost)
        """
        # Accrue risk-free return on cash
        cash = cash * np.exp(self.r * dt)

        W = p * X + cash
        pi = p * X / W

        if pi < self.pi_minus:
            # Buy risky to reach pi_minus
            target_X = self.pi_minus * W / p
            delta = target_X - X
            cost = p * delta * self.lam_cost   # cost of purchase
            # After cost: W_new = W - cost, target fraction may drift slightly
            cash_new = cash - p * delta - cost
            return target_X, cash_new, True, cost

        if pi > self.pi_plus:
            # Sell risky to reach pi_plus
            target_X = self.pi_plus * W / p
            delta = X - target_X
            cost = p * delta * self.lam_cost   # cost of sale
            cash_new = cash + p * delta - cost
            return target_X, cash_new, True, cost

        return X, cash, False, 0.0

    def simulate(
        self, price_path: np.ndarray, times: np.ndarray
    ) -> MertonResult:
        """
        Simulate the Davis-Norman strategy over a price path.

        Parameters
        ----------
        price_path : Array of shape (T+1,) with external prices.
        times      : Corresponding time grid of shape (T+1,).

        Returns
        -------
        MertonResult
            Full trajectory of strategy state.
        """
        n = len(price_path)
        X_arr = np.empty(n)
        cash_arr = np.empty(n)
        weights_arr = np.empty(n)
        wealth_arr = np.empty(n)
        rebalanced_arr = np.zeros(n, dtype=bool)
        tc_arr = np.zeros(n)

        X, cash = self.X0, self.cash0
        cum_cost = 0.0

        for i in range(n):
            p = price_path[i]
            dt = (times[i] - times[i - 1]) if i > 0 else 0.0

            X, cash, rebalanced, cost = self._rebalance(X, cash, p, dt)
            cum_cost += cost

            W = p * X + cash
            pi = p * X / W if W > 0 else 0.0

            X_arr[i] = X
            cash_arr[i] = cash
            weights_arr[i] = pi
            wealth_arr[i] = W
            rebalanced_arr[i] = rebalanced
            tc_arr[i] = cum_cost

        return MertonResult(
            times=times,
            prices=price_path,
            X=X_arr,
            cash=cash_arr,
            weights=weights_arr,
            wealth=wealth_arr,
            rebalanced=rebalanced_arr,
            transaction_costs=tc_arr,
        )
