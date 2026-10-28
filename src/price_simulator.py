"""
Geometric Brownian Motion (GBM) price path simulation.

Generates simulated price paths for use in AMM and Merton strategy experiments.
"""

from __future__ import annotations

import numpy as np


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate Geometric Brownian Motion price paths.

    Exact discrete-time scheme:
        S_{t+dt} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    where Z ~ N(0, 1).

    Parameters
    ----------
    S0      : Initial price.
    mu      : Drift (annualised).
    sigma   : Volatility (annualised).
    T       : Horizon in years.
    n_steps : Number of discrete time steps.
    n_paths : Number of independent paths to simulate.
    seed    : Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Shape (n_paths, n_steps + 1) array of simulated prices.
        Each row is a single price path starting at S0.
    """
    if S0 <= 0:
        raise ValueError("S0 must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if T <= 0:
        raise ValueError("T must be positive")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")

    rng = np.random.default_rng(seed)
    dt = T / n_steps

    # Log-return increments
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    Z = rng.standard_normal(size=(n_paths, n_steps))
    log_increments = drift + diffusion * Z

    # Cumulative log returns; prepend log(S0)
    log_S = np.empty((n_paths, n_steps + 1))
    log_S[:, 0] = np.log(S0)
    log_S[:, 1:] = np.log(S0) + np.cumsum(log_increments, axis=1)

    return np.exp(log_S)


def time_grid(T: float, n_steps: int) -> np.ndarray:
    """
    Generate the time grid corresponding to simulate_gbm output.

    Parameters
    ----------
    T       : Horizon in years.
    n_steps : Number of discrete time steps.

    Returns
    -------
    np.ndarray
        Shape (n_steps + 1,) time points from 0 to T.
    """
    return np.linspace(0.0, T, n_steps + 1)
