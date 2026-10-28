"""
Generate and persist simulated GBM price paths for experiments.

Saves paths as NumPy .npz files under data/ for reproducibility.
This module does NOT fetch real market data — all paths are simulated.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from src.price_simulator import simulate_gbm, time_grid

DATA_DIR = Path(__file__).parent
DEFAULT_PATH_FILE = DATA_DIR / "simulated_paths.npz"

# Default experiment parameters
DEFAULT_PARAMS = dict(
    S0=100.0,
    mu=0.10,
    sigma=0.20,
    T=1.0,
    n_steps=252,
    n_paths=500,
    seed=42,
)


def generate_paths(
    S0: float = DEFAULT_PARAMS["S0"],
    mu: float = DEFAULT_PARAMS["mu"],
    sigma: float = DEFAULT_PARAMS["sigma"],
    T: float = DEFAULT_PARAMS["T"],
    n_steps: int = DEFAULT_PARAMS["n_steps"],
    n_paths: int = DEFAULT_PARAMS["n_paths"],
    seed: int = DEFAULT_PARAMS["seed"],
    save: bool = True,
    filepath: Path = DEFAULT_PATH_FILE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate GBM price paths and optionally persist to disk.

    Parameters
    ----------
    S0      : Initial price.
    mu      : Drift.
    sigma   : Volatility.
    T       : Horizon (years).
    n_steps : Number of time steps.
    n_paths : Number of independent paths.
    seed    : RNG seed.
    save    : If True, save paths to filepath.
    filepath: Destination .npz file.

    Returns
    -------
    (prices, times)
        prices : np.ndarray of shape (n_paths, n_steps+1).
        times  : np.ndarray of shape (n_steps+1,).
    """
    prices = simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, seed)
    times = time_grid(T, n_steps)

    if save:
        os.makedirs(filepath.parent, exist_ok=True)
        np.savez(
            filepath,
            prices=prices,
            times=times,
            S0=S0,
            mu=mu,
            sigma=sigma,
            T=T,
            n_steps=n_steps,
            n_paths=n_paths,
            seed=seed,
        )
        print(f"Saved {n_paths} paths ({n_steps} steps) to {filepath}")

    return prices, times


def load_paths(
    filepath: Path = DEFAULT_PATH_FILE,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load pre-generated paths from disk.

    Returns
    -------
    (prices, times, params)
    """
    data = np.load(filepath)
    prices = data["prices"]
    times = data["times"]
    params = {k: data[k].item() for k in ["S0", "mu", "sigma", "T", "n_steps", "n_paths", "seed"]}
    return prices, times, params


if __name__ == "__main__":
    prices, times = generate_paths()
    print(f"Generated paths shape: {prices.shape}")
    print(f"Price range: [{prices.min():.2f}, {prices.max():.2f}]")
    print(f"Time grid: [{times[0]:.3f}, ..., {times[-1]:.3f}]")
