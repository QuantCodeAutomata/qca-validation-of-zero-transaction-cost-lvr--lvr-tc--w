# AMM Reserve Curve Strategy: Black-Scholes Optimal Strategies via Custom Reserve Curves

## Overview

This repository implements **Experiment 3** from the paper on zero-transaction-cost LVR and the
*Retained-fee Utility-Indifference* construction for Automated Market Makers (AMMs).

The experiment demonstrates that an LP (Liquidity Provider) can implement a **Davis-Norman
no-trade zone optimal investment strategy** by choosing an appropriate AMM reserve curve and
fee structure, instead of paying explicit transaction costs for continuous rebalancing.

## Research Objectives

1. Verify that an AMM can replicate the Davis-Norman optimal investment boundaries `[π⁻, π⁺]`.
2. Test the "Retained-fee Utility-Indifference" construction for a target inventory schedule `q(p)`.
3. Confirm that tracking error decreases with path granularity.
4. Compare LP wealth against a Merton-optimal rebalancing agent.

## Methodology

1. **Davis-Norman Free-Boundary Problem**: Solve numerically for `π_minus`, `π_plus`.
2. **AMM Parameters** (paper formulas):
   - `γ_AMM = √[(π⁻(1-π⁺)) / (π⁺(1-π⁻))]`
   - `τ = 1 - γ_AMM / (1 - λ_ext)`
3. **Reserve Curve**: Geometric-mean CFMM `Y = k · X^{-(1-w)/w}` where `w` is derived from boundaries.
4. **Simulation**: Simulate AMM and Merton agent over GBM price paths.
5. **Metrics**: Tracking error, weight violations, wealth comparison.

## Repository Structure

```
├── data/
│   └── generate_paths.py       # GBM path generation
├── src/
│   ├── merton_boundary.py      # Davis-Norman solver (asymptotic + numerical)
│   ├── amm_construction.py     # Reserve curve construction
│   ├── price_simulator.py      # GBM price simulation
│   ├── amm_simulator.py        # AMM dynamics simulation
│   ├── merton_simulator.py     # Merton-optimal strategy simulation
│   └── metrics.py              # Performance metrics
├── exp/
│   └── run_experiment.py       # Main experiment runner
├── tests/
│   ├── test_merton_boundary.py
│   ├── test_amm_construction.py
│   ├── test_amm_simulator.py
│   └── test_metrics.py
├── results/
│   ├── RESULTS.md              # Experiment results summary
│   ├── metrics.json            # Quantitative metrics
│   └── fig*.png                # Plots
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Running the Experiment

```bash
python exp/run_experiment.py
```

Results (figures and metrics) are saved to `results/`.

## Running Tests

```bash
pytest tests/ -v
```

## Key Results

See `results/RESULTS.md` for detailed experimental outcomes, including:
- Davis-Norman no-trade zone boundaries
- AMM fee parameter τ
- Tracking error by path granularity
- LP vs. Merton wealth comparison

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| μ | 0.10 | Risky asset drift |
| σ | 0.20 | Risky asset volatility |
| r | 0.02 | Risk-free rate |
| γ | 3.0 | CRRA risk aversion |
| λ | 0.005 | Proportional transaction cost |
| λ_ext | 0.003 | External brokerage fee |
| S₀ | 100.0 | Initial price |
| T | 1.0 year | Simulation horizon |
| n_paths | 500 | Number of Monte Carlo paths |
