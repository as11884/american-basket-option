# Deep Neural Networks## Architecture and Methodology

### Mathematical Framework

Our approach formulates the American option pricing problem as a **Backward Stochastic Differential Equation (BSDE)**:

```
dY_t = -f(t, X_t, Y_t, Z_t)dt + Z_t dW_t
Y_T = g(X_T)
```

Where:
- Y_t: Option continuation value at time t
- Z_t: Market sensitivity (hedge ratios)  
- X_t: Underlying asset process
- f(·): Generator function encoding early exercise optimality

### Neural Architecture Components

#### RNN-BSDE Framework (Black-Scholes-Merton)
**Multi-Asset GBM Process → Dual-Head GRU → Price + Delta**

- **PriceGRU**: Learns continuation value Y_t using backward sequence processing with batch-mean look-ahead labels and paper-style BSDE residual terms
- **DeltaGRU**: Learns hedge ratios Z_t with auxiliary loss regularization
- **Innovation**: Time-reversed sequence processing for numerical stability and convergence

#### Enhanced Framework (Heston Stochastic Volatility)  
**Correlated Heston Process → Triple-Head GRU → Price + Delta + Alpha**

- **Extension to incomplete markets** with stochastic volatility modeling
- **Alpha network**: Learns volatility exposure coefficients via ridge-regularized ordinary least squares, mapping learned stock drivers to tradable share quantities in the continuation region
- **Innovation**: Dynamic hedging framework incorporating both stock and volatility risk factors

### Key Algorithmic Innovations

1. **Time-Reversed Sequence Processing**: Exploits the backward nature of BSDEs for enhanced numerical stability
2. **Smooth Payoff Regularization**: Differentiable payoff approximation enabling gradient-based optimization with maturity smoothing
3. **Multi-Scale Training**: Progressive complexity scaling from low to high-dimensional basket problems
4. **Real Market Data Validation**: Comprehensive hedging performance evaluation using actual market price dynamics

## Technical Implementationr High-Dimensional American Option Pricing: A Machine Learning Approach to Stochastic Optimal Control

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)

**A state-of-the-art computational framework bridging deep learning, stochastic control theory, and quantitative finance**

This repository implements novel deep recurrent neural network architectures for solving high-dimensional American option pricing problems—a notoriously challenging class of stochastic optimal control problems that has resisted traditional numerical methods due to the curse of dimensionality. Our approach leverages **Backward Stochastic Differential Equations (BSDEs)** and **Recurrent Neural Networks (RNNs)** to achieve scalable, accurate pricing and hedging for multi-asset derivatives under both **Black-Scholes-Merton** and **Heston stochastic volatility** models.

## Research Impact and Innovation

**Problem Statement**: American options on multi-asset baskets represent high-dimensional optimal stopping problems where traditional PDE methods fail beyond 3-4 dimensions, and Monte Carlo approaches suffer from exponential complexity growth.

**Novel Contributions**:
1. **Theoretical**: BSDE-based neural architecture that learns both continuation values and market sensitivities simultaneously
2. **Methodological**: Extension to incomplete markets (Heston model) with volatility risk hedging via learned stock drivers  
3. **Computational**: Scalable RNN implementation achieving linear complexity in dimension versus exponential for traditional methods
4. **Empirical**: Comprehensive benchmarking against industry-standard Longstaff-Schwartz Monte Carlo with real market data validation

**Applications**: Quantitative finance, derivative pricing, risk management, portfolio optimization, computational stochastic control

---

## What’s here

- **Deep RNN-BSDE (GBM)** — `rnn_model.py`  
  Two-head GRU (price & delta), batch-mean look-ahead labels, and the paper-style BSDE residual term.

- **Deep RNN-BSDE (Heston)** — `heston_rnn_model.py`  
  Three heads (price, delta, **alpha**) with OLS-based alpha labels and hedging that maps the learned **stock driver** to **shares** (continuation region).

- **Longstaff–Schwartz MC baseline** — `longstaff_schwartz.py`  
  Correlated GBM LSM for sanity checks and benchmarking.

- **Calibration (Heston)** — `heston_calib/`  
  Scripts to fit per-asset Heston parameters \((\kappa,\theta,\sigma,v_0,\rho_{sv})\) and the **stock correlation** matrix.  
  Expected CSV outputs (consumed by the Heston trainer, stored in `data/`):
  - `data/heston_parameters.csv`  (columns: `Ticker,Spot_Price,v0,kappa,theta,sigma,rho`)
  - `data/heston_correlation_matrix.csv`  (d×d, symmetric, ones on diagonal)

- **Data** — `data/`
  Storage for model files (.pth) and calibrated data (CSV files):
  - Model checkpoints: `american_heston_alpha.pth`, `american_arith_two_rnn_bsde.pth`, etc.
  - Calibrated parameters: `heston_parameters.csv`, `heston_correlation_matrix.csv`


---

## Why this project

American baskets are high-dimensional **optimal stopping** problems. PDE grids don’t scale; LSM Monte Carlo is robust but can struggle as features/dimension grow.  
This project implements a **BSDE** viewpoint where the continuation value \(Y\) **and** a **market driver** \(Z\) are learned from simulated paths by GRUs.  
Under **Heston** (incomplete market), delta alone is insufficient; we therefore learn the **stock-Brownian driver** \( \alpha \approx Z^{(S)} \) and convert it into **shares** for hedging.

---

## Quickstart

### 1) LSM baseline (GBM)

    python longstaff_schwartz.py

Produces an American price via polynomial regression on simulated **correlated GBM** paths.

### 2) Deep RNN-BSDE (GBM)

    python rnn_model.py

- **PriceGRU** learns continuation \(Y_n\)  
- **DeltaGRU** learns \(\Delta_n\) (paper-style \(Z\) term)  
- Labels from **batch-mean look-ahead**; maturity smoothing; sequence time-reversal for stability.

### 3) Deep RNN-BSDE (Heston, with alpha driver)

    python heston_rnn_model.py

- Three heads: **Price**, **Delta** (aux), **Alpha** (stock driver)  
- Per-step inputs: \([S_n,\sqrt{v_n}, g(S_n)]\) with \(g\) = signed moneyness of the basket payoff  
- **Alpha labels** via ridge OLS of discounted continuation increments on standardized stock shocks  
- Continuation-region hedging: convert \(\alpha\) to **shares**

Console output after training includes:
- \(Y_0\) (continuation) and the American value at \(t_0\)
- Average \(\Delta_0\) and average \(\alpha_0\)
- Suggested **hedge vector** (exercise: payoff gradient; continuation: driver-to-shares mapping)



## Contact

Questions and collaborations welcome—please open an issue.
