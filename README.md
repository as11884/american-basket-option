# American Basket Option Pricing: Deep Learning vs Monte Carlo Methods

**A Comparative Study of Neural Network and Traditional Approaches for Multi-Asset Derivative Valuation**

## Abstract

This repository presents a comprehensive research framework for pricing American basket options using two complementary methodologies: traditional Longstaff-Schwartz Monte Carlo simulation and deep recurrent neural networks based on backward stochastic differential equations (BSDEs). The project implements production-grade algorithms for multi-asset derivative pricing with full correlation structure modeling, providing both theoretical foundations and empirical comparisons of computational finance techniques.

## Research Motivation

American basket options represent a significant challenge in computational finance due to:

1. **Optimal Stopping Problem**: The holder's right to exercise at any time before maturity creates a complex dynamic programming problem
2. **Multi-Asset Complexity**: Correlation structures between underlying assets create high-dimensional pricing challenges  
3. **Path Dependence**: American features require full path simulation rather than closed-form solutions
4. **Computational Efficiency**: Traditional Monte Carlo methods face the curse of dimensionality

This research investigates whether modern deep learning approaches can overcome these limitations while maintaining pricing accuracy comparable to established Monte Carlo techniques.

## Theoretical Framework

### Mathematical Foundation

We consider American basket options on *d* correlated assets with prices following the multivariate geometric Brownian motion:

```
dS_i(t) = r S_i(t) dt + σ_i S_i(t) Σ_{j=1}^d L_{ij} dW_j(t)
```

where:
- **S_i(t)**: Price of asset *i* at time *t*
- **r**: Risk-free interest rate
- **σ_i**: Volatility of asset *i* 
- **L**: Cholesky decomposition of correlation matrix
- **W_j(t)**: Independent Brownian motions

The American basket option value satisfies the variational inequality:
```
max(∂V/∂t + LV - rV, g(S,t) - V) = 0
```

where **g(S,t)** is the payoff function and **L** is the Black-Scholes differential operator.

### Methodological Approaches

#### 1. Longstaff-Schwartz Monte Carlo (LSM)
**Reference**: Longstaff & Schwartz (2001) - "Valuing American Options by Simulation"

The LSM algorithm solves the optimal stopping problem through:
- **Path Generation**: Simulate correlated asset price paths under risk-neutral measure
- **Backward Induction**: Use polynomial regression to estimate continuation values
- **Exercise Decision**: Compare immediate payoff with continuation value at each time step

**Key Innovation**: Transforms infinite-dimensional functional optimization into finite-dimensional regression problem.

#### 2. Deep RNN-BSDE Approach  
**Reference**: Beck et al. (2019) - "Efficient pricing and hedging of high-dimensional American options using deep recurrent networks"

The neural network approach reformulates American option pricing as a BSDE:
```
dY_t = -f(t, X_t, Y_t, Z_t)dt + Z_t dW_t
Y_T = g(X_T)
```

**Architecture Components**:
- **PriceGRU**: Estimates continuation value Y_t
- **DeltaGRU**: Computes hedge ratios Z_t  
- **Look-ahead Labels**: Batch-mean optimal stopping time selection
- **BSDE Loss**: Paper-faithful implementation with variance reduction

## Implementation Architecture

### Core Components

```
src/
├── longstaff_schwartz.py              # LSM implementation with correlated GBM
├── rnn_model.py                       # Deep RNN-BSDE pricing engine
├── heston_calib/                      # Stochastic volatility calibration
│   ├── quantlib_heston_calibrator.py  # Market-based parameter estimation
│   ├── market_data_fetcher.py         # Real-time options data acquisition
│   └── covariance_estimator.py        # Correlation matrix estimation
├── demo/                              # Comparative analysis notebooks
│   ├── rnn_vs_lsm_gbm_comparison.ipynb      # Primary methodology comparison
│   └── correlated_models_comparison_streamlined.ipynb  # Extended analysis
└── american_arith_two_rnn_bsde.pth    # Trained neural network model
```

### Technical Specifications

#### Longstaff-Schwartz Implementation
- **Basis Functions**: Polynomial regression with configurable degree
- **Path Generation**: Correlated geometric Brownian motion via Cholesky decomposition  
- **Basket Types**: Both arithmetic and geometric basket support
- **Variance Reduction**: Antithetic variates and control variates
- **Greeks Computation**: Finite difference and pathwise derivative methods

#### RNN-BSDE Implementation
- **Network Architecture**: Multi-layer GRU with Swish activation
- **Training Strategy**: Batch-mean look-ahead with adaptive kappa scheduling
- **Loss Function**: BSDE residual with delta term weighting
- **Numerical Stability**: Gradient clipping, finite value checks, smooth payoff approximation
- **Correlation Handling**: Full matrix support with backward compatibility

### Research Extensions

The framework supports several research directions:

1. **Stochastic Volatility**: Integration with Heston model parameters
2. **Jump Processes**: Extension to Merton jump-diffusion models  
3. **Multi-Currency**: Cross-currency basket options with FX correlation
4. **Exotic Payoffs**: Barrier options, lookback options, rainbow options
5. **Reinforcement Learning**: Alternative optimal stopping approaches


## Academic References

### Primary Sources
1. **Longstaff, F. A., & Schwartz, E. S.** (2001). "Valuing American Options by Simulation: A Simple Least-Squares Approach." *Review of Financial Studies*, 14(1), 113-147.

2. **Beck, C., Becker, S., Grohs, P., Jaafari, N., & Jentzen, A.** (2019). "Solving the Kolmogorov PDE by means of deep learning." *arXiv preprint arXiv:1906.02563*.

3. **Heston, S. L.** (1993). "A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options." *Review of Financial Studies*, 6(2), 327-343.

### Methodological References  
4. **Glasserman, P.** (2003). *Monte Carlo Methods in Financial Engineering*. Springer Science & Business Media.

5. **Raissi, M., Perdikaris, P., & Karniadakis, G. E.** (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics*, 378, 686-707.

6. **Sirignano, J., & Spiliopoulos, K.** (2018). "DGM: A deep learning algorithm for solving partial differential equations." *Journal of Computational Physics*, 375, 1339-1364.

