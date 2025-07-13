# Quantitative Finance Research Framework

Production-ready implementations of Heston stochastic volatility calibration and American basket option pricing using Monte Carlo methods.

## Overview

This framework provides two main components:

1. **Heston Model Calibration**: Real-time calibration of stochastic volatility models to market options data using QuantLib
2. **American Basket Options**: Pricing multi-asset American options using the Longstaff-Schwartz Monte Carlo method

## Mathematical Models

### Heston Stochastic Volatility
```
dS_t = rS_t dt + √v_t S_t dW_t^S
dv_t = κ(θ - v_t)dt + σ√v_t dW_t^v
```
Parameters: $v_0$ (initial variance), $κ$ (mean reversion), $θ$ (long-term variance), $σ$ (vol-of-vol), $ρ$ (correlation)

### American Basket Options (Longstaff-Schwartz)
Prices American options on baskets of correlated assets using Monte Carlo simulation with polynomial regression for continuation values.

## Project Structure

```
src/
├── heston_calib/                        # Heston volatility calibration
│   ├── quantlib_heston_calibrator.py    # Main calibrator class
│   ├── market_data_fetcher.py           # Real-time data acquisition
│   └── heston_calibration_pipeline.py   # End-to-end workflow
├── demo/                                # Interactive demonstrations
│   ├── quantlib_heston_demo.ipynb       # Heston calibration pipeline
│   └── basket_option_pricing_demo.ipynb # American basket options demo
├── calib/                               # Utilities and validation
│   ├── heston_model.py                  # Direct Heston implementation
│   ├── test_*.py                        # Numerical validation
│   ├── calib_utils.py                   # Helper functions
│   └── diagnose_calibration.py          # Diagnostics
├── old_calib/                           # Legacy implementations
├── longstaff_schwartz.py                # LSM algorithm implementation
└── basket_option_pricing_demo.ipynb     # Basket options demonstration
```

## Quick Start

### Heston Calibration
```python
from heston_calib.quantlib_heston_calibrator import QuantLibHestonCalibrator
from heston_calib.market_data_fetcher import MarketDataFetcher

# Fetch data and calibrate
fetcher = MarketDataFetcher(ticker='NVDA', expiry_list=['1M', '3M'])
market_data = fetcher.prepare_market_data()
spot_price = fetcher.get_spot_price()

calibrator = QuantLibHestonCalibrator(r=0.015, q=0.0)
model, results = calibrator.calibrate(spot_price, market_data, multi_start=True)
```

### American Basket Options
```python
from longstaff_schwartz import CorrelatedGBMSimulator, LongstaffSchwartzPricer

# Setup basket
S0 = [100, 110, 90]
sigma = [0.2, 0.25, 0.18]
correlation_matrix = [[1.0, 0.3, 0.2], [0.3, 1.0, 0.4], [0.2, 0.4, 1.0]]

simulator = CorrelatedGBMSimulator(S0, [0.05]*3, sigma, correlation_matrix, T=1, n_paths=100000)
pricer = LongstaffSchwartzPricer(simulator, basket_put_payoff, 'polynomial')
american_price = pricer.price_american_option()
```

## Requirements

```
numpy >= 1.21.0
pandas >= 1.3.0  
scipy >= 1.7.0
QuantLib >= 1.26
matplotlib >= 3.4.0
yfinance >= 0.1.70       # For market data
jupyter >= 1.0.0         # For notebooks
```

## Getting Started

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Heston Demo**: Open `demo/quantlib_heston_demo.ipynb`
3. **Basket Options**: Open `basket_option_pricing_demo.ipynb`

## Key Features

- **Heston Calibration**: Multi-start optimization, real-time data, comprehensive validation
- **American Options**: Correlated GBM, flexible regression bases, Greeks computation
- **Visualizations**: IV surfaces, volatility smiles, option paths, error analysis
- **Production Ready**: Robust error handling, parameter bounds, quality metrics

## References

- Heston, S. L. (1993). "A Closed-Form Solution for Options with Stochastic Volatility"
- Longstaff, F. A., & Schwartz, E. S. (2001). "Valuing American Options by Simulation"
- Gatheral, J. (2006). "The Volatility Surface: A Practitioner's Guide"

---

**Version**: 2.0.0 | **Updated**: January 2025  
*Research-grade implementations for quantitative finance applications*
