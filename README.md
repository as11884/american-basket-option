# Heston Stochastic Volatility Model Calibration Framework

A production-ready implementation of Heston model calibration for options pricing using QuantLib, featuring robust numerical methods and comprehensive market data analysis.

## Mathematical Foundation

### The Heston Model

The Heston stochastic volatility model describes the evolution of an asset price $S_t$ and its instantaneous variance $v_t$ under the risk-neutral measure:

```
dS_t = rS_t dt + √v_t S_t dW_t^S
dv_t = κ(θ - v_t)dt + σ√v_t dW_t^v
```

where:
- $r$ is the risk-free rate
- $κ > 0$ is the mean reversion speed
- $θ > 0$ is the long-term variance level  
- $σ > 0$ is the volatility of volatility
- $ρ = \mathbb{E}[dW_t^S dW_t^v]$ is the correlation between asset and volatility processes

**Feller Condition**: For $v_t > 0$ almost surely, we require $2κθ > σ^2$.

### Characteristic Function

The Heston model admits a semi-analytical solution via the characteristic function:

```
φ(u, S_0, v_0, τ) = exp(C(τ, u) + D(τ, u)v_0 + iu ln(S_0))
```

where $C(τ, u)$ and $D(τ, u)$ are complex-valued functions satisfying the Riccati equations. European option prices are obtained via Fourier inversion.

### Calibration Methodology

The calibration minimizes the weighted least squares objective:

```
min Σᵢ wᵢ(IVᵢᵐᵃʳᵏᵉᵗ - IVᵢᵐᵒᵈᵉˡ(Θ))²
```

where $Θ = \{v_0, κ, θ, σ, ρ\}$ are the model parameters and $w_i$ are option-specific weights.

## Architecture

### Core Components

```
src/
├── heston_calib/           # Core calibration engine
│   ├── quantlib_heston_calibrator.py    # Main calibrator class
│   ├── market_data_fetcher.py           # Real-time data acquisition
│   └── heston_calibration_pipeline.py   # End-to-end workflow
├── demo/                   # Interactive demonstrations
│   └── quantlib_heston_demo.ipynb       # Complete calibration pipeline
├── calib/                  # Utilities and validation
│   ├── heston_model.py                  # Direct implementation
│   └── test_*.py                        # Numerical validation
└── longstaff_schwartz.py   # American options via Monte Carlo
```

### Class Hierarchy

```python
QuantLibHestonCalibrator
├── __init__(r, q, bounds=None)
├── calibrate(spot, market_data, multi_start=False)
├── _setup_helpers(market_data)
├── _optimize_parameters(helpers, multi_start)
└── _validate_calibration(model, helpers)
```

## Usage

### Basic Calibration

```python
from heston_calib.quantlib_heston_calibrator import QuantLibHestonCalibrator
from heston_calib.market_data_fetcher import MarketDataFetcher

# Fetch market data
fetcher = MarketDataFetcher(
    ticker='NVDA',
    expiry_list=['1W', '1M', '3M'],
    atm_range=0.15
)
market_data = fetcher.prepare_market_data()
spot_price = fetcher.get_spot_price()

# Initialize calibrator
calibrator = QuantLibHestonCalibrator(r=0.015, q=0.0)

# Calibrate with multiple starting points for robustness
model, results = calibrator.calibrate(
    spot=spot_price,
    market_data=market_data,
    multi_start=True
)

# Extract parameters
params = results['calibrated_params']
v0, kappa, theta, sigma, rho = [params[k] for k in ['v0', 'kappa', 'theta', 'sigma', 'rho']]
```

### Advanced Configuration

```python
# Custom parameter bounds
bounds = {
    'v0': (0.001, 1.0),      # Initial variance
    'kappa': (0.1, 20.0),    # Mean reversion speed  
    'theta': (0.001, 1.0),   # Long-term variance
    'sigma': (0.01, 2.0),    # Vol of vol
    'rho': (-0.99, 0.99)     # Correlation
}

calibrator = QuantLibHestonCalibrator(
    r=0.015, 
    q=0.0, 
    bounds=bounds
)

# Multi-start optimization with custom attempts
model, results = calibrator.calibrate(
    spot=spot_price,
    market_data=market_data,
    multi_start=True,
    max_attempts=10
)
```

## Numerical Implementation

### Optimization Algorithm

- **Primary**: Levenberg-Marquardt with analytical Jacobians
- **Fallback**: Differential Evolution for global optimization
- **Multi-start**: Random parameter initialization for robustness

### Convergence Criteria

- **Relative tolerance**: 1e-8 on parameter changes
- **Function tolerance**: 1e-10 on objective function  
- **Maximum iterations**: 1000 per optimization attempt

### Error Handling

```python
# Comprehensive validation
if not results['success']:
    logger.error(f"Calibration failed: {results['error']}")
    
if not results['feller_satisfied']:
    logger.warning("Feller condition violated - model may exhibit issues")
    
if results['average_error'] > 0.05:
    logger.warning("High calibration error - consider parameter bounds")
```

## Performance Metrics

### Calibration Quality Assessment

| Metric | Excellent | Good | Fair | Poor |
|--------|-----------|------|------|------|
| Mean Absolute IV Error | < 2% | < 5% | < 10% | ≥ 10% |
| IV Correlation | > 0.9 | > 0.8 | > 0.6 | ≤ 0.6 |
| RMSE | < 0.02 | < 0.05 | < 0.10 | ≥ 0.10 |

### Computational Complexity

- **Single calibration**: O(n × m) where n = options, m = iterations
- **Multi-start**: O(k × n × m) where k = starting points
- **Typical runtime**: 10-30 seconds for 50-100 options

## Model Validation

### Statistical Tests

1. **Feller Condition**: $2κθ > σ^2$
2. **Parameter Stability**: Bootstrapped confidence intervals
3. **Residual Analysis**: IV error distribution normality
4. **Cross-validation**: Out-of-sample performance

### Market Microstructure

```python
# Volatility smile analysis
def analyze_smile(iv_data):
    """Analyze volatility smile characteristics"""
    return {
        'atm_level': iv_data[iv_data['moneyness'].abs() < 0.01]['iv'].mean(),
        'skew': compute_25delta_skew(iv_data),
        'convexity': compute_butterfly_spread(iv_data)
    }

# Term structure analysis  
def analyze_term_structure(iv_data):
    """Analyze IV term structure"""
    return iv_data.groupby('days_to_expiry')['iv'].mean()
```

## Dependencies

### Core Requirements

```
numpy >= 1.21.0          # Numerical computing
pandas >= 1.3.0          # Data manipulation  
scipy >= 1.7.0           # Optimization algorithms
QuantLib >= 1.26         # Financial mathematics
matplotlib >= 3.4.0      # Visualization
seaborn >= 0.11.0        # Statistical plotting
```

### Optional Extensions

```
yfinance >= 0.1.70       # Market data fetching
plotly >= 5.0.0          # Interactive visualization
jupyter >= 1.0.0         # Notebook environment
```

## File Documentation

### Primary Modules

- **`quantlib_heston_calibrator.py`**: Production-ready calibrator with multi-start optimization and comprehensive error handling
- **`market_data_fetcher.py`**: Real-time options data acquisition with quality filtering
- **`heston_calibration_pipeline.py`**: High-level orchestration of the calibration workflow

### Demonstration

- **`quantlib_heston_demo.ipynb`**: Complete calibration pipeline with enhanced visualizations, error analysis, and trading insights

### Testing Framework

- **`test_heston_cf.py`**: Characteristic function implementation validation
- **`test_heston_sensitivity.py`**: Greeks and parameter sensitivity analysis
- **`test_direct_heston.py`**: Direct vs QuantLib implementation comparison

### Legacy Components

- **`calib/`**: Alternative implementations for comparison and validation
- **`old_calib/`**: Archived versions maintained for reference

## Error Analysis

### Common Calibration Issues

1. **Insufficient Data**: < 20 liquid options
2. **Parameter Bounds**: Overly restrictive constraints
3. **Market Stress**: Extreme volatility regimes
4. **Model Limitations**: Cannot capture all smile dynamics

### Diagnostic Tools

```python
# Calibration diagnostics
def diagnose_calibration(results, iv_comparison):
    """Comprehensive calibration analysis"""
    diagnostics = {
        'parameter_stability': check_parameter_bounds(results),
        'error_distribution': analyze_residuals(iv_comparison),
        'smile_fit_quality': assess_smile_reproduction(iv_comparison),
        'term_structure_fit': assess_term_structure(iv_comparison)
    }
    return diagnostics
```

## Research Extensions

### Advanced Features

- **Jump-diffusion extensions**: Bates and SVJ models
- **Multi-factor models**: Double Heston implementation  
- **Regime-switching**: Markov-modulated parameters
- **Machine learning**: Neural network parameter estimation

### Academic References

1. Heston, S. L. (1993). "A Closed-Form Solution for Options with Stochastic Volatility"
2. Gatheral, J. (2006). "The Volatility Surface: A Practitioner's Guide"
3. Rouah, F. D. (2013). "The Heston Model and Its Extensions in Matlab and C#"

## License

This implementation is provided for research and educational purposes. Commercial usage requires appropriate licensing of QuantLib and compliance with relevant financial regulations.

---

**Authors**: Mathematical Finance Research Team  
**Version**: 2.0.0 (Production Ready)  
**Last Updated**: January 2025  
**Contact**: For technical support, refer to the test suite and demo notebook examples

## License
MIT License

---

*This project is designed for research and educational purposes. Contributions and suggestions are welcome!*
