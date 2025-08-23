# American Basket Options — Deep RNN-BSDE vs. Longstaff–Schwartz (GBM & Heston)

> Research-grade, reproducible code for pricing **and hedging** high-dimensional American **basket** options. Baselines include **LSMC** under correlated GBM and a paper-faithful **two-head RNN-BSDE**; we extend to **correlated Heston** with a learned stock driver (α) to improve hedging in incomplete markets.

---

## Highlights

- **Two paradigms, one repo**
  - **LSMC (Longstaff–Schwartz)** for a transparent Monte-Carlo baseline.
  - **Deep RNN-BSDE** (two heads: price *Y*, delta *Δ*) for scalable pricing & greeks along the time grid.
- **Correlated Heston extension**: adds a third head to learn the **stock Brownian driver α** and converts it to shares for hedging when delta alone is insufficient.
- **Calibration → Training pipeline**: helpers to fit per-asset Heston parameters + cross-asset correlation; exported as CSV for training.
- **Hedging focus**: daily P&L attribution (gamma/theta drift vs realized variance, vega/vanna terms, financing/borrow) and delta-hedge evaluation on simulated paths.
- **Reproducible**: deterministic seeds, minimal dependencies, CPU-friendly defaults (GPU optional).

---

## Repository structure

```
american-basket-option/
├─ data/                         # Model checkpoints & calibrated CSVs (gitignored)
├─ heston_calib/                 # Heston calibration helpers + scripts
├─ results_notebook/             # Result exploration / plotting utilities
├─ Efficient pricing and hedging of high-dimensional American options using deep recurrent networks.pdf
├─ heston_calibrator_top30.py    # Example calibration driver
├─ longstaff_schwartz.py         # Correlated-GBM LSMC baseline
├─ rnn_model.py                  # GBM RNN-BSDE (price + delta)
├─ gbm_rnn_model.py              # Alternate GBM model (optional)
├─ heston_rnn_model.py           # Heston RNN-BSDE (price + delta + alpha)
├─ __init__.py
├─ README.md
└─ license
```

---

## Install

Tested with **Python 3.10+**.

```bash
# (Recommended) Create an isolated environment
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

# Core scientific stack
pip install numpy pandas matplotlib scikit-learn torch tqdm
```

> **Tip:** On CPU, start with modest path counts (e.g., `n_paths=5_000`) for quick runs, then scale.

---

## Quickstart

### 1) LSMC baseline (GBM)

```bash
python longstaff_schwartz.py
```

Outputs an American price via polynomial regression on simulated **correlated GBM** paths.

### 2) Deep RNN-BSDE (GBM)

```bash
python rnn_model.py
```

- **Price head (Y)** learns continuation.
- **Delta head (Δ)** learns the BSDE driver term (paper-style).
- Training tricks: batch-mean look-ahead labels, mild terminal smoothing, optional sequence reversal for stability.

### 3) Deep RNN-BSDE (Heston, with α-driver)

```bash
python heston_rnn_model.py
```

- **Heads:** price, delta (aux), and **α** (stock driver).
- **Inputs per step:** \([S_n, \sqrt{v_n}, g(S_n)]\) with `g` = signed moneyness feature for the basket payoff.
- **α labels:** ridge/OLS of discounted continuation increments on standardized stock shocks.
- **Hedging:** in the continuation region, map learned α → **shares**; at exercise, use payoff gradient.

Console prints include estimates for \(Y_0\), average \(\Delta_0\), average \(\alpha_0\), and a suggested hedge vector.

---

## Data & calibration

Place CSVs in `data/`:

- `heston_parameters.csv` with columns:
  
  ```
  Ticker,Spot_Price,v0,kappa,theta,sigma,rho
  ```

- `heston_correlation_matrix.csv` — a symmetric \(d \times d\) matrix (ones on diagonal).

If your scripts produce different filenames, adjust the constants/paths in `heston_rnn_model.py`.

---
