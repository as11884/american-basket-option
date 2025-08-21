# American Basket Options — Deep RNN-BSDE vs Longstaff–Schwartz (GBM & Heston)

A research-grade framework for pricing **multi-asset American options** and producing **hedges** using:
- **GBM** baseline (Longstaff–Schwartz + paper-faithful RNN-BSDE)
- **Correlated Heston** extension with **alpha (stock driver) learning** and **vega-leak–aware** hedging
- A lightweight **Heston calibration** workflow (per-asset params & cross-asset correlation → CSV → training)

This repo is designed to be **reproducible** and **readable** for graduate-level review and practical enough for desk prototyping.

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

- **Reference** — `Efficient pricing and hedging of high-dimensional American options using deep recurrent networks.pdf` (paper for the GBM case).

> If your calibration scripts emit different filenames/paths, update the constants/paths in `heston_rnn_model.py`.

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
