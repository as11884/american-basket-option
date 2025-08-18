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

- **Calibration (Heston)** — `calibration/`  
  Scripts to fit per-asset Heston parameters \((\kappa,\theta,\sigma,v_0,\rho_{sv})\) and the **stock correlation** matrix.  
  Expected CSV outputs (consumed by the Heston trainer):
  - `heston_parameters.csv`  (columns: `asset,v0,kappa,theta,sigma,rho_sv`)
  - `heston_correlation_matrix.csv`  (d×d, symmetric, ones on diagonal)

- **Reference** — `Efficient pricing and hedging of high-dimensional American options using deep recurrent networks.pdf` (paper for the GBM case).

> If your calibration scripts emit different filenames/paths, update the constants/paths in `heston_rnn_model.py`.

---

## Why this project

American baskets are high-dimensional **optimal stopping** problems. PDE grids don’t scale; LSM Monte Carlo is robust but can struggle as features/dimension grow.  
This project implements a **BSDE** viewpoint where the continuation value \(Y\) **and** a **market driver** \(Z\) are learned from simulated paths by GRUs.  
Under **Heston** (incomplete market), delta alone is insufficient; we therefore learn the **stock-Brownian driver** \( \alpha \approx Z^{(S)} \) and convert it into **shares** for hedging.

---

## Quickstart

### 1) Environment

    python -m venv .venv
    # Windows: .venv\Scripts\activate
    # macOS/Linux:
    source .venv/bin/activate
    pip install --upgrade pip
    pip install numpy torch pandas scipy scikit-learn matplotlib

> GPU is auto-used if available (`torch.cuda.is_available()`).

### 2) LSM baseline (GBM)

    python longstaff_schwartz.py

Produces an American price via polynomial regression on simulated **correlated GBM** paths.

### 3) Deep RNN-BSDE (GBM)

    python rnn_model.py

- **PriceGRU** learns continuation \(Y_n\)  
- **DeltaGRU** learns \(\Delta_n\) (paper-style \(Z\) term)  
- Labels from **batch-mean look-ahead**; maturity smoothing; sequence time-reversal for stability.

### 4) Deep RNN-BSDE (Heston, with alpha driver)

    python heston_rnn_model.py

- Three heads: **Price**, **Delta** (aux), **Alpha** (stock driver)  
- Per-step inputs: \([S_n,\sqrt{v_n}, g(S_n)]\) with \(g\) = signed moneyness of the basket payoff  
- **Alpha labels** via ridge OLS of discounted continuation increments on standardized stock shocks  
- Continuation-region hedging: convert \(\alpha\) to **shares**

Console output after training includes:
- \(Y_0\) (continuation) and the American value at \(t_0\)
- Average \(\Delta_0\) and average \(\alpha_0\)
- Suggested **hedge vector** (exercise: payoff gradient; continuation: driver-to-shares mapping)

---

## Using your own **Heston** parameters (calibration)

### A) Run calibration

Place your scripts in `calibration/` (or adapt paths). They should write:

- `calibration/heston_parameters.csv`  
  Columns per asset: `asset,v0,kappa,theta,sigma,rho_sv`

- `calibration/heston_correlation_matrix.csv`  
  A d×d CSV with ones on the diagonal and symmetric off-diagonals for **stock** correlations.

### B) Wire into training

In `heston_rnn_model.py`, either:
- Load those CSVs (recommended), or
- Paste arrays directly into the trainer (quick test).

Training will:
1. simulate **correlated Heston** paths with your parameters,  
2. learn price + hedge heads,  
3. save a checkpoint (e.g., `american_heston_alpha.pth`) for later inference.

---

## How it works (succinct)

### Look-ahead labels (batch-mean \(j^\*\))

At step \(n\), pick a **single** \(j^\* \in \{n{+}1,\ldots,N\}\) maximizing **batch-mean discounted payoff**.  
Define labels:
\[
c_n = e^{-r (t_{j^\*}-t_n)} f(S_{j^\*}),\qquad
\nabla c_n \approx e^{-r(\cdot)} \nabla f(S_{j^\*}) \odot \frac{S_{j^\*}}{S_n}.
\]
These supervise the **value head** and (optionally) an auxiliary **delta head**.

### Stock-driver \( \alpha \) (Heston)

Discounted continuation increments satisfy a martingale representation. Regress the **per-path** increment
\[
F_n \;\approx\; \frac{e^{-rt_{n+1}}c_{n+1} - e^{-rt_n}c_n}{\sqrt{\Delta t}}
\]
on standardized stock shocks
\[
X_n \;=\; \frac{\Delta S_n/S_n - r\Delta t}{\sqrt{v_n}\sqrt{\Delta t}}.
\]
Batch ridge OLS (small \(\ell_2\)) yields an **alpha label** \( \alpha_c \in \mathbb{R}^d \).  
The **AlphaGRU** outputs \( \alpha_y \) path-wise and is trained to match the **whitened** target
\[
Z^{(S)} = \alpha L^\top,
\]
where \(L\) is the Cholesky of the **stock correlation**. This matches the BSDE’s \(Z\) convention under correlated drivers.

### Loss (per time step, averaged)

\[
\mathcal L \;=\; \underbrace{\mathbb E|c_n - y_n|^2}_{\text{value MSE}}
\;+\; \lambda \,\underbrace{\mathbb E\| Z^{(S)}_{\text{label}} - Z^{(S)}_{\text{net}} \|^2}_{\text{driver MSE}}
\;+\; \gamma \,\underbrace{\mathbb E\|\Delta_{\text{aux}} - \nabla c_n\|^2}_{\text{optional}}.
\]

- \( \lambda \) balances **units** (price vs driver). The raw BSDE scaling can under-weight the driver; here it is tunable.  
- \( \gamma \) is a small auxiliary weight for delta stabilization.

### Hedging (what you actually trade)

- **Exercise region**: \(f(S_n)\ge y_n \Rightarrow\) hedge with the payoff gradient (arith. basket put: \(-w\)).  
- **Continuation region**: map the learned driver to **shares**:  
  1) compute \( \tilde h = \alpha \oslash (S \circ \sqrt{v}) \) (component-wise), then  
  2) if using whitened targets, solve \(L^\top h = \tilde h\) for \(h\).

---

## Contact

Questions and collaborations welcome—please open an issue.
