"""
NVDA Heston Calibration Demo
============================
Fetches market data with **option_data_pipeline**, cleans & averages IVs,
then calibrates Cui‑style Heston parameters and plots market vs model smiles.

Run:
```bash
python nvidia_heston_calibration.py
```
Dependencies already covered by existing modules (`yfinance`, `pandas`,
`numpy`, `scipy`, `matplotlib`).
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np

from market_data import prepare_dataset, plot_iv_2d, plot_iv_3d
from heston_calibrator import HestonParams, IVHestonCalibrator


warnings.filterwarnings("ignore", category=RuntimeWarning)

def main():
    # 1. data
    data = prepare_dataset("NVDA")
    iv_surface = data["iv_surface"]
    spot = data["stock"]["Close"].iloc[-1]
    print(f"Clean surface: {len(iv_surface)} quotes across {iv_surface.DaysToExpiry.nunique()} maturities")

    # 2. initial guess – use realised vol as rough v0, theta
    realised = data["realized_vol"]
    init_params= HestonParams(v0=realised**2, kappa=1.5, theta=realised**2, sigma=0.6, rho=0.3)

    # 3. calibrate
    calibrator = IVHestonCalibrator(r=0.01)  # your risk-free rate
    fitted = calibrator.calibrate(spot, iv_surface, init_params)
    print(f"Calibrated params: {fitted}")
    # # 4. plot market vs model IV for each maturity
    # fig, ax = plt.subplots()
    # cmap = plt.get_cmap("viridis")
    # pricer = calibrator._pricer  # reuse pricing engine
    # for i, T in enumerate(sorted(iv_surface.DaysToExpiry.unique())):
    #     sub = iv_surface[iv_surface.DaysToExpiry == T]
    #     strikes = sub.Strike.values
    #     market_iv = sub.ImpliedVolatility.values
    #     model_iv = []
    #     for K in strikes:
    #         price = pricer.price(spot, K, T/365, params, "call")
    #         model_iv.append(calibrator._iv(price, spot, K, T/365))
    #     ax.scatter(strikes, market_iv, color=cmap(i/5), marker="o", label=f"{T}d market")
    #     ax.plot(strikes, model_iv, color=cmap(i/5), lw=1.2, label=f"{T}d model")
    # ax.set_xlabel("Strike"); ax.set_ylabel("IV"); ax.set_title("NVDA – Market vs Heston fit")
    # ax.legend(); plt.show()

    # # optional 3‑D market surface
    # plot_iv_3d(iv_surface, spot=spot)

if __name__ == "__main__":
    main()
