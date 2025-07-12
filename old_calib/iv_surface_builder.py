"""
iv_surface_builder.py – Build & Visualise an IV Surface
=====================================================
Changes
-------
* Added **`plot_iv_surface_3d`** – a 3‑D scatter (strike × time‑to‑expiry × IV).
* Demo now shows both the 2‑D smile overlay and the 3‑D surface.

Run:
```
python iv_surface_builder.py
```
and two windows will pop up.

Dependencies: `yfinance pandas numpy scipy matplotlib` (no extra 3‑D libs – we
use Matplotlib’s `mplot3d`).
"""
from __future__ import annotations

import math
from math import log, sqrt, exp
from typing import Tuple, List

import matplotlib.pyplot as plt
# mpl_toolkits.mplot3d is imported via projection="3d" parameter
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq
from scipy.stats import norm as scipy_norm

__all__ = [
    "build_iv_surface",
    "plot_iv_surface",
    "plot_iv_surface_3d",
]

# ---------------------------------------------------------------------------
#  Black‑Scholes inverse (price → implied vol)
# ---------------------------------------------------------------------------

def _implied_vol(mid: float, S: float, K: float, T: float, r: float, otype: str) -> float:
    # Ensure otype is properly converted to string and normalized
    otype_str = str(otype).lower().strip()
    
    intrinsic = max(0.0, S - K) if otype_str == "call" else max(0.0, K - S)
    if mid <= intrinsic + 1e-8:
        return np.nan

    def _bs(vol):
        if vol < 1e-10:
            return intrinsic
        d1 = (log(S / K) + (r + 0.5 * vol**2) * T) / (vol * sqrt(T))
        d2 = d1 - vol * sqrt(T)
        if otype_str == "call":
            return S * scipy_norm.cdf(d1) - K * math.exp(-r * T) * scipy_norm.cdf(d2)
        return K * math.exp(-r * T) * scipy_norm.cdf(-d2) - S * scipy_norm.cdf(-d1)

    try:
        return brentq(lambda v: _bs(v) - mid, 1e-6, 5.0, xtol=1e-8)
    except (ValueError, RuntimeError):
        return np.nan

# ---------------------------------------------------------------------------
#  Helper: tidy column names
# ---------------------------------------------------------------------------

def _tidy_cols(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {c: c.title().replace(" ", "") for c in df.columns}
    return df.rename(columns=mapping)

# ---------------------------------------------------------------------------
#  IV‑surface builder
# ---------------------------------------------------------------------------

def build_iv_surface(
    raw_df: pd.DataFrame,
    *,
    spot_price: float,
    risk_free_rate: float = 0.03,
    min_volume: int = 10,
    min_open_interest: int = 10,
    moneyness_bounds: Tuple[float, float] = (0.5, 1.5),
    max_bid_ask_spread: float = 0.5,
    maturities_days: Tuple[int, int] = (7, 730),
) -> Tuple[pd.DataFrame, float]:
    """Clean option quotes and compute implied vols."""

    df = _tidy_cols(raw_df)

    required = {"Strike", "Bid", "Ask", "Volume", "Optiontype", "Expiration"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing}")
    df.rename(columns={"Optiontype": "OptionType"}, inplace=True)

    oi_cols = [c for c in df.columns if c.lower() == "openinterest"]
    df["OpenInterest"] = df[oi_cols[0]] if oi_cols else 0

    df = df[(df.Volume >= min_volume) & (df.OpenInterest >= min_open_interest)]

    df["Mid"] = (df.Bid + df.Ask) / 2.0
    spread = (df.Ask - df.Bid) / df.Mid.replace(0, np.nan)
    df = df[spread <= max_bid_ask_spread]

    df = df[(df.Strike / spot_price >= moneyness_bounds[0]) & (df.Strike / spot_price <= moneyness_bounds[1])]

    today = pd.Timestamp.utcnow().tz_localize(None)
    expiry = pd.to_datetime(df.Expiration).dt.tz_localize(None)
    df["DaysToExpiry"] = (expiry - today).dt.days
    df = df[(df.DaysToExpiry >= maturities_days[0]) & (df.DaysToExpiry <= maturities_days[1])]

    ivs: List[float] = []
    for row in df.itertuples(index=False):
        time_to_expiry = row.DaysToExpiry / 365.0
        ivs.append(_implied_vol(row.Mid, spot_price, row.Strike, time_to_expiry, risk_free_rate, str(row.OptionType).lower()))
    df["ImpliedVolatility"] = ivs
    df.dropna(subset=["ImpliedVolatility"], inplace=True)

    return df[["Strike", "DaysToExpiry", "ImpliedVolatility", "OptionType"]].reset_index(drop=True), spot_price

# ---------------------------------------------------------------------------
#  2‑D scatter
# ---------------------------------------------------------------------------

def plot_iv_surface(df: pd.DataFrame, *, spot: float):
    cmap = plt.get_cmap("viridis")
    maturities = sorted(df.DaysToExpiry.unique())
    _, ax = plt.subplots()
    for i, T in enumerate(maturities):
        sub = df[df.DaysToExpiry == T]
        ax.scatter(sub.Strike, sub.ImpliedVolatility, color=cmap(i / len(maturities)), label=f"{T}d")
    ax.axvline(spot, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied volatility")
    ax.set_title("IV surface snapshot (2‑D)")
    ax.legend(title="Days to expiry")
    plt.show()

# ---------------------------------------------------------------------------
#  3‑D scatter
# ---------------------------------------------------------------------------

def plot_iv_surface_3d(df: pd.DataFrame, *, spot: float):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    xs = df.Strike.values
    ys = df.DaysToExpiry.values
    zs = df.ImpliedVolatility.values

    sc = ax.scatter(xs, ys, zs, c=zs, cmap="viridis", depthshade=True)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Days to expiry")
    ax.set_zlabel("Implied vol")
    ax.set_title(f"IV surface (3‑D scatter) - Spot: ${spot:.2f}")
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label="IV")
    plt.show()

# ---------------------------------------------------------------------------
#  NVDA demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    tkr = yf.Ticker("NVDA")
    spot = float(tkr.history(period="1d")["Close"].iloc[-1])

    raws = []
    for exp in tkr.options[:2]:
        chain = tkr.option_chain(exp)
        for side, d in zip(["call", "put"], [chain.calls, chain.puts]):
            d = _tidy_cols(d)
            d["OptionType"] = side
            d["Expiration"] = exp
            raws.append(d)
    chain_df = pd.concat(raws, ignore_index=True)

    surface, _ = build_iv_surface(chain_df, spot_price=spot,
                                  min_volume=0, min_open_interest=0,
                                  moneyness_bounds=(0.5, 2.0),
                                  max_bid_ask_spread=1.0)
    print(f"Surface built – {len(surface)} quotes, maturities: {sorted(surface.DaysToExpiry.unique())}")
    
    if len(surface) == 0:
        print("Warning: No data points survived filtering. Adjusting filters...")
        surface, _ = build_iv_surface(chain_df, spot_price=spot,
                                      min_volume=0, min_open_interest=0,
                                      moneyness_bounds=(0.1, 5.0),
                                      max_bid_ask_spread=2.0,
                                      maturities_days=(1, 1000))
        print(f"After relaxing filters: {len(surface)} quotes")

    if len(surface) > 0:
        plot_iv_surface(surface, spot=spot)
        plot_iv_surface_3d(surface, spot=spot)
    else:
        print("No valid option data available for plotting")
