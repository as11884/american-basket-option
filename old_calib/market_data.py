"""
option_data_pipeline.py – Fetch, Clean, and Visualise Equity Option Data
=======================================================================
This module keeps **data acquisition & filtering** separate from calibration or
model‑specific code.  It wraps *yfinance* to

* download adjusted equity prices;
* pull option chains at expiries nearest to target days (30/60/90 by default);
* filter/liquidity‑clean the option quotes;
* build a tidy implied‑volatility surface DataFrame ready for any model;
* compute realised volatility from historical closes;
* quick 2‑D and 3‑D IV plots.

Public API
----------
```python
fetch_stock_data(ticker, period="6mo", interval="1d") -> pd.DataFrame
fetch_option_chains(ticker, targets=[30,60,90])         -> dict[int, pd.DataFrame]
build_iv_surface(stock_df, chains_dict, ...)            -> pd.DataFrame
calc_realized_vol(stock_df, window=63)                  -> float
prepare_dataset(ticker)                                 -> dict
plot_iv_2d(df, spot)                                    -> None
plot_iv_3d(df, spot)                                    -> None
```

Run the file directly for an NVDA demo:
```
python option_data_pipeline.py  # pops up 2‑D & 3‑D plots
```
Dependencies: `yfinance pandas numpy scipy matplotlib`.
"""
from __future__ import annotations

import datetime as dt
from math import log, sqrt, exp
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq
from scipy.stats import norm

# ---------------------------------------------------------------------------
#  Fetch equity prices
# ---------------------------------------------------------------------------

def fetch_stock_data(ticker: str, *, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Adjusted historical prices via yfinance."""
    return yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)

# ---------------------------------------------------------------------------
#  Fetch option chains near target DTEs
# ---------------------------------------------------------------------------

def fetch_option_chains(ticker: str, *, targets: List[int] = [30, 60, 90]) -> Dict[int, pd.DataFrame]:
    """Return dict {days_to_expiry: DataFrame(merged calls+puts)}."""

    tk = yf.Ticker(ticker)
    today = dt.date.today()
    expirations = tk.options
    if not expirations:
        raise ValueError(f"No option expirations for {ticker}")

    output: Dict[int, pd.DataFrame] = {}
    for t in targets:
        target_date = today + dt.timedelta(days=t)
        nearest = min(expirations, key=lambda x: abs(dt.datetime.strptime(x, "%Y-%m-%d").date() - target_date))
        try:
            chain = tk.option_chain(nearest)
            df = pd.concat([chain.calls.assign(OptionType="call"), chain.puts.assign(OptionType="put")])
            df["Expiration"] = nearest
            dte = (dt.datetime.strptime(nearest, "%Y-%m-%d").date() - today).days
            output[dte] = df
        except Exception as e:
            print(f"Warning: skip expiry {nearest} – {e}")
    if not output:
        raise ValueError("Failed to fetch any option chains")
    return output

# ---------------------------------------------------------------------------
#  Black‑Scholes IV helper
# ---------------------------------------------------------------------------

def _bs_price(S: float, K: float, T: float, r: float, vol: float, cp: str) -> float:
    d1 = (log(S/K) + (r + 0.5*vol**2)*T) / (vol*sqrt(T))
    d2 = d1 - vol*sqrt(T)
    if cp == "call":
        return S*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
    return K*exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def _implied_vol(mid: float, S: float, K: float, T: float, r: float, cp: str) -> float:
    intrinsic = max(0, S-K) if cp=="call" else max(0, K-S)
    if mid <= intrinsic + 1e-8:
        return np.nan
    try:
        return brentq(lambda v: _bs_price(S,K,T,r,v,cp)-mid, 1e-6, 5.0, xtol=1e-8)
    except (ValueError, RuntimeError):
        return np.nan

# ---------------------------------------------------------------------------
#  Build IV surface DataFrame
# ---------------------------------------------------------------------------

def build_iv_surface(
    stock_df: pd.DataFrame,
    chains: Dict[int, pd.DataFrame],
    *,
    risk_free_rate: float = 0.03,
    min_volume: int = 10,
    min_oi: int = 10,
    max_spread_frac: float = 0.5,
    moneyness: Tuple[float, float] = (0.7, 1.3),
) -> pd.DataFrame:
    """Clean -> IV -> **average call & put IV** per (DTE, Strike).

    Returns one row per strike/maturity with the mean of call/put implied vols.
    """
    spot = stock_df["Close"].iloc[-1]
    rows: List[dict] = []

    # Loop over maturities
    for dte, df in chains.items():
        df = df.copy()
        df["Mid"] = (df["bid"] + df["ask"]) / 2.0
        df["SpreadFrac"] = (df["ask"] - df["bid"]) / df["Mid"].replace(0, np.nan)
        df = df[(df["volume"] >= min_volume) & (df["openInterest"] >= min_oi) & (df["SpreadFrac"] <= max_spread_frac)]
        df = df[(df["strike"] / spot >= moneyness[0]) & (df["strike"] / spot <= moneyness[1])]

        # Compute IV for each remaining quote
        iv_list = []
        for _, r in df.iterrows():
            iv = _implied_vol(r["Mid"], spot, r["strike"], dte / 365, risk_free_rate, r["OptionType"])
            if not np.isnan(iv):
                iv_list.append({"DaysToExpiry": dte, "Strike": r["strike"], "ImpliedVolatility": iv})

        if not iv_list:
            continue

        iv_df = pd.DataFrame(iv_list)
        # Average call & put IVs for identical (Strike, DTE)
        iv_avg = iv_df.groupby(["DaysToExpiry", "Strike"], as_index=False)["ImpliedVolatility"].mean()
        rows.append(iv_avg)

    if not rows:
        return pd.DataFrame()

    surface = pd.concat(rows, ignore_index=True)
    surface["Moneyness"] = surface["Strike"] / spot
    return surface

# ---------------------------------------------------------------------------
#  Realised volatility
# ---------------------------------------------------------------------------

def calc_realized_vol(stock_df: pd.DataFrame, *, window: int = 63) -> float:
    log_ret = np.log(stock_df["Close"] / stock_df["Close"].shift(1)).dropna()
    return log_ret[-window:].std() * np.sqrt(252)

# ---------------------------------------------------------------------------
#  Prepare full dataset
# ---------------------------------------------------------------------------

def prepare_dataset(ticker: str) -> dict:
    stock = fetch_stock_data(ticker, period="6mo")
    realized = calc_realized_vol(stock)
    chains = fetch_option_chains(ticker)
    iv_surface = build_iv_surface(stock, chains)
    return {"stock": stock, "realized_vol": realized, "iv_surface": iv_surface}

# ---------------------------------------------------------------------------
#  Plot helpers
# ---------------------------------------------------------------------------

def plot_iv_2d(df: pd.DataFrame, *, spot: float):
    cmap = plt.get_cmap("viridis")
    for i,T in enumerate(sorted(df.DaysToExpiry.unique())):
        sub = df[df.DaysToExpiry==T]
        plt.scatter(sub.Strike, sub.ImpliedVolatility, color=cmap(i/len(df.DaysToExpiry.unique())), label=f"{T}d")
    plt.axvline(spot, color="k", ls="--")
    plt.xlabel("Strike"); plt.ylabel("IV"); plt.title("IV Smile")
    plt.legend(); plt.show()

def plot_iv_3d(df: pd.DataFrame, *, spot: float):
    fig = plt.figure(); ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(df.Strike, df.DaysToExpiry, df.ImpliedVolatility, c=df.ImpliedVolatility, cmap="viridis")
    ax.set_xlabel("Strike"); ax.set_ylabel("DaysToExpiry"); ax.set_zlabel("IV")
    fig.colorbar(sc, shrink=0.6, pad=0.1, label="IV"); plt.show()

# ---------------------------------------------------------------------------
#  Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = prepare_dataset("NVDA")
    print("Realized vol (63‑day):", round(data["realized_vol"],4))
    plot_iv_2d(data["iv_surface"], spot=data["stock"]["Close"].iloc[-1])
    plot_iv_3d(data["iv_surface"], spot=data["stock"]["Close"].iloc[-1])
