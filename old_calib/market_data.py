import datetime as dt
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# ---------------------------------------------------------------------------
#  Fetch equity prices
# ---------------------------------------------------------------------------

def fetch_stock_data(ticker: str, *, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Adjusted historical prices via yfinance."""
    return yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)

# ---------------------------------------------------------------------------
#  Fetch option chains near target DTEs
# ---------------------------------------------------------------------------

def fetch_option_chains(ticker: str, *, targets: List[int] = [30,60,90]) -> Dict[int, pd.DataFrame]:
    """Return dict {days_to_expiry: DataFrame(merged calls+puts)}."""
    tk = yf.Ticker(ticker)
    today = dt.date.today()
    expirations = tk.options
    if not expirations:
        raise ValueError(f"No option expirations for {ticker}")

    output: Dict[int, pd.DataFrame] = {}
    for t in targets:
        target_date = today + dt.timedelta(days=t)
        nearest = min(
            expirations,
            key=lambda x: abs(dt.datetime.strptime(x, "%Y-%m-%d").date() - target_date)
        )
        try:
            chain = tk.option_chain(nearest)
            df = pd.concat([
                chain.calls.assign(OptionType="call"),
                chain.puts.assign(OptionType="put")
            ])
            df["Expiration"] = nearest
            dte = (dt.datetime.strptime(nearest, "%Y-%m-%d").date() - today).days
            output[dte] = df
        except Exception:
            continue
    if not output:
        raise ValueError("Failed to fetch any option chains")
    return output

# ---------------------------------------------------------------------------
#  Build price surface DataFrame
# ---------------------------------------------------------------------------

def build_price_surface(
    stock_df: pd.DataFrame,
    chains: Dict[int, pd.DataFrame],
    *,
    min_volume: int = 10,
    min_oi: int = 10,
    max_spread_frac: float = 0.5,
    moneyness: tuple[float, float] = (0.9, 1.1),
) -> pd.DataFrame:
    """
    Clean -> mid-market prices for calls & puts per (DTE, Strike).

    Returns a DataFrame with columns [DaysToExpiry, Strike, OptionType, MarketPrice].
    """
    spot = stock_df["Close"].iloc[-1]
    rows: List[pd.DataFrame] = []

    for dte, df in chains.items():
        df = df.copy()
        df["Mid"] = (df["bid"] + df["ask"]) / 2
        df["SpreadFrac"] = (df["ask"] - df["bid"]) / df["Mid"].replace(0, np.nan)
        df = df.loc[
            (df["volume"] >= min_volume) &
            (df["openInterest"] >= min_oi) &
            (df["SpreadFrac"] <= max_spread_frac)
        ]
        df = df.loc[
            (df["strike"]/spot >= moneyness[0]) &
            (df["strike"]/spot <= moneyness[1])
        ]

        if df.empty:
            continue

        df = df.assign(
            DaysToExpiry=dte,
            Strike=df["strike"],
            OptionType=df["OptionType"],
            MarketPrice=df["Mid"],
        )
        rows.append(df[["DaysToExpiry", "Strike", "OptionType", "MarketPrice"]])

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

# ---------------------------------------------------------------------------
#  Realized volatility calculation (single number)
# ---------------------------------------------------------------------------

def calc_realized_vol(stock_df: pd.DataFrame) -> float:
    """
    Compute annualized realized volatility over the entire historical series.

    Returns the volatility as a single float.
    """
    returns = np.log(stock_df["Close"] / stock_df["Close"].shift(1)).dropna()
    vol = returns.std() * np.sqrt(252)
    return float(vol)

# ---------------------------------------------------------------------------
#  Prepare full dataset (prices + realized vol)
# ---------------------------------------------------------------------------

def prepare_dataset(ticker: str) -> dict:
    """
    Fetches stock data, builds a price surface for options,
    and computes realized volatility.

    Returns a dict with keys:
      - 'stock': DataFrame of historical stock prices
      - 'price_surface': DataFrame of mid-market option prices
      - 'realized_vol': float realized volatility
    """
    print(f"Fetching stock data for {ticker}...")
    stock = fetch_stock_data(ticker)

    print("Fetching option chains...")
    chains = fetch_option_chains(ticker)

    print("Building price surface...")
    price_surface = build_price_surface(stock, chains)

    print("Calculating realized volatility...")
    realized_vol = calc_realized_vol(stock)

    return {
        "stock": stock,
        "price_surface": price_surface,
        "realized_vol": realized_vol,
    }

def plot_price_surface(price_surface: pd.DataFrame, ticker: str = "") -> None:
    """
    Plot the option price surface as a 3D surface plot.
    
    Args:
        price_surface: DataFrame with columns [DaysToExpiry, Strike, OptionType, MarketPrice]
        ticker: Stock ticker for plot title
    """
    if price_surface.empty:
        print("Price surface is empty, cannot plot.")
        return
    
    # Create separate plots for calls and puts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), subplot_kw={'projection': '3d'})
    
    # Plot calls
    calls = price_surface[price_surface['OptionType'] == 'call']
    if not calls.empty:
        ax1.scatter(calls['DaysToExpiry'], calls['Strike'], calls['MarketPrice'], 
                   c='blue', alpha=0.6, s=20)
        ax1.set_xlabel('Days to Expiry')
        ax1.set_ylabel('Strike Price')
        ax1.set_zlabel('Option Price')
        ax1.set_title(f'{ticker} Call Options' if ticker else 'Call Options')
    
    # Plot puts
    puts = price_surface[price_surface['OptionType'] == 'put']
    if not puts.empty:
        ax2.scatter(puts['DaysToExpiry'], puts['Strike'], puts['MarketPrice'], 
                   c='red', alpha=0.6, s=20)
        ax2.set_xlabel('Days to Expiry')
        ax2.set_ylabel('Strike Price')
        ax2.set_zlabel('Option Price')
        ax2.set_title(f'{ticker} Put Options' if ticker else 'Put Options')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    TICKER = "NVDA"
    data = prepare_dataset(TICKER)
    plot_price_surface(data["price_surface"], TICKER)
