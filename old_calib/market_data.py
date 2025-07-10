import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def fetch_stock_data(ticker, period="3mo", interval="1d"):
    """Fetch adjusted historical prices."""
    obj = yf.Ticker(ticker)
    return obj.history(period=period, interval=interval, auto_adjust=True)


def fetch_option_data(ticker, expiry_days=[30, 60, 90]):
    """Fetch calls/puts at nearest expiries to days in `expiry_days`, with error handling."""
    obj = yf.Ticker(ticker)
    today = datetime.now().date()
    try:
        expirations = obj.options
    except Exception as e:
        raise ValueError(f"Failed to fetch options for ticker '{ticker}': {e}")
    if not expirations:
        raise ValueError(f"No option expiration data found for ticker '{ticker}'.")

    targets = [today + timedelta(days=d) for d in expiry_days]
    data = {}
    for target in targets:
        # find nearest expiry and attempt to fetch chain
        try:
            closest = min(expirations,
                          key=lambda x: abs(datetime.strptime(x, "%Y-%m-%d").date() - target))
            days = (datetime.strptime(closest, "%Y-%m-%d").date() - today).days
            chain = obj.option_chain(closest)
            data[days] = { 'calls': chain.calls, 'puts': chain.puts }
        except Exception as e:
            # skip problematic expiry and warn
            print(f"Warning: could not fetch option chain for expiry {closest}: {e}")

    if not data:
        raise ValueError(f"No valid option chains retrieved for ticker '{ticker}'.")
    return data


def calculate_implied_vol_surface(stock_df, option_data):
    """Return DataFrame of DaysToExpiry, Strike, Moneyness, ImpliedVolatility, OptionType."""
    spot = stock_df['Close'].iloc[-1]
    rows = []
    for days, d in option_data.items():
        for kind, df in d.items():
            df = df.dropna(subset=['impliedVolatility'])
            for _, r in df.iterrows():
                rows.append({
                    'DaysToExpiry': days,
                    'Strike': r['strike'],
                    'Moneyness': r['strike'] / spot,
                    'ImpliedVolatility': r['impliedVolatility'],
                    'OptionType': 'call' if kind=='calls' else 'put'
                })
    return pd.DataFrame(rows)


def calculate_realized_volatility(stock_df, window_days=63):
    """Annualized vol from past `window_days` log‐returns."""
    log_ret = np.log(stock_df['Close'] / stock_df['Close'].shift(1)).dropna()
    window = log_ret[-window_days:]
    return window.std() * np.sqrt(252)


def prepare_calibration_data(ticker, period='3mo', expiry_days=[30, 60, 90]):
    stock = fetch_stock_data(ticker, period)
    realized = calculate_realized_volatility(stock)
    opts = fetch_option_data(ticker, expiry_days)
    iv = calculate_implied_vol_surface(stock, opts)
    return {'stock_data': stock, 'realized_vol': realized, 'iv_surface': iv}