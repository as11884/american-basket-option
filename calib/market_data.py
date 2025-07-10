"""
Market Data Fetching and Processing Module

This module provides utilities for fetching and processing market data
needed for Heston model calibration, including:
- Historical stock price data
- Option chain data with implied volatilities
- Realized volatility calculations
- Data preparation for calibration

The module uses yfinance for data fetching and includes comprehensive
error handling and data validation.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional


def fetch_historical_stock_data(ticker_symbol: str, historical_period: str = "3mo", 
                              data_interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical stock price data for a given ticker.
    
    This function retrieves adjusted closing prices, volumes, and other
    market data from Yahoo Finance for the specified time period.
    
    Parameters
    ----------
    ticker_symbol : str
        Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
    historical_period : str, default="3mo"
        Time period for historical data ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
    data_interval : str, default="1d"
        Data frequency ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing historical stock data with columns:
        - Open, High, Low, Close: Price data
        - Volume: Trading volume
        - Dividends, Stock Splits: Corporate actions
        
    Raises
    ------
    ValueError
        If the ticker symbol is invalid or no data is available
    """
    print(f"Fetching historical data for {ticker_symbol}...")
    
    try:
        # Create ticker object and fetch data
        ticker_object = yf.Ticker(ticker_symbol)
        historical_data = ticker_object.history(
            period=historical_period, 
            interval=data_interval, 
            auto_adjust=True  # Automatically adjust for splits and dividends
        )
        
        # Validate that we received data
        if historical_data.empty:
            raise ValueError(f"No historical data found for ticker '{ticker_symbol}'")
        
        print(f"Successfully fetched {len(historical_data)} data points for {ticker_symbol}")
        return historical_data
        
    except Exception as e:
        raise ValueError(f"Failed to fetch historical data for '{ticker_symbol}': {str(e)}")


def fetch_option_chain_data(ticker_symbol: str, 
                           target_expiry_days: List[int] = [30, 60, 90]) -> Dict[int, Dict[str, pd.DataFrame]]:
    """
    Fetch option chain data for specified expiration dates.
    
    This function retrieves call and put option data for expiration dates
    closest to the target days. It includes comprehensive error handling
    for missing or invalid option data.
    
    Parameters
    ----------
    ticker_symbol : str
        Stock ticker symbol
    target_expiry_days : List[int], default=[30, 60, 90]
        Target days to expiration for option chains
        
    Returns
    -------
    Dict[int, Dict[str, pd.DataFrame]]
        Dictionary with structure:
        {
            days_to_expiry: {
                'calls': DataFrame with call options,
                'puts': DataFrame with put options
            }
        }
        
    Raises
    ------
    ValueError
        If no option data is available for the ticker
    """
    print(f"Fetching option chain data for {ticker_symbol}...")
    
    try:
        # Create ticker object and get available expiration dates
        ticker_object = yf.Ticker(ticker_symbol)
        available_expirations = ticker_object.options
        
        if not available_expirations:
            raise ValueError(f"No option expiration dates found for ticker '{ticker_symbol}'")
        
        print(f"Found {len(available_expirations)} available expiration dates")
        
    except Exception as e:
        raise ValueError(f"Failed to fetch option expiration dates for '{ticker_symbol}': {str(e)}")
    
    # Current date for calculating days to expiry
    current_date = datetime.now().date()
    
    # Find target expiration dates
    target_expiration_dates = []
    for target_days in target_expiry_days:
        target_date = current_date + timedelta(days=target_days)
        target_expiration_dates.append(target_date)
    
    # Dictionary to store option data
    option_chain_data = {}
    
    # Fetch option chains for each target expiration
    for target_date in target_expiration_dates:
        try:
            # Find the closest available expiration date
            closest_expiration = min(
                available_expirations,
                key=lambda exp_date: abs(datetime.strptime(exp_date, "%Y-%m-%d").date() - target_date)
            )
            
            # Calculate actual days to expiry
            actual_expiry_date = datetime.strptime(closest_expiration, "%Y-%m-%d").date()
            actual_days_to_expiry = (actual_expiry_date - current_date).days
            
            # Fetch option chain for this expiration
            option_chain = ticker_object.option_chain(closest_expiration)
            
            # Store the option data
            option_chain_data[actual_days_to_expiry] = {
                'calls': option_chain.calls,
                'puts': option_chain.puts,
                'expiration_date': closest_expiration
            }
            
            print(f"Fetched option chain for {actual_days_to_expiry} days to expiry "
                  f"(expiration: {closest_expiration})")
            
        except Exception as e:
            print(f"Warning: Could not fetch option chain for expiration {closest_expiration}: {str(e)}")
            continue
    
    if not option_chain_data:
        raise ValueError(f"No valid option chains retrieved for ticker '{ticker_symbol}'")
    
    print(f"Successfully fetched option data for {len(option_chain_data)} expiration dates")
    return option_chain_data


def calculate_implied_volatility_surface(historical_stock_data: pd.DataFrame, 
                                       option_chain_data: Dict[int, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Calculate implied volatility surface from option chain data.
    
    This function processes option chain data to create a structured dataset
    suitable for model calibration, including moneyness calculations and
    data filtering.
    
    Parameters
    ----------
    historical_stock_data : pd.DataFrame
        Historical stock price data
    option_chain_data : Dict[int, Dict[str, pd.DataFrame]]
        Option chain data from fetch_option_chain_data
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - DaysToExpiry: Days until option expiration
        - Strike: Option strike price
        - Moneyness: Strike/Spot ratio
        - ImpliedVolatility: Market implied volatility
        - OptionType: 'call' or 'put'
        
    Raises
    ------
    ValueError
        If no valid implied volatility data is found
    """
    print("Calculating implied volatility surface...")
    
    # Get current spot price (most recent closing price)
    current_spot_price = historical_stock_data['Close'].iloc[-1]
    print(f"Current spot price: ${current_spot_price:.2f}")
    
    # List to store processed option data
    implied_volatility_data = []
    
    # Process each expiration date
    for days_to_expiry, option_data in option_chain_data.items():
        calls_data = option_data['calls']
        puts_data = option_data['puts']
        
        # Process call options
        valid_calls = calls_data.dropna(subset=['impliedVolatility'])
        for _, option_row in valid_calls.iterrows():
            # Filter out options with zero or negative implied volatility
            if option_row['impliedVolatility'] > 0:
                implied_volatility_data.append({
                    'DaysToExpiry': days_to_expiry,
                    'Strike': option_row['strike'],
                    'Moneyness': option_row['strike'] / current_spot_price,
                    'ImpliedVolatility': option_row['impliedVolatility'],
                    'OptionType': 'call',
                    'Volume': option_row.get('volume', 0),
                    'OpenInterest': option_row.get('openInterest', 0)
                })
        
        # Process put options
        valid_puts = puts_data.dropna(subset=['impliedVolatility'])
        for _, option_row in valid_puts.iterrows():
            # Filter out options with zero or negative implied volatility
            if option_row['impliedVolatility'] > 0:
                implied_volatility_data.append({
                    'DaysToExpiry': days_to_expiry,
                    'Strike': option_row['strike'],
                    'Moneyness': option_row['strike'] / current_spot_price,
                    'ImpliedVolatility': option_row['impliedVolatility'],
                    'OptionType': 'put',
                    'Volume': option_row.get('volume', 0),
                    'OpenInterest': option_row.get('openInterest', 0)
                })
        
        print(f"Processed {len(valid_calls)} calls and {len(valid_puts)} puts for {days_to_expiry} days to expiry")
    
    # Create DataFrame from processed data
    if not implied_volatility_data:
        raise ValueError("No valid implied volatility data found in option chains")
    
    implied_vol_surface = pd.DataFrame(implied_volatility_data)
    
    # Sort by expiry and moneyness for better organization
    implied_vol_surface = implied_vol_surface.sort_values(['DaysToExpiry', 'Moneyness'])
    
    print(f"Created implied volatility surface with {len(implied_vol_surface)} data points")
    print(f"Moneyness range: {implied_vol_surface['Moneyness'].min():.3f} to {implied_vol_surface['Moneyness'].max():.3f}")
    print(f"Implied volatility range: {implied_vol_surface['ImpliedVolatility'].min():.3f} to {implied_vol_surface['ImpliedVolatility'].max():.3f}")
    
    return implied_vol_surface


def calculate_realized_volatility(historical_stock_data: pd.DataFrame, 
                                lookback_window_days: int = 63) -> float:
    """
    Calculate annualized realized volatility from historical stock returns.
    
    This function computes the historical volatility using log returns
    over a specified lookback window. The volatility is annualized
    assuming 252 trading days per year.
    
    Parameters
    ----------
    historical_stock_data : pd.DataFrame
        Historical stock price data with 'Close' column
    lookback_window_days : int, default=63
        Number of trading days to use for volatility calculation
        (63 days ≈ 3 months, 252 days ≈ 1 year)
        
    Returns
    -------
    float
        Annualized realized volatility
        
    Raises
    ------
    ValueError
        If insufficient data is available for calculation
    """
    print(f"Calculating realized volatility using {lookback_window_days} days lookback...")
    
    # Calculate log returns
    log_returns = np.log(historical_stock_data['Close'] / historical_stock_data['Close'].shift(1))
    
    # Remove NaN values
    log_returns = log_returns.dropna()
    
    if len(log_returns) < lookback_window_days:
        print(f"Warning: Only {len(log_returns)} data points available, "
              f"less than requested {lookback_window_days} days")
        lookback_window = log_returns  # Use all available data
    else:
        # Use the most recent returns within the lookback window
        lookback_window = log_returns.iloc[-lookback_window_days:]
    
    if len(lookback_window) < 10:
        raise ValueError("Insufficient data for reliable volatility calculation (need at least 10 observations)")
    
    # Calculate daily volatility (standard deviation of log returns)
    daily_volatility = lookback_window.std()
    
    # Annualize volatility (assuming 252 trading days per year)
    annualized_volatility = daily_volatility * np.sqrt(252)
    
    print(f"Realized volatility calculated from {len(lookback_window)} daily returns: {annualized_volatility:.4f}")
    
    return annualized_volatility


def prepare_calibration_data(ticker_symbol: str, 
                           historical_period: str = '3mo',
                           target_expiry_days: List[int] = [30, 60, 90]) -> Dict[str, Any]:
    """
    Comprehensive data preparation for Heston model calibration.
    
    This function fetches all necessary market data and prepares it
    for calibration, including error handling and data validation.
    
    Parameters
    ----------
    ticker_symbol : str
        Stock ticker symbol
    historical_period : str, default='3mo'
        Period for historical data
    target_expiry_days : List[int], default=[30, 60, 90]
        Target days to expiration for option data
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'stock_data': Historical stock price data
        - 'realized_vol': Annualized realized volatility
        - 'iv_surface': Implied volatility surface DataFrame
        - 'spot_price': Current spot price
        - 'option_data': Raw option chain data
        
    Raises
    ------
    ValueError
        If data fetching fails for any component
    """
    print(f"\n=== Preparing Calibration Data for {ticker_symbol} ===")
    
    try:
        # Step 1: Fetch historical stock data
        print("\n1. Fetching historical stock data...")
        historical_data = fetch_historical_stock_data(ticker_symbol, historical_period)
        current_spot_price = historical_data['Close'].iloc[-1]
        
        # Step 2: Calculate realized volatility
        print("\n2. Calculating realized volatility...")
        realized_volatility = calculate_realized_volatility(historical_data, lookback_window_days=63)
        
        # Step 3: Fetch option chain data
        print("\n3. Fetching option chain data...")
        option_chains = fetch_option_chain_data(ticker_symbol, target_expiry_days)
        
        # Step 4: Calculate implied volatility surface
        print("\n4. Processing implied volatility surface...")
        iv_surface = calculate_implied_volatility_surface(historical_data, option_chains)
        
        # Prepare summary statistics
        print(f"\n=== Data Summary for {ticker_symbol} ===")
        print(f"Current spot price: ${current_spot_price:.2f}")
        print(f"Realized volatility: {realized_volatility:.4f} ({realized_volatility*100:.2f}%)")
        print(f"Option data points: {len(iv_surface)}")
        print(f"Expiration dates: {sorted(iv_surface['DaysToExpiry'].unique())} days")
        
        # Return comprehensive data package
        return {
            'stock_data': historical_data,
            'realized_vol': realized_volatility,
            'iv_surface': iv_surface,
            'spot_price': current_spot_price,
            'option_data': option_chains
        }
        
    except Exception as e:
        print(f"\nError preparing calibration data for {ticker_symbol}: {str(e)}")
        raise


def filter_options_for_calibration(iv_surface: pd.DataFrame,
                                 min_moneyness: float = 0.8,
                                 max_moneyness: float = 1.2,
                                 min_volume: int = 10,
                                 max_days_to_expiry: int = 180) -> pd.DataFrame:
    """
    Filter option data for calibration to improve robustness.
    
    This function applies various filters to remove illiquid or
    problematic options that might hurt calibration quality.
    
    Parameters
    ----------
    iv_surface : pd.DataFrame
        Implied volatility surface data
    min_moneyness : float, default=0.8
        Minimum moneyness (strike/spot) to include
    max_moneyness : float, default=1.2
        Maximum moneyness (strike/spot) to include
    min_volume : int, default=10
        Minimum trading volume to include
    max_days_to_expiry : int, default=180
        Maximum days to expiry to include
        
    Returns
    -------
    pd.DataFrame
        Filtered implied volatility surface
    """
    print(f"Filtering options for calibration...")
    print(f"Original dataset: {len(iv_surface)} options")
    
    # Apply filters
    filtered_data = iv_surface[
        (iv_surface['Moneyness'] >= min_moneyness) &
        (iv_surface['Moneyness'] <= max_moneyness) &
        (iv_surface['Volume'] >= min_volume) &
        (iv_surface['DaysToExpiry'] <= max_days_to_expiry) &
        (iv_surface['ImpliedVolatility'] > 0.05) &  # Remove extremely low IVs
        (iv_surface['ImpliedVolatility'] < 2.0)     # Remove extremely high IVs
    ].copy()
    
    print(f"Filtered dataset: {len(filtered_data)} options")
    print(f"Removed {len(iv_surface) - len(filtered_data)} options due to filters")
    
    return filtered_data


if __name__ == "__main__":
    """
    Example usage of the market data module.
    """
    # Test with Apple stock
    test_ticker = 'AAPL'
    
    try:
        # Prepare calibration data
        calibration_data = prepare_calibration_data(test_ticker)
        
        # Display sample of the data
        print(f"\n=== Sample Data for {test_ticker} ===")
        print("\nHistorical Stock Data (last 5 days):")
        print(calibration_data['stock_data'][['Close', 'Volume']].tail())
        
        print("\nImplied Volatility Surface (sample):")
        print(calibration_data['iv_surface'].head(10))
        
        # Apply filters
        filtered_iv = filter_options_for_calibration(calibration_data['iv_surface'])
        print(f"\nFiltered IV surface contains {len(filtered_iv)} options")
        
    except Exception as e:
        print(f"Error in example: {str(e)}")