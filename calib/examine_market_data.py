"""
Examine the market data to understand the calibration issues
"""
import pandas as pd
import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm
import sys
sys.path.append('.')

from market_data import prepare_calibration_data, filter_options_for_calibration

def examine_market_data():
    """Examine the market data quality"""
    
    # Get market data
    data = prepare_calibration_data(
        ticker_symbol="NVDA",
        historical_period="3mo", 
        target_expiry_days=[30, 60, 90]
    )
    spot_price = data['spot_price']
    iv_surface = data['iv_surface']
    
    print(f"=== Market Data Analysis for NVDA ===")
    print(f"Current spot price: ${spot_price:.2f}")
    print(f"Total options: {len(iv_surface)}")
    
    # Filter options
    filtered_iv = filter_options_for_calibration(
        iv_surface,
        min_moneyness=0.8,
        max_moneyness=1.2,
        min_volume=10,
        max_days_to_expiry=90
    )
    
    print(f"Filtered options: {len(filtered_iv)}")
    
    # Calculate moneyness and other metrics
    filtered_iv['Moneyness'] = filtered_iv['Strike'] / spot_price
    filtered_iv['TimeToExpiry'] = filtered_iv['DaysToExpiry'] / 365.0
    
    # Calculate Black-Scholes prices
    def bs_price(row):
        S = spot_price
        K = row['Strike']
        T = row['TimeToExpiry']
        r = 0.03
        vol = row['ImpliedVolatility']
        option_type = row['OptionType']
        
        if T < 1e-6 or vol < 1e-6:
            return 0.0
            
        d1 = (log(S/K) + (r + 0.5*vol**2)*T) / (vol*sqrt(T))
        d2 = d1 - vol*sqrt(T)
        
        if option_type == 'call':
            return S*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
        else:
            return K*exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    filtered_iv['BS_Price'] = filtered_iv.apply(bs_price, axis=1)
    
    # Show summary statistics
    print(f"\n=== Summary Statistics ===")
    print(f"Moneyness range: {filtered_iv['Moneyness'].min():.3f} to {filtered_iv['Moneyness'].max():.3f}")
    print(f"IV range: {filtered_iv['ImpliedVolatility'].min():.3f} to {filtered_iv['ImpliedVolatility'].max():.3f}")
    print(f"BS Price range: ${filtered_iv['BS_Price'].min():.3f} to ${filtered_iv['BS_Price'].max():.3f}")
    print(f"Time to expiry range: {filtered_iv['TimeToExpiry'].min():.3f} to {filtered_iv['TimeToExpiry'].max():.3f}")
    
    # Show problematic options
    print(f"\n=== Options with Issues ===")
    problem_options = filtered_iv[
        (filtered_iv['BS_Price'] < 0.01) & (filtered_iv['ImpliedVolatility'] > 0.05)
    ]
    print(f"Options with price < $0.01 but IV > 5%: {len(problem_options)}")
    
    if len(problem_options) > 0:
        print(problem_options[['Strike', 'OptionType', 'Moneyness', 'ImpliedVolatility', 'BS_Price', 'Volume']].head(10))
    
    # Show reasonable options
    print(f"\n=== Reasonable Options ===")
    good_options = filtered_iv[
        (filtered_iv['BS_Price'] >= 0.01) & 
        (filtered_iv['Moneyness'] >= 0.9) & 
        (filtered_iv['Moneyness'] <= 1.1)
    ]
    print(f"Options with price >= $0.01 and moneyness 0.9-1.1: {len(good_options)}")
    
    if len(good_options) > 0:
        print(good_options[['Strike', 'OptionType', 'Moneyness', 'ImpliedVolatility', 'BS_Price', 'Volume']].head(10))
    
    # Suggest better filtering
    print(f"\n=== Recommendations ===")
    print("1. Filter out options with very low prices (< $0.01)")
    print("2. Focus on closer to ATM options (moneyness 0.9-1.1)")
    print("3. Use higher volume threshold")
    print("4. Check for stale or erroneous option data")

if __name__ == "__main__":
    examine_market_data()
