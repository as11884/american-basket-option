"""
Test to diagnose calibration issues by comparing model vs market prices
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from math import log, sqrt, exp
import sys
sys.path.append('.')

from heston_calibrator import HestonCalibrator
from calib_utils import HestonParams
from market_data import prepare_calibration_data

def black_scholes_price(spot, strike, time_to_expiry, risk_free_rate, vol, option_type='call'):
    """Black-Scholes option pricing for comparison"""
    d1 = (log(spot/strike) + (risk_free_rate + 0.5*vol**2)*time_to_expiry) / (vol*sqrt(time_to_expiry))
    d2 = d1 - vol*sqrt(time_to_expiry)
    
    if option_type == 'call':
        return spot*norm.cdf(d1) - strike*exp(-risk_free_rate*time_to_expiry)*norm.cdf(d2)
    else:
        return strike*exp(-risk_free_rate*time_to_expiry)*norm.cdf(-d2) - spot*norm.cdf(-d1)

def diagnose_calibration():
    """Diagnose why calibration is poor"""
    
    print("=== CALIBRATION DIAGNOSTIC ===")
    
    # Get market data
    data = prepare_calibration_data(
        ticker_symbol="NVDA",
        historical_period="3mo", 
        target_expiry_days=[30, 60, 90]
    )
    spot_price = data['spot_price']
    iv_surface = data['iv_surface']
    
    # Use all options
    option_data = iv_surface
    
    print(f"Spot price: ${spot_price:.2f}")
    print(f"Using {len(option_data)} options")
    
    # Test with reasonable Heston parameters
    test_params = HestonParams(
        v0=0.16,      # 40% vol
        kappa=2.0,    # Mean reversion
        theta=0.16,   # Long-term vol
        sigma=0.3,    # Vol of vol
        rho=-0.7      # Correlation
    )
    
    # Test the calibrated parameters too
    calibrated_params = HestonParams(
        v0=2.0,
        theta=2.0,
        kappa=0.1,
        sigma=1.0,
        rho=-0.95
    )
    
    calibrator = HestonCalibrator(risk_free_rate=0.03)
    
    print("\n=== SAMPLE OPTION PRICING COMPARISON ===")
    
    # Take first 5 options for detailed analysis
    sample_options = option_data.head(5)
    
    for i, (_, option) in enumerate(sample_options.iterrows()):
        strike = option['Strike']
        time_to_expiry = option['DaysToExpiry'] / 365.0
        market_iv = option['ImpliedVolatility']
        option_type = option['OptionType']
        
        print(f"\n--- Option {i+1}: {option_type} K={strike:.1f} T={time_to_expiry:.3f} ---")
        print(f"Market IV: {market_iv:.4f} ({market_iv*100:.1f}%)")
        
        # Market price using Black-Scholes with market IV
        market_price = black_scholes_price(spot_price, strike, time_to_expiry, 0.03, market_iv, option_type)
        print(f"Market price (BS): ${market_price:.3f}")
        
        # Heston model prices
        try:
            test_price = calibrator._price_option_fft(spot_price, strike, time_to_expiry, test_params, option_type)
            test_iv = calibrator._calculate_model_implied_volatility(spot_price, strike, time_to_expiry, test_params, option_type)
            print(f"Test Heston price: ${test_price:.3f}, IV: {test_iv:.4f} ({test_iv*100:.1f}%)")
        except Exception as e:
            print(f"Test Heston error: {e}")
        
        try:
            cal_price = calibrator._price_option_fft(spot_price, strike, time_to_expiry, calibrated_params, option_type)
            cal_iv = calibrator._calculate_model_implied_volatility(spot_price, strike, time_to_expiry, calibrated_params, option_type)
            print(f"Calibrated Heston price: ${cal_price:.3f}, IV: {cal_iv:.4f} ({cal_iv*100:.1f}%)")
        except Exception as e:
            print(f"Calibrated Heston error: {e}")
    
    print(f"\n=== OBJECTIVE FUNCTION ANALYSIS ===")
    
    # Calculate objective function values
    test_obj = calibrator._calibration_objective_function(
        [test_params.v0, test_params.kappa, test_params.theta, test_params.sigma, test_params.rho],
        option_data, spot_price
    )
    
    cal_obj = calibrator._calibration_objective_function(
        [calibrated_params.v0, calibrated_params.kappa, calibrated_params.theta, calibrated_params.sigma, calibrated_params.rho],
        option_data, spot_price
    )
    
    print(f"Test parameters objective: {test_obj:.6f}")
    print(f"Calibrated parameters objective: {cal_obj:.6f}")
    
    # Check if calibrated is actually better
    if cal_obj < test_obj:
        print("✓ Calibrated parameters are better")
    else:
        print("✗ Calibrated parameters are worse - optimization failed")

if __name__ == "__main__":
    diagnose_calibration()
