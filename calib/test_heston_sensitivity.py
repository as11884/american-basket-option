"""
Test script to check if the Heston model is sensitive to parameter changes
"""
import numpy as np
import sys
sys.path.append('.')

from heston_calibrator import HestonCalibrator
from calib_utils import HestonParams

def test_heston_sensitivity():
    """Test if Heston model prices change with different parameters"""
    
    # Create calibrator
    calibrator = HestonCalibrator(risk_free_rate=0.03)
    
    # Test parameters
    spot_price = 100.0
    strike_price = 100.0
    time_to_expiry = 0.25  # 3 months
    
    # Base parameters
    base_params = HestonParams(
        v0=0.04,      # 20% vol
        kappa=2.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.7
    )
    
    # Test different parameter sets
    test_cases = [
        ("Base", base_params),
        ("High vol", HestonParams(v0=0.16, kappa=2.0, theta=0.16, sigma=0.3, rho=-0.7)),
        ("Low vol", HestonParams(v0=0.01, kappa=2.0, theta=0.01, sigma=0.3, rho=-0.7)),
        ("High kappa", HestonParams(v0=0.04, kappa=5.0, theta=0.04, sigma=0.3, rho=-0.7)),
        ("Low kappa", HestonParams(v0=0.04, kappa=0.5, theta=0.04, sigma=0.3, rho=-0.7)),
        ("High sigma", HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.8, rho=-0.7)),
        ("Low sigma", HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.1, rho=-0.7)),
        ("High rho", HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=0.5)),
        ("Low rho", HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.9)),
    ]
    
    print("Testing Heston model sensitivity...")
    print(f"Spot: {spot_price}, Strike: {strike_price}, Time: {time_to_expiry}")
    print("=" * 60)
    
    for name, params in test_cases:
        try:
            # Price call option
            call_price = calibrator._price_option_fft(
                spot_price, strike_price, time_to_expiry, params, 'call'
            )
            
            # Calculate implied volatility
            impl_vol = calibrator._calculate_model_implied_volatility(
                spot_price, strike_price, time_to_expiry, params, 'call'
            )
            
            print(f"{name:12s}: Call=${call_price:6.3f}, ImplVol={impl_vol:.4f}")
            
        except Exception as e:
            print(f"{name:12s}: ERROR - {e}")
    
    print("=" * 60)

if __name__ == "__main__":
    test_heston_sensitivity()
