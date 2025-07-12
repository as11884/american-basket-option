"""
Test to verify the Heston characteristic function against known results
"""
import numpy as np
from math import log, sqrt, exp, pi
import sys
sys.path.append('.')

from heston_calibrator import HestonCalibrator
from calib_utils import HestonParams

def test_heston_cf():
    """Test the Heston characteristic function"""
    
    # Simple test case
    spot = 100.0
    strike = 100.0
    time_to_expiry = 0.25
    risk_free_rate = 0.05
    
    # Simple Heston parameters
    params = HestonParams(
        v0=0.04,      # 20% vol
        kappa=2.0,    # Mean reversion
        theta=0.04,   # Long-term vol
        sigma=0.1,    # Vol of vol
        rho=-0.5      # Correlation
    )
    
    calibrator = HestonCalibrator(risk_free_rate=risk_free_rate)
    
    # Test characteristic function at u=0 (should be 1)
    cf_at_zero = calibrator._heston_characteristic_function(
        np.array([0.0]), spot, strike, time_to_expiry, params
    )
    
    print(f"Characteristic function at u=0: {cf_at_zero[0]}")
    print(f"Should be close to S/K = {spot/strike:.3f}")
    
    # Test at u=1 (should be related to forward price)
    cf_at_one = calibrator._heston_characteristic_function(
        np.array([1.0]), spot, strike, time_to_expiry, params
    )
    
    print(f"Characteristic function at u=1: {cf_at_one[0]}")
    
    # Test option pricing
    call_price = calibrator._price_option_fft(spot, strike, time_to_expiry, params, 'call')
    put_price = calibrator._price_option_fft(spot, strike, time_to_expiry, params, 'put')
    
    print(f"Call price: ${call_price:.3f}")
    print(f"Put price: ${put_price:.3f}")
    
    # Check put-call parity
    forward = spot * exp(risk_free_rate * time_to_expiry)
    pv_strike = strike * exp(-risk_free_rate * time_to_expiry)
    pcp_call = put_price + spot - pv_strike
    pcp_put = call_price - spot + pv_strike
    
    print(f"Put-call parity check:")
    print(f"  Call from put: ${pcp_call:.3f} vs actual call: ${call_price:.3f}")
    print(f"  Put from call: ${pcp_put:.3f} vs actual put: ${put_price:.3f}")
    
    # Compare with intrinsic values
    call_intrinsic = max(0, spot - pv_strike)
    put_intrinsic = max(0, pv_strike - spot)
    
    print(f"Intrinsic values:")
    print(f"  Call intrinsic: ${call_intrinsic:.3f}")
    print(f"  Put intrinsic: ${put_intrinsic:.3f}")

if __name__ == "__main__":
    test_heston_cf()
