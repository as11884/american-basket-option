"""
Direct Heston pricing test without FFT
"""
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from math import log, sqrt, exp, pi
import sys
sys.path.append('.')

from calib_utils import HestonParams

def heston_call_price_direct(spot, strike, time_to_expiry, risk_free_rate, heston_params):
    """
    Direct Heston call option pricing using numerical integration
    """
    
    def heston_char_func(u, spot, strike, time_to_expiry, risk_free_rate, heston_params):
        """Heston characteristic function"""
        v0, kappa, theta, sigma, rho = heston_params.v0, heston_params.kappa, heston_params.theta, heston_params.sigma, heston_params.rho
        
        i = 1j
        
        # Complex coefficients  
        d = np.sqrt((rho * sigma * i * u - kappa)**2 - sigma**2 * (-i * u - u**2))
        g = (kappa - rho * sigma * i * u - d) / (kappa - rho * sigma * i * u + d)
        
        # Time-dependent terms
        exp_dt = np.exp(-d * time_to_expiry)
        
        # C and D functions
        C = (risk_free_rate * i * u * time_to_expiry + 
             (kappa * theta / sigma**2) * 
             ((kappa - rho * sigma * i * u - d) * time_to_expiry - 
              2 * np.log((1 - g * exp_dt) / (1 - g))))
        
        D = ((kappa - rho * sigma * i * u - d) / sigma**2) * ((1 - exp_dt) / (1 - g * exp_dt))
        
        # Characteristic function
        char_func = np.exp(C + D * v0 + i * u * log(spot))
        
        return char_func
    
    def integrand_1(u):
        """Integrand for P1"""
        char_func = heston_char_func(u - 1j, spot, strike, time_to_expiry, risk_free_rate, heston_params)
        return (np.exp(-1j * u * log(strike)) * char_func / (1j * u)).real
    
    def integrand_2(u):
        """Integrand for P2"""  
        char_func = heston_char_func(u, spot, strike, time_to_expiry, risk_free_rate, heston_params)
        return (np.exp(-1j * u * log(strike)) * char_func / (1j * u)).real
    
    # Numerical integration
    try:
        P1, _ = quad(integrand_1, 0, 100, limit=100)
        P2, _ = quad(integrand_2, 0, 100, limit=100)
        
        P1 = 0.5 + P1 / pi
        P2 = 0.5 + P2 / pi
        
        call_price = spot * P1 - strike * exp(-risk_free_rate * time_to_expiry) * P2
        
        return max(0, call_price)
    except:
        return np.nan

def test_direct_heston():
    """Test direct Heston pricing"""
    
    # Test parameters
    spot = 100.0
    strike = 100.0
    time_to_expiry = 0.25
    risk_free_rate = 0.03
    
    # Different parameter sets
    test_cases = [
        ("Base", HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)),
        ("High vol", HestonParams(v0=0.16, kappa=2.0, theta=0.16, sigma=0.3, rho=-0.7)),
        ("Low vol", HestonParams(v0=0.01, kappa=2.0, theta=0.01, sigma=0.3, rho=-0.7)),
    ]
    
    print("Testing direct Heston pricing...")
    print("=" * 50)
    
    for name, params in test_cases:
        try:
            price = heston_call_price_direct(spot, strike, time_to_expiry, risk_free_rate, params)
            
            # Compare with Black-Scholes
            vol = sqrt(params.v0)
            d1 = (log(spot/strike) + (risk_free_rate + 0.5*vol**2)*time_to_expiry) / (vol*sqrt(time_to_expiry))
            d2 = d1 - vol*sqrt(time_to_expiry)
            bs_price = spot*norm.cdf(d1) - strike*exp(-risk_free_rate*time_to_expiry)*norm.cdf(d2)
            
            print(f"{name:10s}: Heston=${price:6.3f}, BS=${bs_price:6.3f}")
            
        except Exception as e:
            print(f"{name:10s}: ERROR - {e}")

if __name__ == "__main__":
    test_direct_heston()
