"""
Simple working Heston implementation for testing
"""
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from math import log, sqrt, exp, pi
import sys
sys.path.append('.')

from calib_utils import HestonParams

def heston_call_price_simple(S, K, T, r, v0, kappa, theta, sigma, rho):
    """
    Simple Heston call option pricing using the classic formulation
    """
    
    def heston_integrand(phi, j):
        """Heston integrand for probability calculation"""
        if j == 1:
            u = phi - 1j
            b = kappa - rho * sigma
        else:
            u = phi
            b = kappa
            
        a = kappa * theta
        
        d = np.sqrt((rho * sigma * u * 1j - b)**2 - sigma**2 * (2 * u * 1j - u**2))
        g = (b - rho * sigma * u * 1j - d) / (b - rho * sigma * u * 1j + d)
        
        exp_term = np.exp(-d * T)
        
        C = r * u * 1j * T + (a / sigma**2) * ((b - rho * sigma * u * 1j - d) * T - 2 * np.log((1 - g * exp_term) / (1 - g)))
        D = (b - rho * sigma * u * 1j - d) / sigma**2 * (1 - exp_term) / (1 - g * exp_term)
        
        f = np.exp(C + D * v0 + 1j * u * log(S))
        
        return np.real(np.exp(-1j * u * log(K)) * f / (1j * u))
    
    # Calculate P1 and P2
    try:
        P1 = 0.5 + (1/pi) * quad(lambda phi: heston_integrand(phi, 1), 0, 100)[0]
        P2 = 0.5 + (1/pi) * quad(lambda phi: heston_integrand(phi, 2), 0, 100)[0]
        
        call_price = S * P1 - K * exp(-r * T) * P2
        
        return max(0, call_price)
    except:
        return np.nan

def test_simple_heston():
    """Test simple Heston pricing"""
    
    # Test parameters
    S = 100.0
    K = 100.0
    T = 0.25
    r = 0.03
    
    # Different parameter sets
    test_cases = [
        ("Base", 0.04, 2.0, 0.04, 0.3, -0.7),
        ("High vol", 0.16, 2.0, 0.16, 0.3, -0.7),
        ("Low vol", 0.01, 2.0, 0.01, 0.3, -0.7),
    ]
    
    print("Testing simple Heston pricing...")
    print("=" * 50)
    
    for name, v0, kappa, theta, sigma, rho in test_cases:
        try:
            price = heston_call_price_simple(S, K, T, r, v0, kappa, theta, sigma, rho)
            
            # Compare with Black-Scholes
            vol = sqrt(v0)
            d1 = (log(S/K) + (r + 0.5*vol**2)*T) / (vol*sqrt(T))
            d2 = d1 - vol*sqrt(T)
            bs_price = S*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
            
            print(f"{name:10s}: Heston=${price:6.3f}, BS=${bs_price:6.3f}")
            
        except Exception as e:
            print(f"{name:10s}: ERROR - {e}")

if __name__ == "__main__":
    test_simple_heston()
