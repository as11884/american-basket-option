import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import least_squares

# ---------------------------------------------------------------------------
#  Heston parameter container
# ---------------------------------------------------------------------------
@dataclass
class HestonParams:
    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float

    @classmethod
    def from_vector(cls, x: np.ndarray) -> "HestonParams":
        return cls(v0=x[0], kappa=x[1], theta=x[2], sigma=x[3], rho=x[4])

    def as_vector(self) -> np.ndarray:
        return np.array([self.v0, self.kappa, self.theta, self.sigma, self.rho])

# ---------------------------------------------------------------------------
#  Bounds for calibration
# ---------------------------------------------------------------------------
_BOUNDS = [
    (1e-4, 0.5),     # v0 - initial variance (up to 70% vol)
    (0.1, 20.0),     # kappa - mean reversion speed
    (1e-4, 0.5),     # theta - long-term variance
    (0.01, 2.0),     # sigma - vol of vol
    (-0.95, 0.95),   # rho - correlation
]

# ---------------------------------------------------------------------------
#  Black–Scholes pricer & vega
# ---------------------------------------------------------------------------
from math import log, sqrt, exp, pi
class _BS:
    @staticmethod
    def price(S, K, T, r, vol, otype='call'):
        if vol < 1e-12 or T <= 0:
            return max(0.0, S - K) if otype=='call' else max(0.0, K - S)
        d1 = (log(S/K) + (r + 0.5*vol**2)*T) / (vol*sqrt(T))
        d2 = d1 - vol*sqrt(T)
        return S * norm.cdf(d1) - K * exp(-r*T) * norm.cdf(d2) if otype=='call' else K * exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def vega(S, K, T, r, vol):
        if vol <= 0 or T <= 0:
            return 0.0
        d1 = (log(S/K) + (r + 0.5*vol**2)*T) / (vol*sqrt(T))
        return S * sqrt(T) * norm.pdf(d1)

# ---------------------------------------------------------------------------
#  Heston characteristic-function pricer
# ---------------------------------------------------------------------------
class HestonPricer:
    def __init__(self, r: float, h_cs: float = 1e-5):
        self.r = r
        self.h = h_cs

    def _phi(self, u: complex, T: float, p: HestonParams, S: float) -> complex:
        """Heston characteristic function with numerical stability improvements"""
        i = 1j
        v0, k, th, sig, rh = p.v0, p.kappa, p.theta, p.sigma, p.rho
        
        # Add small epsilon to avoid division by zero
        eps = 1e-12
        
        # Calculate discriminant with stability check
        d = np.sqrt((rh*sig*i*u - k)**2 + sig**2*(i*u + u**2) + eps)
        
        # Use the stable formulation to avoid cancellation
        g = (k - rh*sig*i*u - d) / (k - rh*sig*i*u + d)
        
        # Prevent overflow in exponential
        if abs(d*T) > 50:
            return 0.0 + 0.0j
            
        e_dt = np.exp(-d*T)
        
        # Check for numerical issues
        if abs(1 - g) < eps or abs(1 - g*e_dt) < eps:
            return 0.0 + 0.0j
        
        # Calculate C and D with stability checks
        log_term = np.log((1 - g*e_dt) / (1 - g))
        if not np.isfinite(log_term):
            return 0.0 + 0.0j
            
        C = (k*th/sig**2) * ((k - rh*sig*i*u - d)*T - 2*log_term)
        D = ((k - rh*sig*i*u - d)/sig**2) * ((1 - e_dt)/(1 - g*e_dt))
        
        # Final check
        result = np.exp(C + D*v0 + i*u*(np.log(S) + self.r*T))
        
        if not np.isfinite(result):
            return 0.0 + 0.0j
            
        return result

    def _price_internal(self, S, K, T, p: HestonParams) -> float:
        """Calculate Heston option price using characteristic function"""
        try:
            # Use a simpler but more robust integration approach
            def integrand1(u):
                phi = self._phi(u - 1j, T, p, S)
                if not np.isfinite(phi) or abs(phi) > 1e10:
                    return 0.0
                result = (phi * np.exp(-1j * u * np.log(K)) / (1j * u)).real
                return result if np.isfinite(result) else 0.0
            
            def integrand2(u):
                phi = self._phi(u, T, p, S)
                if not np.isfinite(phi) or abs(phi) > 1e10:
                    return 0.0
                result = (phi * np.exp(-1j * u * np.log(K)) / (1j * u)).real
                return result if np.isfinite(result) else 0.0
            
            # Integrate with more conservative bounds
            I1, _ = quad(integrand1, 1e-8, 50, limit=200, epsabs=1e-10)
            I2, _ = quad(integrand2, 1e-8, 50, limit=200, epsabs=1e-10)
            
            P1 = 0.5 + I1 / np.pi
            P2 = 0.5 + I2 / np.pi
            
            call_price = S * P1 - K * np.exp(-self.r * T) * P2
            
            # Basic sanity checks
            if not np.isfinite(call_price) or call_price < 0:
                raise ValueError("Invalid price computed")
                
            # Additional check: price shouldn't be too far from intrinsic value
            intrinsic = max(0, S - K * np.exp(-self.r * T))
            if call_price < intrinsic * 0.99:  # Allow small numerical error
                raise ValueError("Price below intrinsic value")
                
            return call_price
            
        except Exception as e:
            # If Heston fails, print debug info and fall back to BS
            print(f"Heston pricing failed: {e}, falling back to BS")
            vol_approx = np.sqrt(p.v0)
            return _BS.price(S, K, T, self.r, vol_approx, 'call')

    def price_and_grad(self, S, K, T, p: HestonParams) -> Tuple[float, np.ndarray]:
        base = float(self._price_internal(S, K, T, p))  # Ensure float type
        grad = np.zeros(5)
        for j, name in enumerate(["v0", "kappa", "theta", "sigma", "rho"]):
            vec = p.as_vector().astype(complex)
            vec[j] += self.h * 1j
            shifted = self._price_internal(S, K, T, HestonParams.from_vector(vec))
            if isinstance(shifted, complex):
                grad[j] = shifted.imag / self.h
            else:
                # Finite difference if complex step fails
                vec_real = p.as_vector()
                vec_real[j] += self.h
                shifted_real = self._price_internal(S, K, T, HestonParams.from_vector(vec_real))
                grad[j] = (shifted_real - base) / self.h
        return base, grad

# ---------------------------------------------------------------------------
#  Calibrator minimizing relative price errors
# ---------------------------------------------------------------------------
class PriceHestonCalibrator:
    def __init__(self, r: float):
        self.r = r
        self._pricer = HestonPricer(r)

    def _residual(self, x: np.ndarray, S: float, market_data: pd.DataFrame) -> np.ndarray:
        p = HestonParams.from_vector(x)
        res = np.full(len(market_data), 0.0)  # Fixed size array
        
        for i, (_, row) in enumerate(market_data.iterrows()):
            try:
                K = row.Strike
                T = row.DaysToExpiry / 365.0  # Use 365 instead of 252
                mp = row.MarketPrice
                opt_type = row.OptionType
                
                # Get call price from Heston
                call_price, _ = self._pricer.price_and_grad(S, K, T, p)
                
                # Convert to put price if needed using put-call parity
                if opt_type == 'put':
                    # Put = Call - S + K*e^(-r*T)
                    model_price = call_price - S + K * np.exp(-self.r * T)
                else:
                    model_price = call_price
                
                # Check for valid prices
                if mp <= 0 or not np.isfinite(model_price):
                    res[i] = 100.0  # Large penalty for invalid prices
                elif model_price <= 0:
                    res[i] = 100.0  # Large penalty for negative model prices
                else:
                    # Use relative error
                    res[i] = (model_price - mp) / mp
                    
            except Exception as e:
                # Large penalty for failed calculations
                res[i] = 100.0
            
        return res

    def calibrate(self, spot: float, market_data: pd.DataFrame, initial: HestonParams) -> HestonParams:
        print(f"[Price] Calibrating on {len(market_data)} option prices...")
        print("Initial guess:", initial.as_vector())
        lb, ub = zip(*_BOUNDS)
        res = least_squares(
            self._residual,
            initial.as_vector(),
            jac='2-point',
            bounds=(lb,ub),
            method="trf",
            ftol=1e-8,xtol=1e-8,gtol=1e-8,
            verbose=2,
            args=(spot, market_data)
        )
        fitted = HestonParams.from_vector(res.x)
        print("\nCalibration complete. Parameters:", fitted.as_vector())
        return fitted
