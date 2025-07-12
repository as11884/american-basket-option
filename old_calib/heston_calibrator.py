"""
heston_calibrator_iv.py

Unified Heston calibration module in implied-volatility space.
Contains all components: parameter container, Black–Scholes helper, Heston FFT pricer,
and IV-based calibration via complex-step Jacobian.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import log, sqrt, exp, pi
from typing import Tuple, List

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import least_squares, brentq
from scipy.stats import norm

# ---------------------------------------------------------------------------
#  Parameter container
# ---------------------------------------------------------------------------
@dataclass
class HestonParams:
    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float

    @classmethod
    def from_vector(cls, x: np.ndarray) -> HestonParams:
        return cls(v0=x[0], kappa=x[1], theta=x[2], sigma=x[3], rho=x[4])

    def as_vector(self) -> np.ndarray:
        return np.array([self.v0, self.kappa, self.theta, self.sigma, self.rho])

    def to_dict(self) -> dict:
        return dict(zip(["v0", "kappa", "theta", "sigma", "rho"], self.as_vector()))

# ---------------------------------------------------------------------------
#  Parameter bounds
# ---------------------------------------------------------------------------
_BOUNDS: List[Tuple[float, float]] = [
    (1e-8, 3.0),      # v0
    (1e-3, 25.0),     # kappa
    (1e-8, 3.0),      # theta
    (1e-3, 6.0),      # sigma
    (-0.999, 0.999),  # rho
]

# ---------------------------------------------------------------------------
#  Black–Scholes pricing helper
# ---------------------------------------------------------------------------
def _bs_price(
    S: float, K: float, T: float, r: float, vol: float, option_type: str = "call"
) -> float:
    if vol < 1e-12 or T <= 0.0:
        # intrinsic value
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    d1 = (log(S / K) + (r + 0.5 * vol**2) * T) / (vol * sqrt(T))
    d2 = d1 - vol * sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def _bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black–Scholes vega: ∂Price/∂σ
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

# ---------------------------------------------------------------------------
#  Heston characteristic-function pricer
# ---------------------------------------------------------------------------
class HestonPricer:
    def __init__(self, r: float):
        self.r = r

    def _phi(self, u: complex, T: float, p: HestonParams, S: float) -> complex:
        i = 1j
        v0, kappa, theta, sigma, rho = p.v0, p.kappa, p.theta, p.sigma, p.rho
        d = np.sqrt((rho * sigma * i * u - kappa)**2 + sigma**2 * (i * u + u**2))
        g = (kappa - rho * sigma * i * u - d) / (kappa - rho * sigma * i * u + d)
        e_dt = np.exp(-d * T)
        C = (kappa * theta / sigma**2) * (
            (kappa - rho * sigma * i * u - d) * T - 2.0 * np.log((1 - g * e_dt) / (1 - g))
        )
        D = ((kappa - rho * sigma * i * u - d) / sigma**2) * ((1 - e_dt) / (1 - g * e_dt))
        return np.exp(C + D * v0 + i * u * (np.log(S) + self.r * T))

    def price(self, S: float, K: float, T: float, p: HestonParams, otype: str = "call") -> complex:
        lnK = log(K)
        # integrands
        def f1(u): return np.exp(-1j * u * lnK) * self._phi(u - 1j, T, p, S) / (1j * u * self._phi(-1j, T, p, S))
        def f2(u): return np.exp(-1j * u * lnK) * self._phi(u, T, p, S) / (1j * u)
        
        # avoid singularity at zero
        eps = 1e-5
        def int_complex(func):
            re = quad(lambda x: func(x).real, eps, 150.0, epsabs=1e-7, limit=500)[0]
            im = quad(lambda x: func(x).imag, eps, 150.0, epsabs=1e-7, limit=500)[0]
            return re + 1j * im

        P1 = 0.5 + int_complex(f1) / pi
        P2 = 0.5 + int_complex(f2) / pi
        call_cf = S * P1 - K * exp(-self.r * T) * P2

        # propagate complex for gradient
        if any(isinstance(val, complex) for val in p.as_vector()):
            return call_cf
        
        # return real price
        price = call_cf.real
        if otype == "call":
            return max(0.0, price)
        # put-call parity for put price
        return max(0.0, price - S + K * exp(-self.r * T))

# ---------------------------------------------------------------------------
#  IV-space Heston calibrator
# ---------------------------------------------------------------------------
class IVHestonCalibrator:
    """Calibrate Heston by minimizing (model_IV - market_IV)^2."""

    def __init__(self, r: float, h: float = 1e-4):
        self.r = r
        self._pricer = HestonPricer(r)
        self._h = h # Complex-step size

    def _price_and_grad(
        self, S: float, K: float, T: float, p: HestonParams, otype: str
    ) -> Tuple[float, np.ndarray]:
        base_price = self._pricer.price(S, K, T, p, otype)
        grad = np.empty(5)
        for j, nm in enumerate(["v0", "kappa", "theta", "sigma", "rho"]):
            p_shift = HestonParams(*p.as_vector())
            setattr(p_shift, nm, getattr(p_shift, nm) + self._h * 1j)
            shifted_price = self._pricer.price(S, K, T, p_shift, otype)
            grad[j] = shifted_price.imag / self._h
        return base_price, grad

    def _residual_jac(self, x: np.ndarray, iv_surface: pd.DataFrame, S: float) -> Tuple[np.ndarray, np.ndarray]:
        p = HestonParams.from_vector(x)
        n = len(iv_surface)
        res = np.zeros(n)
        J = np.zeros((n, len(x)))

        for i, (_, row) in enumerate(iv_surface.iterrows()):
            K = float(row.Strike)
            T = float(row.DaysToExpiry) / 365.0
            iv_mkt = float(row.ImpliedVolatility)
            otype = "call"

            # 1. Price & gradient from Heston model using complex-step
            price0, grad0 = self._price_and_grad(S, K, T, p, otype)

            # 2. Invert Heston price to get model's implied volatility
            try:
                model_iv = brentq(
                    lambda v: _bs_price(S, K, T, self.r, v, otype) - price0,
                    1e-6, 10.0, xtol=1e-8, rtol=1e-8 # Increased upper bound for robustness
                )
            except ValueError:
                # Could not find a root, likely due to extreme parameters
                model_iv = 1e-6 # Assign a floor value

            # 3. Calculate Black-Scholes vega at the model's IV
            vega_model = _bs_vega(S, K, T, self.r, model_iv)

            # 4. Fill residual and Jacobian
            res[i] = model_iv - iv_mkt
            
            # **STABILITY FIX**: Avoid division by near-zero vega
            if vega_model > 1e-8:
                J[i, :] = grad0 / vega_model
            else:
                J[i, :] = 0.0 # Set gradient to zero for this point if vega is negligible

        return res, J

    def calibrate(
        self,
        spot: float,
        iv_surface: pd.DataFrame,
        initial_params: HestonParams,
        max_iter: int = 200,
    ) -> HestonParams:
        n = len(iv_surface)
        print(f"[IV] Calibrating on {n} IV quotes …")
        print("Initial guess:", initial_params.to_dict())

        # Objective and Jacobian functions for the optimizer
        # NOTE: This calls _residual_jac twice per iteration. For better performance,
        # you could use a class to cache the results of the last call.
        fun = lambda x: self._residual_jac(x, iv_surface, spot)[0]
        jac = lambda x: self._residual_jac(x, iv_surface, spot)[1]

        # Extract bounds
        lb, ub = zip(*_BOUNDS)

        # Run Trust Region Reflective (TRF) solver
        res = least_squares(
            fun,
            initial_params.as_vector(),
            jac=jac,
            bounds=(np.array(lb), np.array(ub)),
            method="trf",
            x_scale="jac",      # Auto-scale step sizes by Jacobian
            ftol=1e-9,          # Convergence on cost
            xtol=1e-9,          # Convergence on step changes
            max_nfev=max_iter,
            verbose=2
        )

        fitted = HestonParams.from_vector(res.x)
        mse = np.mean((res.fun)**2) # Mean Squared Error of IVs
        print(f"Completed after {res.nfev} evals — IV MSE = {mse:.6g}")
        print("Calibrated parameters:", fitted.to_dict())
        return fitted