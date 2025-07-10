"""
Patched HestonCalibrator with Feller constraint enforcement and robust implied vol solver.
Save this as heston_calibrator_fixed.py and import it instead of the original.
"""

import os
import sys
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.stats import norm
from math import log, sqrt, exp

# Dynamically load the original HestonCalibrator to avoid circular imports
orig_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'heston_calibrator.py'))
import importlib.util
spec = importlib.util.spec_from_file_location('orig_calibrator', orig_path)
orig_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orig_mod)
BaseCalibrator = orig_mod.HestonCalibrator

class HestonCalibrator(BaseCalibrator):
    def _calibration_objective_function(self, parameter_vector, market_option_data, spot_price):
        """Enforce Feller's condition via penalty before calibration."""
        # Unpack parameters
        v0, kappa, theta, sigma, rho = parameter_vector
        # Penalize any violation of the Feller condition
        if 2 * kappa * theta <= sigma**2:
            return 1e6
        # Otherwise defer to original objective
        return super()._calibration_objective_function(parameter_vector, market_option_data, spot_price)

    def _calculate_model_implied_volatility(self, spot_price, strike_price, time_to_expiry, heston_params, option_type):
        """
        Compute implied volatility by:
        1) pricing with the Heston-FFT engine
        2) inverting with a robust bisection (brentq) on the Black-Scholes price
        """
        # 1) Model price via FFT
        model_price = self._price_option_fft(
            spot_price, strike_price, time_to_expiry, heston_params, option_type
        )

        # 2) Define Black-Scholes pricing function
        def bs_price(vol):
            d1 = (log(spot_price / strike_price) + (self.risk_free_rate + 0.5 * vol**2) * time_to_expiry) / (vol * sqrt(time_to_expiry))
            d2 = d1 - vol * sqrt(time_to_expiry)
            if option_type.lower() == 'call':
                return spot_price * norm.cdf(d1) - strike_price * exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2)
            else:
                return strike_price * exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)

        # 3) Solve for implied vol with brentq over [1e-6, 5.0]
        try:
            implied_vol = brentq(lambda v: bs_price(v) - model_price, 1e-6, 5.0)
        except ValueError:
            implied_vol = np.nan
        return implied_vol
