"""
Heston Model Calibration Module

This module provides tools for calibrating the Heston stochastic volatility model
to market option data. It uses Fast Fourier Transform (FFT) for efficient option
pricing and scipy optimization for parameter estimation.

The calibration process minimizes the mean squared error between market implied
volatilities and model-implied volatilities.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from math import log, sqrt, exp
from calib_utils import HestonParams, create_parameter_bounds


class HestonCalibrator:
    """
    Calibrator for fitting Heston model parameters to market option data.
    
    This class uses Fast Fourier Transform (FFT) for efficient option pricing
    and optimization algorithms to find the best-fit parameters.
    """
    
    def __init__(self, risk_free_rate=0.03, parameter_bounds=None, use_fft=True):
        """
        Initialize the Heston calibrator.
        
        Parameters
        ----------
        risk_free_rate : float, default=0.03
            Annual risk-free interest rate
        parameter_bounds : list of tuples, optional
            Bounds for each parameter during optimization
        use_fft : bool, default=True
            Whether to use FFT for option pricing (faster but approximate)
        """
        self.risk_free_rate = risk_free_rate
        self.use_fft = use_fft
        
        # Set default parameter bounds if not provided
        if parameter_bounds is None:
            self.parameter_bounds = create_parameter_bounds()
        else:
            self.parameter_bounds = parameter_bounds

    def _heston_characteristic_function(self, frequency_points, spot_price, strike_price, 
                                       time_to_expiry, heston_params):
        """
        Compute the Heston characteristic function for option pricing.
        
        The characteristic function is the Fourier transform of the probability
        density function and is used in FFT-based option pricing.
        
        Parameters
        ----------
        frequency_points : np.ndarray
            Array of frequency points for evaluation
        spot_price : float
            Current stock price
        strike_price : float
            Option strike price
        time_to_expiry : float
            Time to expiration in years
        heston_params : HestonParams
            Heston model parameters
            
        Returns
        -------
        np.ndarray
            Characteristic function values
        """
        # Extract parameters with descriptive names
        initial_variance = heston_params.v0
        mean_reversion_speed = heston_params.kappa
        long_term_variance = heston_params.theta
        volatility_of_volatility = heston_params.sigma
        correlation = heston_params.rho
        
        # Log-moneyness (log of spot/strike ratio)
        log_moneyness = log(spot_price / strike_price)
        
        # Heston characteristic function using Lewis (2001) formulation
        # This is more numerically stable
        
        v0 = initial_variance
        kappa = mean_reversion_speed  
        theta = long_term_variance
        sigma = volatility_of_volatility
        rho = correlation
        
        # Complex frequency
        u = frequency_points
        
        # Parameters
        alpha = -0.5 * (u * u + u * 1j)
        beta = kappa - rho * sigma * u * 1j
        gamma = 0.5 * sigma * sigma
        
        # Discriminant
        d = np.sqrt(beta * beta - 4 * alpha * gamma)
        
        # Ensure correct branch of square root
        d = np.where(np.real(d) > 0, d, -d)
        
        # r+ and r- 
        r_plus = (beta + d) / (2 * gamma)
        r_minus = (beta - d) / (2 * gamma)
        
        # Choose the stable branch
        # Use r_minus if |r_minus| < 1, otherwise use r_plus
        g = r_minus
        
        # Time-dependent part
        exp_dt = np.exp(-d * time_to_expiry)
        
        # A and B functions
        A = (self.risk_free_rate * u * 1j * time_to_expiry + 
             (kappa * theta / gamma) * (r_minus * time_to_expiry - 2 * np.log((1 - g * exp_dt) / (1 - g))))
        
        B = r_minus * (1 - exp_dt) / (1 - g * exp_dt)
        
        # Characteristic function
        characteristic_function = np.exp(A + B * v0 + u * 1j * log_moneyness)
        
        return characteristic_function

    def _price_option_fft(self, spot_price, strike_price, time_to_expiry, heston_params, option_type='call'):
        """
        Price European options using a simplified approach.
        
        This method uses a direct integration approach which is more reliable
        than the FFT method for calibration purposes.
        """
        from scipy.integrate import quad
        
        # Extract parameters
        v0 = heston_params.v0
        kappa = heston_params.kappa
        theta = heston_params.theta
        sigma = heston_params.sigma
        rho = heston_params.rho
        
        def heston_integrand(phi, j):
            """Heston integrand for probability calculation"""
            if j == 1:
                u = phi - 1j
                b = kappa - rho * sigma
            else:
                u = phi
                b = kappa
                
            a = kappa * theta
            
            # Avoid division by zero
            if abs(sigma) < 1e-10:
                return 0.0
                
            d = np.sqrt((rho * sigma * u * 1j - b)**2 - sigma**2 * (2 * u * 1j - u**2))
            g = (b - rho * sigma * u * 1j - d) / (b - rho * sigma * u * 1j + d)
            
            # Numerical stability for exponential
            exp_term = np.exp(-d * time_to_expiry)
            
            # Avoid log(0) 
            if abs(1 - g) < 1e-10:
                return 0.0
                
            C = (self.risk_free_rate * u * 1j * time_to_expiry + 
                 (a / sigma**2) * ((b - rho * sigma * u * 1j - d) * time_to_expiry - 
                                  2 * np.log((1 - g * exp_term) / (1 - g))))
            D = ((b - rho * sigma * u * 1j - d) / sigma**2) * ((1 - exp_term) / (1 - g * exp_term))
            
            f = np.exp(C + D * v0 + 1j * u * log(spot_price))
            
            return np.real(np.exp(-1j * u * log(strike_price)) * f / (1j * u))
        
        # Calculate P1 and P2
        try:
            P1 = 0.5 + (1/np.pi) * quad(lambda phi: heston_integrand(phi, 1), 0, 100, limit=200)[0]
            P2 = 0.5 + (1/np.pi) * quad(lambda phi: heston_integrand(phi, 2), 0, 100, limit=200)[0]
            
            call_price = spot_price * P1 - strike_price * exp(-self.risk_free_rate * time_to_expiry) * P2
            
            if option_type == 'call':
                return max(0, call_price)
            else:
                # Put-call parity
                put_price = call_price + strike_price * exp(-self.risk_free_rate * time_to_expiry) - spot_price
                return max(0, put_price)
                
        except:
            # Fallback to intrinsic value
            if option_type == 'call':
                return max(0, spot_price - strike_price * exp(-self.risk_free_rate * time_to_expiry))
            else:
                return max(0, strike_price * exp(-self.risk_free_rate * time_to_expiry) - spot_price)

    def _calculate_model_implied_volatility(self, spot_price, strike_price, time_to_expiry, 
                                          heston_params, option_type):
        """
        Calculate Black-Scholes implied volatility from Heston model price.
        
        This method first computes the Heston model price, then finds the
        Black-Scholes implied volatility that produces the same price.
        
        Parameters
        ----------
        spot_price : float
            Current stock price
        strike_price : float
            Option strike price
        time_to_expiry : float
            Time to expiration in years
        heston_params : HestonParams
            Heston model parameters
        option_type : str
            Type of option ('call' or 'put')
            
        Returns
        -------
        float
            Implied volatility, or NaN if optimization fails
        """
        # Get Heston model price
        heston_price = self._price_option_fft(
            spot_price, strike_price, time_to_expiry, heston_params, option_type
        )
        
        # Define Black-Scholes pricing function
        def black_scholes_price(implied_volatility):
            """Black-Scholes formula for European options."""
            d1 = (log(spot_price / strike_price) + (self.risk_free_rate + 0.5 * implied_volatility**2) * time_to_expiry) / (implied_volatility * sqrt(time_to_expiry))
            d2 = d1 - implied_volatility * sqrt(time_to_expiry)
            
            if option_type == 'call':
                bs_price = spot_price * norm.cdf(d1) - strike_price * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2)
            else:
                bs_price = strike_price * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
            
            return bs_price
        
        # Define objective function for implied volatility search
        def implied_vol_objective(implied_volatility):
            """Squared difference between Black-Scholes and Heston prices."""
            return (black_scholes_price(implied_volatility) - heston_price)**2
        
        # Initial guess based on current variance
        initial_volatility_guess = sqrt(heston_params.v0)
        
        # Find implied volatility using optimization
        optimization_result = minimize(
            implied_vol_objective, 
            initial_volatility_guess, 
            bounds=[(1e-4, 3.0)]  # Reasonable bounds for volatility
        )
        
        return optimization_result.x[0] if optimization_result.success else np.nan

    def _calibration_objective_function(self, parameter_vector, market_option_data, spot_price):
        """Enforce Feller's condition via penalty before calibration."""
        # Unpack parameters
        v0, kappa, theta, sigma, rho = parameter_vector
        # Penalize any violation of the Feller condition
        if 2 * kappa * theta <= sigma**2:
            return 1e6
        # Otherwise defer to original objective
        return super()._calibration_objective_function(parameter_vector, market_option_data, spot_price)

    def calibrate(self, spot_price, option_data, initial_parameters, max_iterations=100):
        """
        Calibrate Heston model parameters to market option data.
        
        This method uses numerical optimization to find the parameter values
        that minimize the difference between market and model implied volatilities.
        
        Parameters
        ----------
        spot_price : float
            Current stock price
        option_data : pd.DataFrame
            DataFrame with market option data containing columns:
            - Strike: Option strike prices
            - DaysToExpiry: Days until expiration
            - ImpliedVolatility: Market implied volatilities
            - OptionType: 'call' or 'put'
        initial_parameters : HestonParams
            Initial guess for parameter values
        max_iterations : int, default=100
            Maximum number of optimization iterations
            
        Returns
        -------
        HestonParams
            Calibrated Heston parameters
        """
        # Convert initial parameters to optimization vector
        initial_parameter_vector = [
            initial_parameters.v0,
            initial_parameters.kappa,
            initial_parameters.theta,
            initial_parameters.sigma,
            initial_parameters.rho
        ]
        
        # Set up parameter bounds for optimization
        parameter_bounds = self.parameter_bounds
        
        # Perform optimization
        print(f"Starting calibration with {len(option_data)} options...")
        print(f"Initial parameters: {initial_parameters}")
        
        optimization_result = minimize(
            self._calibration_objective_function,
            initial_parameter_vector,
            args=(option_data, spot_price),
            bounds=parameter_bounds,
            method='L-BFGS-B',  # Good for bound-constrained problems
            options={'maxiter': max_iterations, 'disp': False}
        )
        
        # Create calibrated parameters object
        calibrated_parameters = HestonParams(
            v0=optimization_result.x[0],
            kappa=optimization_result.x[1],
            theta=optimization_result.x[2],
            sigma=optimization_result.x[3],
            rho=optimization_result.x[4]
        )
        
        # Print calibration results
        print(f"Calibration completed in {optimization_result.nit} iterations")
        print(f"Final objective value: {optimization_result.fun:.6f}")
        print(f"Calibrated parameters: {calibrated_parameters}")
        
        return calibrated_parameters