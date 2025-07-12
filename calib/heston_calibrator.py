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
        
        This uses the standard Heston (1993) formulation with correct branch cuts.
        """
        # Extract parameters
        v0 = heston_params.v0
        kappa = heston_params.kappa
        theta = heston_params.theta
        sigma = heston_params.sigma
        rho = heston_params.rho
        
        # Frequency points
        u = frequency_points
        
        # Auxiliary parameters
        d = np.sqrt((rho * sigma * u * 1j - kappa)**2 - sigma**2 * (-u * 1j - u**2))
        g = (kappa - rho * sigma * u * 1j - d) / (kappa - rho * sigma * u * 1j + d)
        
        # Ensure correct branch cut for g
        # We want |g| < 1 for stability
        g = np.where(np.abs(g) < 1, g, 1/g)
        
        # Exponential term
        exp_dt = np.exp(-d * time_to_expiry)
        
        # A and B functions
        A = (self.risk_free_rate * u * 1j * time_to_expiry + 
             (kappa * theta / sigma**2) * 
             ((kappa - rho * sigma * u * 1j - d) * time_to_expiry - 
              2 * np.log((1 - g * exp_dt) / (1 - g))))
        
        B = ((kappa - rho * sigma * u * 1j - d) / sigma**2) * ((1 - exp_dt) / (1 - g * exp_dt))
        
        # Characteristic function
        return np.exp(A + B * v0)

    def _price_option_fft(self, spot_price, strike_price, time_to_expiry, heston_params, option_type='call'):
        """
        Price European options using the Heston model.
        
        Uses the standard Heston semi-analytical pricing formula.
        """
        from scipy.integrate import quad
        from scipy.stats import norm
        
        # For very short times or extreme parameters, use Black-Scholes
        if time_to_expiry < 1e-6 or heston_params.sigma > 5 or heston_params.v0 > 5:
            vol = min(sqrt(heston_params.v0), 3.0)  # Cap volatility
            d1 = (log(spot_price/strike_price) + (self.risk_free_rate + 0.5*vol**2)*time_to_expiry) / (vol*sqrt(time_to_expiry))
            d2 = d1 - vol*sqrt(time_to_expiry)
            
            if option_type == 'call':
                return spot_price * norm.cdf(d1) - strike_price * exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2)
            else:
                return strike_price * exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
        
        def integrand(u, j):
            """Integrand for Heston pricing formula"""
            try:
                if j == 1:
                    # P1: shift u by -i
                    u_shifted = u - 1j
                    char_func = self._heston_characteristic_function(
                        u_shifted, spot_price, strike_price, time_to_expiry, heston_params
                    )
                    # Adjust for the shift
                    char_func = char_func * np.exp(-1j * u * log(spot_price))
                else:
                    # P2: standard characteristic function
                    char_func = self._heston_characteristic_function(
                        u, spot_price, strike_price, time_to_expiry, heston_params
                    )
                
                # Return the real part of the integrand
                return np.real(char_func * np.exp(-1j * u * log(strike_price)) / (1j * u))
            except:
                return 0.0
        
        # Calculate the probabilities P1 and P2
        try:
            # Integrate with smaller upper limit for stability
            P1_integral, _ = quad(lambda u: integrand(u, 1), 0.0001, 50, limit=100, epsabs=1e-6)
            P2_integral, _ = quad(lambda u: integrand(u, 2), 0.0001, 50, limit=100, epsabs=1e-6)
            
            P1 = 0.5 + P1_integral / np.pi
            P2 = 0.5 + P2_integral / np.pi
            
            # Ensure probabilities are in [0,1]
            P1 = max(0, min(1, P1))
            P2 = max(0, min(1, P2))
            
            # Heston call price formula
            call_price = spot_price * P1 - strike_price * exp(-self.risk_free_rate * time_to_expiry) * P2
            
            # Ensure reasonable bounds
            max_price = spot_price  # Call can't be worth more than the stock
            min_price = max(0, spot_price - strike_price * exp(-self.risk_free_rate * time_to_expiry))
            
            call_price = max(min_price, min(max_price, call_price))
            
            if option_type == 'call':
                return call_price
            else:
                # Use put-call parity for put
                put_price = call_price - spot_price + strike_price * exp(-self.risk_free_rate * time_to_expiry)
                return max(0, put_price)
                
        except Exception as e:
            # Fallback to Black-Scholes
            vol = min(sqrt(heston_params.v0), 3.0)
            d1 = (log(spot_price/strike_price) + (self.risk_free_rate + 0.5*vol**2)*time_to_expiry) / (vol*sqrt(time_to_expiry))
            d2 = d1 - vol*sqrt(time_to_expiry)
            
            if option_type == 'call':
                return spot_price * norm.cdf(d1) - strike_price * exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2)
            else:
                return strike_price * exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)

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
        """
        Objective function for parameter calibration.
        
        This function computes the mean squared error between market implied
        volatilities and model implied volatilities for all options in the dataset.
        
        Parameters
        ----------
        parameter_vector : np.ndarray
            Array of Heston parameters [v0, kappa, theta, sigma, rho]
        market_option_data : pd.DataFrame
            DataFrame with market option data
        spot_price : float
            Current stock price
            
        Returns
        -------
        float
            Mean squared error between market and model implied volatilities
        """
        # Convert parameter vector to HestonParams object
        heston_params = HestonParams(
            v0=parameter_vector[0],
            kappa=parameter_vector[1],
            theta=parameter_vector[2],
            sigma=parameter_vector[3],
            rho=parameter_vector[4]
        )
        
        # Calculate model implied volatilities for all options
        squared_errors = []
        
        for _, option_row in market_option_data.iterrows():
            # Extract option characteristics
            strike_price = option_row['Strike']
            time_to_expiry = option_row['DaysToExpiry'] / 365.0  # Convert days to years
            market_implied_vol = option_row['ImpliedVolatility']
            option_type = option_row['OptionType']
            
            # Calculate model implied volatility
            model_implied_vol = self._calculate_model_implied_volatility(
                spot_price, strike_price, time_to_expiry, heston_params, option_type
            )
            
            # Add squared error if calculation was successful
            if not np.isnan(model_implied_vol):
                squared_error = (model_implied_vol - market_implied_vol)**2
                squared_errors.append(squared_error)
        
        # Return mean squared error, or large penalty if no valid calculations
        if squared_errors:
            return np.mean(squared_errors)
        else:
            return 1e6  # Large penalty for invalid parameter combinations

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