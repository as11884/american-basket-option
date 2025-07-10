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
from math import log, sqrt
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
        
        # Complex coefficients for the characteristic function
        # These come from the analytical solution of the Heston PDE
        alpha = -0.5  # Related to the integration measure
        beta = alpha - 1j * correlation * volatility_of_volatility * frequency_points
        gamma = 0.5 * volatility_of_volatility * volatility_of_volatility
        
        # Discriminant of the quadratic in the characteristic function
        discriminant = np.sqrt(beta**2 - 4 * alpha * gamma * frequency_points * (frequency_points + 1j))
        
        # Ratio appearing in the characteristic function
        ratio_g = (beta - discriminant) / (beta + discriminant)
        
        # Time-dependent exponential term
        exponential_term = np.exp(-discriminant * mean_reversion_speed * time_to_expiry)
        
        # Complex logarithm terms
        complex_integral_1 = ratio_g * (1 - exponential_term) / (1 - ratio_g * exponential_term)
        complex_integral_2 = (
            np.log((1 - ratio_g * exponential_term) / (1 - ratio_g)) - 
            mean_reversion_speed * time_to_expiry
        )
        
        # Assemble the characteristic function
        # This is the analytical solution for the Heston model
        characteristic_function = np.exp(
            1j * frequency_points * log_moneyness +
            1j * frequency_points * (self.risk_free_rate - 0.5 * initial_variance) * time_to_expiry +
            initial_variance * mean_reversion_speed * long_term_variance * complex_integral_2 / (volatility_of_volatility * volatility_of_volatility) +
            initial_variance * complex_integral_1 / (volatility_of_volatility * volatility_of_volatility)
        )
        
        return characteristic_function

    def _price_option_fft(self, spot_price, strike_price, time_to_expiry, heston_params, option_type='call'):
        """
        Price European options using Fast Fourier Transform.
        
        This method uses FFT to efficiently compute option prices from the
        characteristic function. It's much faster than Monte Carlo but gives
        approximate results.
        
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
        option_type : str, default='call'
            Type of option ('call' or 'put')
            
        Returns
        -------
        float
            Option price
        """
        # FFT parameters - these control the accuracy and range of the FFT
        num_fft_points = 2**10  # Must be power of 2 for efficient FFT
        integration_upper_bound = 500  # Upper bound for integration
        
        # Grid spacing in log-strike and frequency domains
        log_strike_spacing = integration_upper_bound / num_fft_points
        frequency_spacing = 2 * np.pi / (num_fft_points * log_strike_spacing)
        
        # Frequency grid points
        frequency_grid = np.arange(num_fft_points) * frequency_spacing
        
        # Evaluate characteristic function at frequency grid points
        characteristic_function_values = self._heston_characteristic_function(
            frequency_grid - 0.5j, spot_price, strike_price, time_to_expiry, heston_params
        )
        
        # Damping factor to ensure convergence
        log_strike_origin = -0.5 * num_fft_points * log_strike_spacing
        damping_multiplier = np.exp(1j * frequency_grid * log_strike_origin) / (frequency_grid * frequency_grid + 0.25)
        
        # Prepare input for FFT
        fft_input = characteristic_function_values * damping_multiplier * frequency_spacing
        
        # Special treatment for zero frequency to avoid division by zero
        fft_input[0] = characteristic_function_values[0] * frequency_spacing * 0.5
        
        # Perform FFT to get option values
        fft_output = np.fft.fft(fft_input).real
        
        # Extract option price at the desired strike
        log_strike = log(strike_price)
        strike_index = int((log_strike - log_strike_origin) / log_strike_spacing)
        
        # Check if strike is within the grid range
        if 0 <= strike_index < num_fft_points:
            # Get call price from FFT output
            call_price = np.exp(-log_strike) * fft_output[strike_index] / np.pi
        else:
            # Fallback to intrinsic value if strike is out of range
            call_price = max(0, spot_price - strike_price * np.exp(-self.risk_free_rate * time_to_expiry))
        
        # Convert to put price using put-call parity if needed
        if option_type == 'call':
            return max(0.0, call_price)
        else:
            # Put-call parity: Put = Call - Spot + PV(Strike)
            put_price = call_price + strike_price * np.exp(-self.risk_free_rate * time_to_expiry) - spot_price
            return max(0.0, put_price)

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