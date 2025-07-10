"""
Heston Stochastic Volatility Model Implementation

This module implements the Heston model for simulating stock price paths
where the volatility follows a stochastic process (CIR process).

The Heston model is defined by:
- dS(t) = r * S(t) * dt + sqrt(V(t)) * S(t) * dW1(t)
- dV(t) = κ * (θ - V(t)) * dt + σ * sqrt(V(t)) * dW2(t)
- where Corr(dW1, dW2) = ρ

Parameters:
- v0: Initial variance
- θ (theta): Long-term mean variance
- κ (kappa): Speed of mean reversion
- σ (sigma): Volatility of volatility
- ρ (rho): Correlation between asset and volatility shocks
"""

import numpy as np
from calib_utils import HestonParams


class HestonModel:
    """
    Heston stochastic volatility model for simulating stock price paths.
    
    The model assumes that volatility follows a mean-reverting square-root process
    (Cox-Ingersoll-Ross process) that is correlated with the stock price process.
    """
    
    def __init__(self, heston_parameters: HestonParams, risk_free_rate: float = 0.03, 
                 time_step: float = 1/252):
        """
        Initialize the Heston model with calibrated parameters.
        
        Parameters
        ----------
        heston_parameters : HestonParams
            Object containing the five Heston model parameters
        risk_free_rate : float, default=0.03
            Annual risk-free interest rate (3% default)
        time_step : float, default=1/252
            Time step for simulation (daily by default, 1/252 years)
        """
        # Store model parameters with descriptive names
        self.heston_params = heston_parameters
        self.risk_free_rate = risk_free_rate
        self.time_step = time_step
        
        # Validate parameters
        heston_parameters.validate_parameters()
        
        # Extract individual parameters for easier access
        self.initial_variance = heston_parameters.v0
        self.long_term_variance = heston_parameters.theta
        self.mean_reversion_speed = heston_parameters.kappa
        self.volatility_of_volatility = heston_parameters.sigma
        self.correlation_asset_vol = heston_parameters.rho

    def generate_paths(self, initial_stock_price, time_horizon, num_simulation_paths=1000):
        """
        Generate Monte Carlo paths for stock prices using the Heston model.
        
        This method uses the Euler-Maruyama discretization scheme with full truncation
        to ensure variance remains non-negative.
        
        Parameters
        ----------
        initial_stock_price : float
            Starting stock price (S0)
        time_horizon : float
            Total simulation time in years (T)
        num_simulation_paths : int, default=1000
            Number of Monte Carlo paths to generate
            
        Returns
        -------
        np.ndarray
            Array of shape (num_simulation_paths, num_time_steps) containing
            simulated stock price paths
        """
        # Calculate number of time steps
        num_time_steps = int(time_horizon / self.time_step) + 1
        
        # Initialize arrays to store stock prices and variances
        # Shape: (number of paths, number of time steps)
        stock_price_paths = np.zeros((num_simulation_paths, num_time_steps))
        variance_paths = np.zeros((num_simulation_paths, num_time_steps))
        
        # Set initial conditions
        stock_price_paths[:, 0] = initial_stock_price
        variance_paths[:, 0] = self.initial_variance
        
        # Precompute square root of time step for efficiency
        sqrt_time_step = np.sqrt(self.time_step)
        
        # Generate independent standard normal random variables
        # Z1 drives the variance process, Z2 drives the stock price process
        random_variance_shocks = np.random.normal(
            size=(num_simulation_paths, num_time_steps - 1)
        )
        random_price_shocks = np.random.normal(
            size=(num_simulation_paths, num_time_steps - 1)
        )
        
        # Create correlated random variables for stock price process
        # This implements the correlation structure between asset and volatility
        correlated_price_shocks = (
            self.correlation_asset_vol * random_variance_shocks +
            np.sqrt(1 - self.correlation_asset_vol**2) * random_price_shocks
        )
        
        # Simulate paths using Euler-Maruyama discretization
        for time_step_idx in range(num_time_steps - 1):
            # Current variance values (ensure non-negative using full truncation)
            current_variance = np.maximum(variance_paths[:, time_step_idx], 0)
            
            # Update variance using CIR process with full truncation
            # dV = κ(θ - V)dt + σ√V dW1
            variance_drift = (
                self.mean_reversion_speed * 
                (self.long_term_variance - current_variance) * 
                self.time_step
            )
            variance_diffusion = (
                self.volatility_of_volatility * 
                np.sqrt(current_variance) * 
                random_variance_shocks[:, time_step_idx] * 
                sqrt_time_step
            )
            
            variance_paths[:, time_step_idx + 1] = (
                variance_paths[:, time_step_idx] + 
                variance_drift + 
                variance_diffusion
            )
            
            # Update stock price using geometric Brownian motion with stochastic volatility
            # dS = rS dt + √V S dW2
            stock_price_drift = (
                (self.risk_free_rate - 0.5 * current_variance) * 
                self.time_step
            )
            stock_price_diffusion = (
                np.sqrt(current_variance) * 
                correlated_price_shocks[:, time_step_idx] * 
                sqrt_time_step
            )
            
            # Use exponential form to ensure stock prices remain positive
            stock_price_paths[:, time_step_idx + 1] = (
                stock_price_paths[:, time_step_idx] * 
                np.exp(stock_price_drift + stock_price_diffusion)
            )
        
        return stock_price_paths