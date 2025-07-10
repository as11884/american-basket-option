"""
Calibration Utilities Module

This module provides utility functions and classes for managing
Heston model calibration workflows, including parameter management,
result storage, and batch processing utilities.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class HestonParams:
    """
    Container for Heston model parameters with validation.
    
    The Heston model has five parameters that control the stochastic volatility process:
    - v0: Initial variance level
    - theta: Long-term mean variance
    - kappa: Speed of mean reversion to long-term variance
    - σ (sigma): Volatility of the variance process ("vol of vol")
    - ρ (rho): Correlation between asset returns and variance innovations
    """
    
    v0: float                    # Initial variance
    theta: float                 # Long-term variance
    kappa: float                 # Mean reversion speed
    sigma: float                 # Volatility of volatility
    rho: float                   # Asset-vol correlation
    
    def __post_init__(self):
        """Validate parameter values after initialization."""
        self.validate_parameters()
    
    def validate_parameters(self) -> None:
        """
        Validate Heston parameter values to ensure they satisfy model constraints.
        
        Raises
        ------
        ValueError
            If parameters violate Heston model constraints
        """
        if self.v0 <= 0:
            raise ValueError(f"Initial variance must be positive, got {self.v0}")
        
        if self.theta <= 0:
            raise ValueError(f"Long-term variance must be positive, got {self.theta}")
        
        if self.kappa <= 0:
            raise ValueError(f"Mean reversion speed must be positive, got {self.kappa}")
        
        if self.sigma <= 0:
            raise ValueError(f"Volatility of volatility must be positive, got {self.sigma}")
        
        if not -1 <= self.rho <= 1:
            raise ValueError(f"Correlation must be between -1 and 1, got {self.rho}")
        
        # Feller condition: 2*kappa*theta > sigma^2
        feller_condition = 2 * self.kappa * self.theta
        if feller_condition <= self.sigma**2:
            print(f"Warning: Feller condition violated. 2κθ = {feller_condition:.4f} <= σ² = {self.sigma**2:.4f}")
            print("This may lead to the variance process hitting zero.")
    
    def to_dict(self) -> Dict[str, float]:
        """Convert parameters to dictionary format."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, param_dict: Dict[str, float]) -> 'HestonParams':
        """Create HestonParams instance from dictionary."""
        return cls(**param_dict)
    
    def __str__(self) -> str:
        """String representation of parameters."""
        return (f"HestonParams(v0={self.v0:.6f}, theta={self.theta:.6f}, "
                f"kappa={self.kappa:.6f}, sigma={self.sigma:.6f}, rho={self.rho:.6f})")


@dataclass
class CalibrationResult:
    """
    Container for calibration results and metadata.
    """
    
    ticker: str
    calibration_date: str
    parameters: HestonParams
    spot_price: float
    risk_free_rate: float
    objective_value: float
    optimization_success: bool
    n_options_used: int
    calibration_time_seconds: float
    market_data_summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert calibration result to dictionary."""
        result_dict = asdict(self)
        result_dict['parameters'] = self.parameters.to_dict()
        return result_dict
    
    @classmethod
    def from_dict(cls, result_dict: Dict[str, Any]) -> 'CalibrationResult':
        """Create CalibrationResult from dictionary."""
        params = HestonParams.from_dict(result_dict['parameters'])
        result_dict['parameters'] = params
        return cls(**result_dict)


class CalibrationManager:
    """
    Manager class for handling calibration workflows and result storage.
    """
    
    def __init__(self, results_directory: str = "calibration_results"):
        """
        Initialize calibration manager.
        
        Parameters
        ----------
        results_directory : str
            Directory to store calibration results
        """
        self.results_directory = results_directory
        self.ensure_results_directory()
    
    def ensure_results_directory(self) -> None:
        """Create results directory if it doesn't exist."""
        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory)
    
    def save_calibration_result(self, result: CalibrationResult) -> str:
        """
        Save calibration result to file.
        
        Parameters
        ----------
        result : CalibrationResult
            Calibration result to save
        
        Returns
        -------
        str
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.ticker}_calibration_{timestamp}.json"
        filepath = os.path.join(self.results_directory, filename)
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        return filepath
    
    def load_calibration_result(self, filepath: str) -> CalibrationResult:
        """
        Load calibration result from file.
        
        Parameters
        ----------
        filepath : str
            Path to calibration result file
        
        Returns
        -------
        CalibrationResult
            Loaded calibration result
        """
        with open(filepath, 'r') as f:
            result_dict = json.load(f)
        
        return CalibrationResult.from_dict(result_dict)
    
    def list_calibration_results(self, ticker: Optional[str] = None) -> List[str]:
        """
        List available calibration result files.
        
        Parameters
        ----------
        ticker : str, optional
            Filter results by ticker symbol
        
        Returns
        -------
        List[str]
            List of calibration result file paths
        """
        files = []
        for filename in os.listdir(self.results_directory):
            if filename.endswith('.json'):
                if ticker is None or filename.startswith(ticker):
                    files.append(os.path.join(self.results_directory, filename))
        
        return sorted(files)
    
    def get_latest_calibration(self, ticker: str) -> Optional[CalibrationResult]:
        """
        Get the most recent calibration result for a ticker.
        
        Parameters
        ----------
        ticker : str
            Ticker symbol
        
        Returns
        -------
        CalibrationResult or None
            Latest calibration result, or None if not found
        """
        files = self.list_calibration_results(ticker)
        if not files:
            return None
        
        latest_file = files[-1]  # Files are sorted by timestamp
        return self.load_calibration_result(latest_file)


def load_calibrated_parameters(ticker: str, 
                             results_directory: str = "calibration_results") -> Optional[HestonParams]:
    """
    Convenience function to load the latest calibrated parameters for a ticker.
    
    Parameters
    ----------
    ticker : str
        Ticker symbol
    results_directory : str
        Directory containing calibration results
    
    Returns
    -------
    HestonParams or None
        Latest calibrated parameters, or None if not found
    """
    manager = CalibrationManager(results_directory)
    result = manager.get_latest_calibration(ticker)
    
    if result is None:
        return None
    
    return result.parameters


def create_parameter_bounds() -> List[Tuple[float, float]]:
    """
    Create reasonable parameter bounds for Heston model optimization.
    
    Returns
    -------
    List[Tuple[float, float]]
        Parameter bounds in order: (v0, kappa, theta, sigma, rho)
    """
    return [
        (1e-6, 2.0),     # v0: initial variance (allow up to 141% vol)
        (0.1, 20.0),     # kappa: mean reversion speed (faster range)
        (1e-6, 2.0),     # theta: long-term variance (allow up to 141% vol)
        (0.01, 1.0),     # sigma: vol of vol (more conservative)
        (-0.95, 0.95)    # rho: correlation (slightly less extreme)
    ]


def validate_option_data(option_data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean option data for calibration.
    
    Parameters
    ----------
    option_data : pd.DataFrame
        Option data to validate
    
    Returns
    -------
    pd.DataFrame
        Cleaned option data
    """
    required_columns = ['Strike', 'DaysToExpiry', 'ImpliedVolatility', 'OptionType']
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in option_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Remove rows with missing or invalid data
    clean_data = option_data.copy()
    clean_data = clean_data.dropna(subset=required_columns)
    
    # Remove options with invalid implied volatility
    clean_data = clean_data[clean_data['ImpliedVolatility'] > 0]
    clean_data = clean_data[clean_data['ImpliedVolatility'] < 5.0]  # Remove extreme IVs
    
    # Remove options with invalid time to expiry
    clean_data = clean_data[clean_data['DaysToExpiry'] > 0]
    
    # Remove options with invalid strikes
    clean_data = clean_data[clean_data['Strike'] > 0]
    
    return clean_data


def calculate_model_errors(market_ivs: np.ndarray, 
                         model_ivs: np.ndarray) -> Dict[str, float]:
    """
    Calculate various error metrics between market and model implied volatilities.
    
    Parameters
    ----------
    market_ivs : np.ndarray
        Market implied volatilities
    model_ivs : np.ndarray
        Model implied volatilities
    
    Returns
    -------
    Dict[str, float]
        Dictionary of error metrics
    """
    errors = model_ivs - market_ivs
    
    return {
        'mse': np.mean(errors**2),
        'rmse': np.sqrt(np.mean(errors**2)),
        'mae': np.mean(np.abs(errors)),
        'max_abs_error': np.max(np.abs(errors)),
        'mean_error': np.mean(errors),
        'std_error': np.std(errors)
    }


def print_calibration_summary(result: CalibrationResult) -> None:
    """
    Print a formatted summary of calibration results.
    
    Parameters
    ----------
    result : CalibrationResult
        Calibration result to summarize
    """
    print(f"\n{'='*60}")
    print(f"CALIBRATION SUMMARY: {result.ticker}")
    print(f"{'='*60}")
    print(f"Date: {result.calibration_date}")
    print(f"Spot Price: ${result.spot_price:.2f}")
    print(f"Risk-free Rate: {result.risk_free_rate:.3f}")
    print(f"Options Used: {result.n_options_used}")
    print(f"Calibration Time: {result.calibration_time_seconds:.2f} seconds")
    print(f"Optimization Success: {result.optimization_success}")
    print(f"Final Objective Value: {result.objective_value:.6f}")
    
    print(f"\nCalibrated Parameters:")
    print(f"  Initial Variance (v0): {result.parameters.v0:.6f}")
    print(f"  Long-term Variance (θ): {result.parameters.theta:.6f}")
    print(f"  Mean Reversion (κ): {result.parameters.kappa:.6f}")
    print(f"  Vol of Vol (σ): {result.parameters.sigma:.6f}")
    print(f"  Correlation (ρ): {result.parameters.rho:.6f}")
    
    print(f"\nMarket Data Summary:")
    for key, value in result.market_data_summary.items():
        print(f"  {key}: {value}")
    
    print(f"{'='*60}")
