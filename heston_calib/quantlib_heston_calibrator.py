import numpy as np
import pandas as pd
import QuantLib as ql
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
from scipy.optimize import differential_evolution

# Handle both relative and absolute imports
try:
    from .market_data_fetcher import MarketDataFetcher
except ImportError:
    from market_data_fetcher import MarketDataFetcher


class QuantLibHestonCalibrator:
    
    def __init__(self, r: float = 0.05, q: float = 0.0, calculation_date: Optional[datetime] = None):
        """Initialize Heston calibrator with risk-free rate, dividend yield, and evaluation date."""
        self.r = r
        self.q = q
        self.calculation_date = calculation_date or datetime.now()

        # Setup QuantLib environment
        self.calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        self.evaluation_date = ql.Date(self.calculation_date.day, self.calculation_date.month, self.calculation_date.year)
        ql.Settings.instance().evaluationDate = self.evaluation_date
        
        # Setup yield curves
        self.day_count = ql.Actual365Fixed()
        
        # Create handles for the curves
        self.yield_ts = ql.YieldTermStructureHandle(ql.FlatForward(self.evaluation_date, r, self.day_count))
        self.dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(self.evaluation_date, q, self.day_count))

    def _setup_model(self, spot: float, init_condition: Tuple[float, float, float, float, float] = (0.02, 0.2, 0.5, 0.1, 0.01)) -> Tuple[ql.HestonModel, ql.AnalyticHestonEngine]:
        """Setup Heston model and engine with initial parameters."""
        theta, kappa, sigma, rho, v0 = init_condition
        
        process = ql.HestonProcess(
            self.yield_ts, 
            self.dividend_ts,
            ql.QuoteHandle(ql.SimpleQuote(spot)), 
            v0, kappa, theta, sigma, rho
        )
        
        model = ql.HestonModel(process)
        engine = ql.AnalyticHestonEngine(model)
        
        return model, engine
    
    def _setup_helpers(self, engine: ql.AnalyticHestonEngine, market_data: pd.DataFrame, spot: float) -> Tuple[list, list]:
        """Create QuantLib Heston helpers from market data - using Yahoo Finance IV directly."""
        helpers = []
        grid_data = []
        
        for _, row in market_data.iterrows():

            # Basic data quality filters
            if (pd.isna(row.MarketPrice) or row.MarketPrice <= 0 or
                row.DaysToExpiry <= 0):
                continue
            
            # Use implied volatility from Yahoo Finance directly
            if 'ImpliedVolatility' in row and pd.notna(row.ImpliedVolatility):
                impl_vol = float(row.ImpliedVolatility)
            else:
                continue  # Skip if no IV available
                            
            # Create QuantLib period and helper
            period = ql.Period(int(row.DaysToExpiry), ql.Days)
            
            helper = ql.HestonModelHelper(
                period,
                self.calendar,
                spot,                                    
                float(row.Strike),
                ql.QuoteHandle(ql.SimpleQuote(impl_vol)),  # Use Yahoo Finance IV
                self.yield_ts,
                self.dividend_ts
            )

            helper.setPricingEngine(engine)
            helpers.append(helper)
            
            # Store grid data for reporting
            expiry_date = self.evaluation_date + period
            grid_data.append((expiry_date, row.Strike))
                        
        print(f"Created {len(helpers)} valid helpers from {len(market_data)} options")
        
        # Additional check: ensure we have enough helpers for calibration
        if len(helpers) < 5:
            print(f"Warning: Only {len(helpers)} helpers created - may not be sufficient for robust calibration")
        
        return helpers, grid_data
    
    def _calculate_implied_vol(self, option_price: float, spot: float, strike: float, 
                              time_to_expiry: float, option_type: str) -> Optional[float]:
        """Calculate implied volatility from option price using QuantLib's implied volatility solver."""
        try:
            # Simple Black-Scholes implied volatility calculation
            from scipy.optimize import brentq
            
            # Define Black-Scholes pricing function
            def black_scholes_price(vol):
                try:
                    from math import log, exp, sqrt
                    from scipy.stats import norm
                    
                    d1 = (log(spot / strike) + (self.r - self.q + 0.5 * vol**2) * time_to_expiry) / (vol * sqrt(time_to_expiry))
                    d2 = d1 - vol * sqrt(time_to_expiry)
                    
                    if option_type.lower() == 'call':
                        price = spot * exp(-self.q * time_to_expiry) * norm.cdf(d1) - strike * exp(-self.r * time_to_expiry) * norm.cdf(d2)
                    else:
                        price = strike * exp(-self.r * time_to_expiry) * norm.cdf(-d2) - spot * exp(-self.q * time_to_expiry) * norm.cdf(-d1)
                    
                    return price
                except:
                    return float('inf')
            
            # Objective function for root finding
            def objective(vol):
                return black_scholes_price(vol) - option_price
            
            # Use Brent's method to find the implied volatility
            try:
                implied_vol = brentq(objective, 0.001, 5.0, xtol=1e-6, maxiter=100)
                return implied_vol if 0.01 < implied_vol < 5.0 else None
            except:
                return None
                
        except Exception as e:
            return None

    def _cost_function_generator(self, model: ql.HestonModel, helpers: list, norm: bool = True):
        """Generate cost function for scipy optimization"""
        def cost_function(params):            
            # Set model parameters
            params_array = ql.Array(list(params))
            model.setParams(params_array)
            # Calculate calibration errors for each helper 
            errors = [h.calibrationError() for h in helpers]
            if norm:
                return np.sqrt(np.sum(np.abs(errors)))
            else:
                return errors
        return cost_function
   
    def _calibration_report(self, helpers: list, grid_data: list, detailed: bool = False) -> float:
        """Generate calibration report showing fit quality - following Goutham Balaraman's approach."""
        avg_error = 0.0
        if detailed:
            print(f"{'Strikes':<15} {'Expiry':<25} {'Market Value':<15} {'Model Value':<15} {'Relative Error (%)':<20}")
            print("=" * 100)
        
        for i, helper in enumerate(helpers):
            market_val = helper.marketValue()
            model_val = helper.modelValue()
            if market_val > 0:
                rel_error = (model_val / market_val - 1.0)
                avg_error += abs(rel_error)
                
                if detailed:
                    date, strike = grid_data[i]
                    print(f"{strike:<15.2f} {str(date):<25} {market_val:<14.5f} {model_val:<15.5f} {100.0 * rel_error:<20.7f}")

        # Convert to percentage and average
        avg_error = avg_error * 100.0 / len(helpers) if helpers else 0.0
        if detailed:
            print("-" * 100)
        return avg_error
    
    def calibrate(self, spot: float, market_data: pd.DataFrame, 
                 maxiter: int = 1000, detailed_report: bool = False) -> Tuple[ql.HestonModel, Dict[str, Any]]:
        """
        Calibrate Heston model using scipy differential evolution global optimizer.
        
        Parameters:
        -----------
        spot : float
            Current stock price
        market_data : pd.DataFrame
            Market data with columns: Strike, MarketPrice, DaysToExpiry, OptionType
        maxiter : int
            Maximum iterations for differential evolution (default 100)
        detailed_report : bool
            Whether to print detailed calibration report
            
        Returns:
        --------
        Tuple[ql.HestonModel, Dict[str, Any]]
            Calibrated model and calibration information
        """
        print(f"Starting Heston calibration using Differential Evolution...")
        print(f"Market data: {len(market_data)} options")
        
        # Try multiple diverse initial conditions for robust global optimization
        # These are realistic starting points based on typical equity market parameters
        initial_conditions = [   
            (0.09, 2.0, 0.3, -0.7, 0.04),   # High Vol: 30% vol, moderate mean reversion
            # (0.01, 5.0, 0.6, -0.8, 0.02),   # Low vol: 10% vol, fast mean reversion
        ]
        
        best_result = None
        best_cost = float('inf')
        best_initial = None
        
        for init_condition in initial_conditions:
            print(f"Trying initial condition: theta={init_condition[0]:.3f}, kappa={init_condition[1]:.3f}, "
                  f"sigma={init_condition[2]:.3f}, rho={init_condition[3]:.3f}, v0={init_condition[4]:.3f}")
            
            # Setup model with this initial condition
            model, engine = self._setup_model(spot, init_condition)
            
            # Create helpers for this setup
            helpers, grid_data = self._setup_helpers(engine, market_data, spot)
            
            if not helpers:
                continue
                
            # Parameter bounds - following Goutham Balaraman's exact approach
            # Order: theta, kappa, sigma, rho, v0
            bounds = [
                (0.01, 1.0),    # theta: long-term variance (0.01 to 1.0)
                (0.01, 15.0),   # kappa: mean reversion speed (0.01 to 15.0)
                (0.01, 4),    # sigma: volatility of volatility (0.01 to 1.5)
                (-0.99, 0.99),  # rho: correlation (-0.99 to 0.99)
                (0.01, 1)     # v0: initial variance (0.01 to 1.0)
            ]
            
            # Create cost function
            cost_function = self._cost_function_generator(model, helpers, norm=True)
            
            # Test the cost function with initial parameters to ensure it's working
            initial_params = list(init_condition)
            initial_cost = cost_function(initial_params)
            print(f"  → Initial cost with these parameters: {initial_cost:.6f}")
            
            try:
                # Run differential evolution with this initial condition
                result = differential_evolution(
                    cost_function, 
                    bounds, 
                    maxiter=maxiter,
                    popsize=15,        # Slightly larger population
                    atol=1e-6,         # Tighter tolerance
                    tol=1e-6,          # Tighter tolerance  
                    workers=1,         # Single threaded for stability
                    polish=True        # Local optimization at the end
                )
                
                # Accept result if cost is reasonable, even if maxiter exceeded
                cost_acceptable = result.fun < best_cost  # Reasonable cost threshold
                print(f"  → Result: fun={result.fun:.6f}, nfev={result.nfev}")

                if cost_acceptable:
                    best_result = result
                    best_cost = result.fun
                    best_initial = init_condition
                    print(f"  → NEW BEST solution! Cost: {result.fun:.6f}")
                else:
                    print(f"  → Cost: {result.fun:.6f} (not improved from {best_cost:.6f})")
                    
            except Exception as e:
                print(f"  → Exception: {str(e)}")
                continue
        
        # If no valid result found, return empty model and failure info
        if best_result is None:
            return model, {
                'success': False,
                'error': 'All initial conditions failed',
                'calibrated_params': {},
                'average_error': float('inf'),
                'num_helpers': 0
            }
        
        # Use the best result found
        if best_result:
            # Extract calibrated parameters
            theta, kappa, sigma, rho, v0 = best_result.x
            model, engine = self._setup_model(spot, best_result.x)
            helpers, grid_data = self._setup_helpers(engine, market_data, spot)

            # Calculate final calibration error
            avg_error = self._calibration_report(helpers, grid_data, detailed=detailed_report)
            
            print(f"\nCalibration successful with initial condition: {best_initial}")
            print(f"Iterations: {best_result.nit}, Function evaluations: {best_result.nfev}")
            print(f"Final cost: {best_result.fun:.6f}")

            calibration_info = {
                'success': True,
                'calibrated_params': {
                    'theta': theta,
                    'kappa': kappa, 
                    'sigma': sigma,
                    'rho': rho,
                    'v0': v0
                },
                'average_error': avg_error,
                'num_helpers': len(helpers),
                'best_initial_condition': best_initial,
                'multi_start_used': True,
                'optimization_result': {
                    'fun': result.fun,
                    'nit': result.nit,
                    'nfev': result.nfev,
                    'message': result.message
                }
            }
            
        else:
            calibration_info = {
                'success': False,
                'error': f"Optimization failed: {result.message}",
                'calibrated_params': {},
                'average_error': float('inf'),
                'num_helpers': len(helpers) if helpers else 0,
                'best_initial_condition': best_initial,
                'multi_start_used': True
            }
            print(f"Calibration failed: {result.message}")
                
        return model, calibration_info
    
    def print_results(self, calibration_info: Dict[str, Any]):
        """Print calibration results in a formatted table."""
        if not calibration_info['success']:
            print(f"Calibration failed: {calibration_info.get('error', 'Unknown error')}")
            return
            
        params = calibration_info['calibrated_params']
        
        print(f"\nHESTON CALIBRATION RESULTS")
        print(f"{'='*50}")
        print(f"Options used: {calibration_info['num_helpers']}, Average error: {calibration_info['average_error']:.4f}%")
        print(f"\nParameters:")
        print(f"  theta (θ): {params['theta']:.4f}  |  kappa (κ): {params['kappa']:.4f}")
        print(f"  sigma (σ): {params['sigma']:.4f}  |  rho (ρ):   {params['rho']:.4f}")
        print(f"  v0:        {params['v0']:.4f}")
        print(f"\nVolatilities: Initial {np.sqrt(params['v0']):.2%}, Long-term {np.sqrt(params['theta']):.2%}")

# Quick test
if __name__ == '__main__':
    # Test the calibrator

    print("Testing QuantLib Heston Calibrator with Differential Evolution...")
    
    # Fetch market data
    fetcher = MarketDataFetcher(ticker='NVDA', expiry_list=['1M', '2M', '3M'], atm_range=0.1)
    market_data = fetcher.prepare_market_data()
    spot = fetcher.get_spot_price()
    
    print(f"Spot price: ${spot:.2f}")
    print(f"Market data shape: {market_data.shape}")
    
    # Create and run calibrator with more generous settings
    calibrator = QuantLibHestonCalibrator(r=0.05, q=0.0)
    model, info = calibrator.calibrate(
        spot=spot, 
        market_data=market_data, 
        maxiter=1000,  # More iterations for real data
        detailed_report=False
    )
    
    # Print results
    calibrator.print_results(info)
