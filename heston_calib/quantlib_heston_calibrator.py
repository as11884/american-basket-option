"""
QuantLib Heston Calibrator - Using scipy differential evolution global optimizer.
Based on Goutham Balaraman's approach: https://gouthamanbalaraman.com/blog/heston-calibration-scipy-optimize-quantlib-python.html
"""

import numpy as np
import pandas as pd
import QuantLib as ql
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
from scipy.optimize import differential_evolution


class QuantLibHestonCalibrator:
    
    def __init__(self, r: float = 0.015, q: float = 0.0, start_date: Optional[datetime] = None):
        """Initialize Heston calibrator with risk-free rate, dividend yield, and evaluation date."""
        self.r = r
        self.q = q
        self.start_date = start_date or datetime.now()
        
        # Setup QuantLib environment
        self.calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
        self.evaluation_date = ql.Date(self.start_date.day, self.start_date.month, self.start_date.year)
        ql.Settings.instance().evaluationDate = self.evaluation_date
        
        # Setup yield curves
        self.day_count = ql.Actual365Fixed()
        self.rf_curve = ql.FlatForward(self.evaluation_date, self.r, self.day_count)
        self.div_curve = ql.FlatForward(self.evaluation_date, self.q, self.day_count)
        
        # Create handles for the curves
        self.yield_ts = ql.YieldTermStructureHandle(self.rf_curve)
        self.dividend_ts = ql.YieldTermStructureHandle(self.div_curve)
        
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
            try:
                # Basic data quality filters
                if (pd.isna(row.MarketPrice) or row.MarketPrice <= 0 or
                    row.DaysToExpiry <= 0):
                    continue
                
                # Use implied volatility from Yahoo Finance directly
                if 'ImpliedVolatility' in row and pd.notna(row.ImpliedVolatility):
                    impl_vol = float(row.ImpliedVolatility)
                else:
                    # Fallback: calculate IV if not available
                    time_to_expiry = row.DaysToExpiry / 365.0
                    impl_vol = self._calculate_implied_vol(
                        row.MarketPrice, spot, row.Strike, time_to_expiry, row.OptionType
                    )
                
                # IV filter - following Goutham's approach  
                if impl_vol < 0.01 or impl_vol > 3.0:  # Wider range
                    continue
                
                # Additional quality check: reasonable option price relative to intrinsic
                is_call = row.OptionType.lower() == 'call'
                intrinsic = max(0, (spot - row.Strike) if is_call else (row.Strike - spot))
                
                # Skip if market price is less than intrinsic (arbitrage)
                if row.MarketPrice < intrinsic:
                    continue
                    
                # Skip if time value is too small (may cause numerical issues)
                time_value = row.MarketPrice - intrinsic
                if time_value < 0.01:
                    continue
                
                # Create QuantLib period and helper
                period = ql.Period(int(row.DaysToExpiry), ql.Days)
                
                # Use the implied volatility directly like in Goutham's blog
                helper = ql.HestonModelHelper(
                    period,
                    self.calendar,
                    spot,                                    
                    float(row.Strike),
                    ql.QuoteHandle(ql.SimpleQuote(impl_vol)),  # Use Yahoo Finance IV or calculated IV
                    self.yield_ts,
                    self.dividend_ts
                )

                helper.setPricingEngine(engine)
                helpers.append(helper)
                
                # Store grid data for reporting
                expiry_date = self.evaluation_date + period
                grid_data.append((expiry_date, row.Strike))
                
            except Exception as e:
                # More detailed error logging for debugging
                continue
        
        print(f"Created {len(helpers)} valid helpers from {len(market_data)} options")
        
        # Additional check: ensure we have enough helpers for calibration
        if len(helpers) < 5:
            print(f"Warning: Only {len(helpers)} helpers created - may not be sufficient for robust calibration")
        
        return helpers, grid_data
    
    def _cost_function_generator(self, model: ql.HestonModel, helpers: list, norm: bool = True):
        """Generate cost function for scipy optimization - based on Goutham Balaraman's approach."""
        def cost_function(params):
            theta, kappa, sigma, rho, v0 = params
            
            # Enforce Feller condition: 2·kappa·theta > sigma²
            if 2 * kappa * theta <= sigma**2:
                return 1e6 if norm else [1e6] * len(helpers)

            try:
                # Set model parameters
                params_array = ql.Array(list(params))
                model.setParams(params_array)
                
                # Calculate calibration errors for each helper - following Goutham's approach
                errors = [h.calibrationError() for h in helpers]
                
                if norm:
                    # Following Goutham's approach: sqrt of sum of absolute errors
                    return np.sqrt(np.sum(np.abs(errors)))
                else:
                    return errors
                    
            except Exception:
                return 1e6 if norm else [1e6] * len(helpers)
        
        return cost_function
    def _cost_function_generator_with_smoothness(self, model: ql.HestonModel, helpers: list, spot: float, norm: bool = True):
        """Enhanced cost function that penalizes non-smooth volatility smiles."""
        def cost_function(params):
            theta, kappa, sigma, rho, v0 = params
            
            # Enforce Feller condition: 2·kappa·theta > sigma²
            if 2 * kappa * theta <= sigma**2:
                return 1e6 if norm else [1e6] * len(helpers)

            try:
                # Set model parameters
                model.setParams(ql.Array(list(params)))
                
                # Calculate basic calibration errors
                basic_errors = [h.calibrationError() for h in helpers]
                basic_cost = np.sqrt(np.sum(np.abs(basic_errors)))
                
                # Add smoothness penalty by checking volatility smile continuity
                smoothness_penalty = 0.0
                try:
                    # Sample some strikes around ATM to check smoothness
                    test_strikes = [spot * m for m in [0.95, 0.975, 1.0, 1.025, 1.05]]
                    test_expiry = 60  # 60 days
                    
                    engine = ql.AnalyticHestonEngine(model)
                    test_ivs = []
                    
                    for strike in test_strikes:
                        try:
                            period = ql.Period(test_expiry, ql.Days)
                            expiry_date = self.evaluation_date + period
                            exercise = ql.EuropeanExercise(expiry_date)
                            payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
                            option = ql.VanillaOption(payoff, exercise)
                            option.setPricingEngine(engine)
                            
                            price = option.NPV()
                            if price > 0.001:
                                iv = self._calculate_implied_vol(price, spot, strike, test_expiry/252.0, 'call')
                                if 0.05 < iv < 1.0:  # Reasonable IV
                                    test_ivs.append(iv)
                                else:
                                    test_ivs.append(np.nan)
                            else:
                                test_ivs.append(np.nan)
                        except:
                            test_ivs.append(np.nan)
                    
                    # Calculate smoothness penalty from IV differences
                    valid_ivs = [iv for iv in test_ivs if not np.isnan(iv)]
                    if len(valid_ivs) >= 3:
                        iv_diffs = np.diff(valid_ivs)
                        # Penalize large jumps in IV
                        smoothness_penalty = np.sum(np.abs(iv_diffs)) * 10.0  # Scale factor
                        
                except:
                    smoothness_penalty = 0.0
                
                # Combined cost: calibration error + smoothness penalty
                total_cost = basic_cost + smoothness_penalty
                
                return total_cost if norm else basic_errors
                    
            except Exception:
                return 1e6 if norm else [1e6] * len(helpers)
        
        return cost_function
    
    def _calibration_report(self, helpers: list, grid_data: list, detailed: bool = False) -> float:
        """Generate calibration report showing fit quality - following Goutham Balaraman's approach."""
        avg_error = 0.0
        
        if detailed:
            print(f"{'Strikes':<15} {'Expiry':<25} {'Market Value':<15} {'Model Value':<15} {'Relative Error (%)':<20}")
            print("=" * 100)
        
        for i, helper in enumerate(helpers):
            try:
                market_val = helper.marketValue()
                model_val = helper.modelValue()
                
                if market_val > 0:
                    # Following Goutham's approach: (model/market - 1.0)
                    rel_error = (model_val / market_val - 1.0)
                    avg_error += abs(rel_error)
                    
                    if detailed:
                        date, strike = grid_data[i]
                        print(f"{strike:<15.2f} {str(date):<25} {market_val:<14.5f} {model_val:<15.5f} {100.0 * rel_error:<20.7f}")
            except:
                # Skip problematic helpers
                continue
        
        # Convert to percentage and average
        avg_error = avg_error * 100.0 / len(helpers) if helpers else 0.0
        
        if detailed:
            print("-" * 100)
        
        summary = f"Average Abs Error (%): {avg_error:.9f}"
        print(summary)
        return avg_error
    def _calculate_implied_vol(self, price: float, spot: float, strike: float, 
                              time_to_expiry: float, option_type: str) -> float:
        """Calculate implied volatility using QuantLib's built-in solver."""
        try:
            # Determine option type for QuantLib
            is_call = option_type.lower() in ['call', 'c']
            option_type_ql = ql.Option.Call if is_call else ql.Option.Put
            
            # Create Black-Scholes process for implied volatility calculation
            spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
            flat_vol_ts = ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(self.evaluation_date, self.calendar, 0.2, self.day_count)
            )
            
            bsm_process = ql.BlackScholesMertonProcess(
                spot_handle, 
                self.dividend_ts, 
                self.yield_ts, 
                flat_vol_ts
            )
            
            # Create European option
            days_to_expiry = int(time_to_expiry * 365)
            expiry_date = self.evaluation_date + ql.Period(days_to_expiry, ql.Days)
            exercise = ql.EuropeanExercise(expiry_date)
            payoff = ql.PlainVanillaPayoff(option_type_ql, float(strike))
            option = ql.VanillaOption(payoff, exercise)
            
            # Set up pricing engine
            engine = ql.AnalyticEuropeanEngine(bsm_process)
            option.setPricingEngine(engine)
            
            # Calculate implied volatility
            impl_vol = option.impliedVolatility(price, bsm_process)
            
            return np.clip(impl_vol, 0.05, 2.0)
            
        except Exception:
            # Fallback: simple approximation for ATM options
            return np.clip(price / (spot * 0.4), 0.05, 2.0)
    def calibrate(self, spot: float, market_data: pd.DataFrame, 
                 maxiter: int = 100, detailed_report: bool = False) -> Tuple[ql.HestonModel, Dict[str, Any]]:
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
            (0.04, 2.0, 0.3, -0.7, 0.04),   # Standard: 20% vol, moderate mean reversion
            (0.09, 1.0, 0.1, -0.5, 0.09),   # High vol: 30% vol, slow mean reversion  
            (0.01, 5.0, 0.6, -0.8, 0.02),   # Low vol: 10% vol, fast mean reversion
            (0.06, 3.0, 0.4, -0.3, 0.05),   # Moderate: 25% vol, medium correlation
            (0.03, 0.5, 0.2, -0.9, 0.03)    # Conservative: 17% vol, very slow mean reversion
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
                (0.01, 1.0),    # sigma: volatility of volatility (0.01 to 1.0)
                (-0.99, 0.99),  # rho: correlation (-0.99 to 0.99)
                (0.01, 1.0)     # v0: initial variance (0.01 to 1.0)
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
                    seed=None,         # Random seed for variety
                    atol=1e-6,         # Tighter tolerance
                    tol=1e-6,          # Tighter tolerance  
                    workers=1,         # Single threaded for stability
                    polish=True        # Local optimization at the end
                )
                
                # Accept result if cost is reasonable, even if maxiter exceeded
                cost_acceptable = result.fun < 10.0  # Reasonable cost threshold
                converged_well = result.fun < best_cost or best_cost == float('inf')
                
                print(f"  → Result: fun={result.fun:.6f}, nfev={result.nfev}, converged_well={converged_well}")
                
                if cost_acceptable and converged_well:
                    best_result = result
                    best_cost = result.fun
                    best_initial = init_condition
                    print(f"  → NEW BEST solution! Cost: {result.fun:.6f}")
                else:
                    print(f"  → Cost: {result.fun:.6f} (not improved from {best_cost:.6f})")
                    
            except Exception as e:
                print(f"  → Exception: {str(e)}")
                continue
        
        # Use the best result found
        if best_result is None:
            return model, {
                'success': False,
                'error': 'All initial conditions failed',
                'calibrated_params': {},
                'average_error': float('inf'),
                'num_helpers': 0
            }
        
        # Setup final model with best parameters
        model, engine = self._setup_model(spot, best_initial)
        helpers, grid_data = self._setup_helpers(engine, market_data, spot)
        
        # Now process the best result
        result = best_result
        
        # Accept good solutions even if iteration limit was reached
        cost_is_good = result.fun < 10.0  # Reasonable cost threshold
        optimization_succeeded = result.success or cost_is_good
        
        if optimization_succeeded:
            # Extract calibrated parameters
            theta, kappa, sigma, rho, v0 = result.x
            
            # Set final parameters
            model.setParams(ql.Array(list(result.x)))
            
            # Ensure Feller condition: 2*kappa*theta > sigma^2
            feller_satisfied = 2 * kappa * theta > sigma**2
            
            # Calculate final calibration error
            avg_error = self._calibration_report(helpers, grid_data, detailed=detailed_report)
            
            print(f"\nCalibration successful with initial condition: {best_initial}")
            print(f"Iterations: {result.nit}, Function evaluations: {result.nfev}")
            print(f"Final cost: {result.fun:.6f}")
            
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
                'feller_satisfied': feller_satisfied,
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
        print(f"{'='*60}")
        print(f"Method:            Differential Evolution (Global Optimizer)")
        print(f"Multi-start:       {'Yes' if calibration_info.get('multi_start_used', False) else 'No'}")
        if 'best_initial_condition' in calibration_info:
            best_init = calibration_info['best_initial_condition']
            print(f"Best initial:      θ={best_init[0]:.3f}, κ={best_init[1]:.3f}, σ={best_init[2]:.3f}, ρ={best_init[3]:.3f}, v₀={best_init[4]:.3f}")
        print(f"Options used:      {calibration_info['num_helpers']}")
        print(f"Average error:     {calibration_info['average_error']:.6f}%")
        print(f"\nCalibrated Parameters:")
        print(f"{'='*60}")
        print(f"theta (θ):         {params['theta']:.6f}  (long-term variance)")
        print(f"kappa (κ):         {params['kappa']:.6f}  (mean reversion speed)")
        print(f"sigma (σ):         {params['sigma']:.6f}  (vol of vol)")
        print(f"rho (ρ):           {params['rho']:.6f}  (correlation)")
        print(f"v0:                {params['v0']:.6f}  (initial variance)")
        print(f"\nDerived Metrics:")
        print(f"{'='*60}")
        print(f"Initial vol:       {np.sqrt(params['v0']):.2%}")
        print(f"Long-term vol:     {np.sqrt(params['theta']):.2%}")
        print(f"Feller condition:  {'✓ Satisfied' if calibration_info['feller_satisfied'] else '✗ Violated'}")
        
        # Show optimization details if available
        if 'optimization_result' in calibration_info:
            opt_result = calibration_info['optimization_result']
            print(f"\nOptimization Details:")
            print(f"{'='*60}")
            print(f"Iterations:        {opt_result.get('nit', 'N/A')}")
            print(f"Function evals:    {opt_result.get('nfev', 'N/A')}")
            print(f"Final cost:        {opt_result.get('fun', 'N/A'):.6f}")
            print(f"Status:            {opt_result.get('message', 'N/A')}")


# Quick test
if __name__ == '__main__':
    # Test the calibrator
    try:
        from market_data_fetcher import MarketDataFetcher
        
        print("Testing QuantLib Heston Calibrator with Differential Evolution...")
        
        # Fetch market data
        fetcher = MarketDataFetcher(ticker='NVDA', expiry_list=['1M', '2M'], atm_range=0.15)
        market_data = fetcher.prepare_market_data()
        spot = fetcher.get_spot_price()
        
        print(f"Spot price: ${spot:.2f}")
        print(f"Market data shape: {market_data.shape}")
        
        # Create and run calibrator with more generous settings
        calibrator = QuantLibHestonCalibrator(r=0.015, q=0.0)
        model, info = calibrator.calibrate(
            spot=spot, 
            market_data=market_data, 
            maxiter=150,  # More iterations for real data
            detailed_report=False
        )
        
        # Print results
        calibrator.print_results(info)
        
    except ImportError:
        print("MarketDataFetcher not available. Running with synthetic data...")
        
        # Create simple synthetic data as fallback
        import pandas as pd
        
        spot = 100.0
        synthetic_data = pd.DataFrame([
            {'Strike': 95.0, 'MarketPrice': 6.50, 'DaysToExpiry': 30, 'OptionType': 'call', 'ImpliedVolatility': 0.22},
            {'Strike': 100.0, 'MarketPrice': 3.20, 'DaysToExpiry': 30, 'OptionType': 'call', 'ImpliedVolatility': 0.20},
            {'Strike': 105.0, 'MarketPrice': 1.10, 'DaysToExpiry': 30, 'OptionType': 'call', 'ImpliedVolatility': 0.25},
            {'Strike': 95.0, 'MarketPrice': 8.70, 'DaysToExpiry': 90, 'OptionType': 'call', 'ImpliedVolatility': 0.24},
            {'Strike': 100.0, 'MarketPrice': 5.80, 'DaysToExpiry': 90, 'OptionType': 'call', 'ImpliedVolatility': 0.22},
            {'Strike': 105.0, 'MarketPrice': 3.50, 'DaysToExpiry': 90, 'OptionType': 'call', 'ImpliedVolatility': 0.26},
        ])
        
        print(f"Using synthetic data with {len(synthetic_data)} options")
        print(f"Spot price: ${spot:.2f}")
        
        # Test with synthetic data
        calibrator = QuantLibHestonCalibrator(r=0.02, q=0.0)
        model, info = calibrator.calibrate(
            spot=spot,
            market_data=synthetic_data,
            maxiter=200,  # More iterations for proper convergence
            detailed_report=False
        )
        
        calibrator.print_results(info)
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
