"""
QuantLib Heston Calibrator - Streamlined version for option model calibration.
"""

import numpy as np
import pandas as pd
import QuantLib as ql
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any


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
        self.rf_curve = ql.FlatForward(self.evaluation_date, self.r, ql.Actual365Fixed())
        self.div_curve = ql.FlatForward(self.evaluation_date, self.q, ql.Actual365Fixed())
        
    def _calculate_initial_parameters(self, market_data: pd.DataFrame, spot: float) -> Dict[str, float]:
        """Estimate initial Heston parameters from market data."""
        v0 = 0.04  # Default fallback 20% vol
        
        try:
            # Estimate volatility from ATM options using implied volatility
            atm_data = market_data[
                (market_data['Strike'] >= spot * 0.95) & 
                (market_data['Strike'] <= spot * 1.05)
            ]
            
            if not atm_data.empty:
                # Calculate implied volatilities for ATM options
                iv_values = []
                for _, row in atm_data.iterrows():
                    try:
                        time_to_expiry = row.DaysToExpiry / 252.0
                        iv = self._calculate_implied_vol(
                            row.MarketPrice, spot, row.Strike, time_to_expiry, row.OptionType
                        )
                        iv_values.append(iv)
                    except:
                        continue
                
                if iv_values:
                    avg_iv = np.mean(iv_values)
                    v0 = avg_iv ** 2
        except Exception:
            pass  # Use fallback v0

        # Conservative parameter bounds
        params = {
            'v0': np.clip(v0, 0.01, 0.25),
            'kappa': 1.5,
            'theta': np.clip(v0 * 0.8, 0.01, 0.16),
            'sigma': 0.3,
            'rho': -0.7
        }
        
        # Ensure Feller condition: 2*kappa*theta > sigma^2
        while 2 * params['kappa'] * params['theta'] <= params['sigma']**2:
            params['kappa'] *= 1.2
            if params['kappa'] > 5.0:
                params['sigma'] = 0.2
                break
        
        return params
        
    def _calculate_implied_vol(self, price: float, spot: float, strike: float, 
                              time_to_expiry: float, option_type: str) -> float:
        """Calculate implied volatility using QuantLib's built-in solver."""
        try:
            # Determine option type for QuantLib
            is_call = option_type.lower() in ['call', 'c']
            option_type_ql = ql.Option.Call if is_call else ql.Option.Put
            
            # Create Black-Scholes process for implied volatility calculation
            spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
            flat_ts = ql.YieldTermStructureHandle(self.rf_curve)
            flat_vol_ts = ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(self.evaluation_date, self.calendar, 0.2, ql.Actual365Fixed())
            )
            
            bsm_process = ql.BlackScholesMertonProcess(
                spot_handle, 
                ql.YieldTermStructureHandle(self.div_curve), 
                flat_ts, 
                flat_vol_ts
            )
            
            # Create European option
            period = ql.Period(int(time_to_expiry * 252), ql.Days)
            expiry_date = self.evaluation_date + period
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
        
    def _create_helpers(self, market_data: pd.DataFrame, spot: float, model: ql.HestonModel) -> list:
        """Create QuantLib calibration helpers from market data using option prices."""
        helpers = []
        engine = ql.AnalyticHestonEngine(model)
        valid_count = 0
        
        # Setup yield curve handles
        rf_handle = ql.YieldTermStructureHandle(self.rf_curve)
        div_handle = ql.YieldTermStructureHandle(self.div_curve)
        
        debug_counts = {'total': 0, 'basic_filter': 0, 'moneyness_filter': 0, 'price_filter': 0, 'iv_filter': 0, 'success': 0}
        
        for idx, (_, row) in enumerate(market_data.iterrows()):
            debug_counts['total'] += 1
            try:
                # Basic data quality filters (less restrictive)
                if (pd.isna(row.MarketPrice) or row.MarketPrice <= 0 or
                    row.DaysToExpiry <= 0):
                    debug_counts['basic_filter'] += 1
                    continue
                
                # Moneyness filter: more lenient range  
                moneyness = spot / row.Strike
                if not (0.5 < moneyness < 2.0 and 5 <= row.DaysToExpiry <= 365):
                    debug_counts['moneyness_filter'] += 1
                    continue
                
                # Quality filters for implied volatility-based calibration
                if row.MarketPrice < 0.10:  # Avoid very cheap options
                    debug_counts['price_filter'] += 1
                    continue
                
                # Calculate implied volatility
                time_to_expiry = row.DaysToExpiry / 252.0
                impl_vol = self._calculate_implied_vol(
                    row.MarketPrice, spot, row.Strike, time_to_expiry, row.OptionType
                )
                
                # Filter based on reasonable implied volatility range
                if impl_vol < 0.05 or impl_vol > 2.0:
                    debug_counts['iv_filter'] += 1
                    continue
                
                # Create QuantLib period and Heston helper
                period = ql.Period(int(row.DaysToExpiry), ql.Days)
                helper = ql.HestonModelHelper(
                    period, self.calendar, spot, float(row.Strike),
                    ql.QuoteHandle(ql.SimpleQuote(impl_vol)),  # Use implied volatility
                    rf_handle, div_handle
                )
                
                helper.setPricingEngine(engine)
                helpers.append(helper)
                valid_count += 1
                debug_counts['success'] += 1
                
            except Exception as e:
                print(f"Failed to create helper for option {idx} (Strike={row.Strike}): {e}")
                continue
        
        print(f"Debug counts: {debug_counts}")
        
        print(f"Created {valid_count} helpers from {len(market_data)} options using implied volatilities")
        return helpers
    
    def calibrate(self, spot: float, market_data: pd.DataFrame, 
                 multi_start: bool = True) -> Tuple[ql.HestonModel, Dict[str, Any]]:
        """Calibrate Heston model to market data using implied volatility approach."""
        print(f"Starting Heston calibration for {len(market_data)} options using implied volatility...")
        
        # Get initial parameters
        initial_params = self._calculate_initial_parameters(market_data, spot)
        print(f"Initial params: v0={initial_params['v0']:.3f}, κ={initial_params['kappa']:.1f}, "
              f"θ={initial_params['theta']:.3f}, σ={initial_params['sigma']:.1f}, ρ={initial_params['rho']:.1f}")
        
        # Create model
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
        process = ql.HestonProcess(
            ql.YieldTermStructureHandle(self.rf_curve),
            ql.YieldTermStructureHandle(self.div_curve),
            spot_handle,
            initial_params['v0'], initial_params['kappa'], initial_params['theta'],
            initial_params['sigma'], initial_params['rho']
        )
        model = ql.HestonModel(process)
        
        # Create helpers
        helpers = self._create_helpers(market_data, spot, model)
        if not helpers:
            raise ValueError("No valid calibration helpers created")
        
        # Calibrate with multi-start if requested
        best_model = model
        best_error = float('inf')
        best_params = None
        
        attempts = 3 if multi_start else 1
        
        for attempt in range(attempts):
            try:
                # Create fresh model for each attempt
                if attempt > 0:
                    # Slightly perturb initial parameters for multi-start
                    perturbed_params = initial_params.copy()
                    perturbed_params['v0'] *= (0.8 + 0.4 * np.random.random())
                    perturbed_params['kappa'] *= (0.8 + 0.4 * np.random.random())
                    perturbed_params['theta'] *= (0.8 + 0.4 * np.random.random())
                    perturbed_params['sigma'] *= (0.8 + 0.4 * np.random.random())
                    perturbed_params['rho'] = np.clip(
                        perturbed_params['rho'] + 0.2 * (np.random.random() - 0.5), -0.99, 0.99
                    )
                    
                    process = ql.HestonProcess(
                        ql.YieldTermStructureHandle(self.rf_curve),
                        ql.YieldTermStructureHandle(self.div_curve),
                        spot_handle,
                        perturbed_params['v0'], perturbed_params['kappa'], perturbed_params['theta'],
                        perturbed_params['sigma'], perturbed_params['rho']
                    )
                    current_model = ql.HestonModel(process)
                    
                    # Update helpers with new model
                    engine = ql.AnalyticHestonEngine(current_model)
                    for helper in helpers:
                        helper.setPricingEngine(engine)
                else:
                    current_model = model
                
                # Calibrate
                optimizer = ql.LevenbergMarquardt()
                end_criteria = ql.EndCriteria(1000, 500, 1e-8, 1e-8, 1e-8)
                
                current_model.calibrate(helpers, optimizer, end_criteria)
                
                # Calculate implied volatility errors (QuantLib calibrates on implied vol)
                errors = []
                for h in helpers:
                    try:
                        model_vol = h.impliedVolatility(h.modelValue(), 1e-6, 100, 0.01, 4.0)
                        market_vol = h.volatility()
                        errors.append(abs(model_vol - market_vol))
                    except:
                        errors.append(0.1)  # fallback error
                avg_error = np.mean(errors)
                
                if avg_error < best_error:
                    best_error = avg_error
                    best_model = current_model
                    best_params = current_model.params()
                
                if multi_start:
                    print(f"  Attempt {attempt + 1}: avg error = {avg_error:.3f} (implied vol)")
                
            except Exception as e:
                if multi_start:
                    print(f"  Attempt {attempt + 1} failed: {e}")
                continue
        
        if best_params is None:
            return model, {'success': False, 'error': 'All calibration attempts failed'}
        
        # Validate best parameters
        params = [
            np.clip(best_params[0], 0.001, 1.0),    # v0
            np.clip(best_params[1], 0.1, 20.0),     # kappa
            np.clip(best_params[2], 0.001, 0.5),    # theta
            np.clip(best_params[3], 0.01, 1.5),     # sigma
            np.clip(best_params[4], -0.99, 0.99)    # rho
        ]
        
        calibration_info = {
            'success': True,
            'calibrated_params': {
                'v0': params[0], 'kappa': params[1], 'theta': params[2],
                'sigma': params[3], 'rho': params[4]
            },
            'average_error': best_error,
            'num_helpers': len(helpers),
            'feller_satisfied': 2 * params[1] * params[2] > params[3]**2,
            'multi_start_used': multi_start
        }
        
        if multi_start:
            print(f"Multi-start calibration completed! Best avg error: {best_error:.3f} (implied vol)")
        else:
            print(f"Calibration completed! Avg error: {best_error:.3f} (implied vol)")
        
        return best_model, calibration_info
    
    def print_results(self, calibration_info: Dict[str, Any]):
        """Print calibration results in a concise format."""
        if not calibration_info['success']:
            print(f"Calibration failed: {calibration_info['error']}")
            return
            
        params = calibration_info['calibrated_params']
        multi_start = calibration_info.get('multi_start_used', False)
        
        print(f"HESTON CALIBRATION RESULTS")
        print(f"{'='*50}")
        print(f"Method:          Implied Volatility (HestonModelHelper)")
        if multi_start:
            print(f"Multi-start:     ✓ (3 attempts)")
        print(f"Initial vol:     {np.sqrt(params['v0']):.1%}")
        print(f"Long-term vol:   {np.sqrt(params['theta']):.1%}")
        print(f"Mean reversion:  {params['kappa']:.1f}")
        print(f"Vol-of-vol:      {params['sigma']:.1%}")
        print(f"Correlation:     {params['rho']:.2f}")
        print(f"Avg error:       {calibration_info['average_error']:.3f} (implied vol)")
        print(f"Options used:    {calibration_info['num_helpers']}")
        print(f"Feller condition: {'✓' if calibration_info['feller_satisfied'] else '✗'}")


# Quick test
if __name__ == '__main__':
    from market_data_fetcher import MarketDataFetcher
    
    fetcher = MarketDataFetcher(ticker='NVDA', expiry_list=['1M', '3M'], atm_range=0.15)
    market_data = fetcher.prepare_market_data()
    spot = fetcher.get_spot_price()
    
    calibrator = QuantLibHestonCalibrator(r=0.015, q=0.0)
    model, info = calibrator.calibrate(spot, market_data)
    calibrator.print_results(info)
