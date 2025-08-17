#!/usr/bin/env python3
"""
Simple Multi-Stock Heston Calibration
=====================================

A streamlined version for quick multi-stock calibration with
easy customization and clear output.

Usage:
    python simple_multi_stock.py

Features:
- Parallel calibration of multiple stocks
- Clear progress reporting  
- Automatic result export
- Parameter comparison table
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import time

# Configuration for Top 30 Stocks (existing since 2017)
STOCKS = [
    # Tech giants
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX', 'CRM', 'ORCL',
    # Financial sector
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C',
    # Healthcare & Consumer
    'JNJ', 'PFE', 'UNH', 'PG', 'KO', 'PEP', 'WMT', 'HD',
    # Industrial & Energy
    'BA', 'CAT', 'GE', 'XOM', 'CVX', 'V'
]

EXPIRIES = ['1M', '2M', '3M']                # Option expiries to use
ATM_RANGE = 0.1                              # ±10% around spot price (tighter for better quality)
PARALLEL_WORKERS = 8                         # More workers for 30 stocks

# Add project paths
project_root = r'C:\Users\Ao Shen\Desktop\mfin research\src'
heston_path = r'C:\Users\Ao Shen\Desktop\mfin research\src\heston_calib'

for path in [project_root, heston_path]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import required modules
from market_data_fetcher import MarketDataFetcher
from quantlib_heston_calibrator import QuantLibHestonCalibrator
from covariance_estimator import CovarianceEstimator
import QuantLib as ql


warnings.filterwarnings('ignore')

def calibrate_stock(ticker):
    """Calibrate a single stock and return results."""
    print(f"📈 Calibrating {ticker}...")
    start_time = time.time()
    
    try:
        # Fetch market data
        fetcher = MarketDataFetcher(ticker=ticker, expiry_list=EXPIRIES, atm_range=ATM_RANGE)
        market_data = fetcher.prepare_market_data()
        spot_price = fetcher.get_spot_price()
        
        # Calibrate
        calibrator = QuantLibHestonCalibrator(r=0.05, q=0.0)
        heston_model, info = calibrator.calibrate(
            spot=spot_price, 
            market_data=market_data, 
            maxiter=100, 
            detailed_report=False
        )
        
        if info['success']:
            params = info['calibrated_params']
            result = {
                'ticker': ticker,
                'success': True,
                'spot_price': spot_price,
                'options_count': len(market_data),
                'error_pct': info['average_error'],
                'current_vol': np.sqrt(params['v0']) * 100,
                'long_term_vol': np.sqrt(params['theta']) * 100,
                'mean_reversion': params['kappa'],
                'vol_of_vol': params['sigma'] * 100,
                'correlation': params['rho'],
                'time_taken': time.time() - start_time,
                'raw_params': params  # Include raw Heston parameters
            }
            print(f"{ticker} completed - Error: {info['average_error']:.1f}% ({time.time()-start_time:.1f}s)")
            return result
        else:
            return {'ticker': ticker, 'success': False, 'error': 'Calibration failed'}
            
    except Exception as e:
        print(f"{ticker} failed: {str(e)}")
        return {'ticker': ticker, 'success': False, 'error': str(e)}

def main():
    """Main calibration function for top 30 stocks."""
    print("TOP 30 STOCKS HESTON CALIBRATION")
    print("=" * 80)
    print(f"Target: {len(STOCKS)} major US stocks (existing since 2017)")
    print(f"Stocks: {STOCKS[:10]}... (showing first 10)")
    print(f"Expiries: {EXPIRIES}")
    print(f"ATM Range: ±{ATM_RANGE*100:.0f}%")
    print(f"Workers: {PARALLEL_WORKERS}")
    print()
    
    # Run calibrations in parallel
    start_time = time.time()
    results = []
    
    print("Starting parallel calibrations...")
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        # Submit all tasks
        futures = {executor.submit(calibrate_stock, ticker): ticker for ticker in STOCKS}
        
        # Collect results
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            if completed % 5 == 0 or completed == len(STOCKS):
                print(f"   Progress: {completed}/{len(STOCKS)} stocks completed")
    
    # Process results
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n Total time: {time.time() - start_time:.1f} seconds")
    print(f" Successful: {len(successful)}/{len(STOCKS)}")
    
    if failed:
        print(f"Failed stocks: {[f['ticker'] for f in failed]}")
        for f in failed:
            print(f"   • {f['ticker']}: {f['error']}")
    
    if not successful:
        print("No successful calibrations to analyze.")
        return

    print("\n CALIBRATION RESULTS")
    print("=" * 80)
    
    # Export comprehensive results
    
    # 1. Save detailed results with raw Heston parameters (v0, theta, kappa, sigma, rho)
    raw_params = []
    for result in successful:
        # Extract raw Heston parameters directly from calibration
        if 'raw_params' in result:
            raw = result['raw_params']
            raw_params.append({
                'Ticker': result['ticker'],
                'Spot_Price': result['spot_price'],
                'Options_Count': result['options_count'],
                'Calibration_Error_%': result['error_pct'],
                'v0': raw['v0'],  # Initial variance
                'theta': raw['theta'],  # Long-term variance
                'kappa': raw['kappa'],  # Mean reversion speed
                'sigma': raw['sigma'],  # Volatility of volatility
                'rho': raw['rho'],  # Correlation
                'Calibration_Time_s': result['time_taken']
            })
    
    raw_params_df = pd.DataFrame(raw_params)
    
    # 2. Calculate ACTUAL stock correlation matrix using historical returns
    print("Calculating stock correlation matrix from historical returns...")
    successful_tickers = [r['ticker'] for r in successful]
    
    try:
        corr_estimator = CovarianceEstimator(successful_tickers, lookback_days=90)
        corr_estimator.fetch_price_data()
        corr_estimator.calculate_returns()
        stock_corr_matrix = corr_estimator.get_correlation_matrix()
        
        # Convert to DataFrame for better handling
        stock_corr_df = pd.DataFrame(
            stock_corr_matrix, 
            index=successful_tickers, 
            columns=successful_tickers
        )
        
        print(f"Stock correlation matrix calculated using 1-year historical returns")
        
    except Exception as e:
        print(f"Failed to calculate stock correlations: {e}")
        print("Creating identity matrix as fallback...")
        stock_corr_df = pd.DataFrame(
            np.eye(len(successful_tickers)), 
            index=successful_tickers, 
            columns=successful_tickers
        )
    
    # Save files with clean names
    raw_params_df.to_csv('heston_parameters.csv', index=False)
    stock_corr_df.to_csv('heston_correlation_matrix.csv')

    print(f"RESULTS SAVED:")
    print(f"• Heston Parameters: heston_parameters.csv")
    print(f"• Stock Correlation Matrix: heston_correlation_matrix.csv")
    print(f"• Total successful calibrations: {len(successful)}")

if __name__ == "__main__":
    main()
