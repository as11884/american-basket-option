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

EXPIRIES = ['1M', '2M']                      # Option expiries to use
ATM_RANGE = 0.10                             # ±10% around spot price (tighter for better quality)
PARALLEL_WORKERS = 4                         # More workers for 30 stocks

# Add project paths
project_root = r'C:\Users\Ao Shen\Desktop\mfin research\src'
heston_path = r'C:\Users\Ao Shen\Desktop\mfin research\src\heston_calib'

for path in [project_root, heston_path]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import required modules
try:
    from market_data_fetcher import MarketDataFetcher
    from quantlib_heston_calibrator import QuantLibHestonCalibrator
    import QuantLib as ql
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running from the correct directory")
    sys.exit(1)

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
        
        if len(market_data) < 10:
            return {'ticker': ticker, 'success': False, 'error': 'Insufficient data'}
        
        # Calibrate
        calibrator = QuantLibHestonCalibrator(r=0.04, q=0.0)
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
            print(f"✅ {ticker} completed - Error: {info['average_error']:.1f}% ({time.time()-start_time:.1f}s)")
            return result
        else:
            return {'ticker': ticker, 'success': False, 'error': 'Calibration failed'}
            
    except Exception as e:
        print(f"❌ {ticker} failed: {str(e)}")
        return {'ticker': ticker, 'success': False, 'error': str(e)}

def main():
    """Main calibration function for top 30 stocks."""
    print("🚀 TOP 30 STOCKS HESTON CALIBRATION")
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
    
    print("🔄 Starting parallel calibrations...")
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
    
    print(f"\n⏱️  Total time: {time.time() - start_time:.1f} seconds")
    print(f"✅ Successful: {len(successful)}/{len(STOCKS)}")
    
    if failed:
        print(f"❌ Failed stocks: {[f['ticker'] for f in failed]}")
        for f in failed:
            print(f"   • {f['ticker']}: {f['error']}")
    
    if not successful:
        print("No successful calibrations to analyze.")
        return

    print("\n📊 CALIBRATION RESULTS")
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
    
    # 2. Save parameter correlation matrix
    param_corr_matrix = raw_params_df[['v0', 'theta', 'kappa', 'sigma', 'rho']].T.corr()
    param_corr_matrix.index = raw_params_df['Ticker']
    param_corr_matrix.columns = raw_params_df['Ticker']
    
    # Save files with clean names
    raw_params_df.to_csv('heston_parameters.csv', index=False)
    param_corr_matrix.to_csv('heston_correlation_matrix.csv')
    
    print(f"\n💾 RESULTS SAVED:")
    print(f"   • Heston Parameters: heston_parameters.csv")
    print(f"   • Correlation Matrix: heston_correlation_matrix.csv")
    
    # Display correlation matrix summary
    print(f"\n🔗 PARAMETER CORRELATION MATRIX SUMMARY")
    print("=" * 80)
    print(f"Highest correlated parameters:")
    
    # Find highest correlations (excluding diagonal)
    corr_flat = param_corr_matrix.values
    np.fill_diagonal(corr_flat, 0)  # Remove diagonal
    max_corr_idx = np.unravel_index(np.argmax(np.abs(corr_flat)), corr_flat.shape)
    max_corr = corr_flat[max_corr_idx]
    
    param_names = param_corr_matrix.columns.tolist()
    print(f"   • {param_names[max_corr_idx[0]]} vs {param_names[max_corr_idx[1]]}: {max_corr:.3f}")
    
    # Show average correlations
    mean_corr = np.mean(np.abs(corr_flat[corr_flat != 0]))
    print(f"   • Average absolute correlation: {mean_corr:.3f}")
    
    # Market analysis using raw parameters
    print(f"\n📈 MARKET ANALYSIS - TOP 30 STOCKS")
    print("=" * 80)
    print(f"Successfully calibrated: {len(successful)}/{len(STOCKS)} stocks")
    
    # Calculate stats from raw parameters
    avg_current_vol = np.sqrt(raw_params_df['v0']).mean() * 100
    avg_longterm_vol = np.sqrt(raw_params_df['theta']).mean() * 100  
    avg_correlation = raw_params_df['rho'].mean()
    avg_mean_reversion = raw_params_df['kappa'].mean()
    avg_error = raw_params_df['Calibration_Error_%'].mean()
    
    print(f"Average Current Vol: {avg_current_vol:.1f}%")
    print(f"Average Long-term Vol: {avg_longterm_vol:.1f}%")
    print(f"Average Correlation: {avg_correlation:.3f}")
    print(f"Average Mean Reversion: {avg_mean_reversion:.2f}")
    print(f"Average Calibration Error: {avg_error:.1f}%")
    
    # Regime classification
    if avg_correlation < -0.5:
        regime = "Strong Leverage Effect"
    elif avg_correlation < -0.2:
        regime = "Moderate Leverage Effect" 
    else:
        regime = "Weak Leverage Effect"
    
    print(f"Market Regime: {regime}")
    
    # Quality assessment
    good_quality = len(raw_params_df[raw_params_df['Calibration_Error_%'] < 15])
    print(f"High Quality Calibrations (<15% error): {good_quality}/30")
    
    print("\n🎉 Top 30 Stock Calibration completed successfully!")

if __name__ == "__main__":
    main()
