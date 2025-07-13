"""Streamlined Heston model calibration pipeline."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
import QuantLib as ql

from market_data_fetcher import MarketDataFetcher
from quantlib_heston_calibrator import QuantLibHestonCalibrator


class HestonCalibrationPipeline:
    """Complete Heston calibration pipeline."""
    
    def __init__(
        self,
        ticker: str,
        r: float = 0.015,
        q: float = 0.0,
        start_date: Optional[datetime] = None,
        expiry_list: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        atm_range: float = 0.15
    ):
        """Initialize pipeline with all parameters."""
        self.ticker = ticker
        self.r = r
        self.q = q
        self.start_date = start_date or datetime.now()
        
        # Market data parameters
        self.expiry_list = expiry_list
        self.start_time = start_time
        self.end_time = end_time
        self.atm_range = atm_range
        
        # Initialize components
        self.data_fetcher = None
        self.calibrator = QuantLibHestonCalibrator(r=r, q=q, start_date=start_date)
        self.market_data = None
        self.spot_price = None
        self.calibration_results = None
    
    def fetch_market_data(self) -> pd.DataFrame:
        """Fetch market data using initialization parameters."""
        print(f"🔄 FETCHING MARKET DATA FOR {self.ticker}")
        print("=" * 50)
        
        self.data_fetcher = MarketDataFetcher(
            ticker=self.ticker,
            start_time=self.start_time,
            end_time=self.end_time,
            atm_range=self.atm_range,
            expiry_list=self.expiry_list
        )
        
        self.market_data = self.data_fetcher.prepare_market_data()
        self.spot_price = self.data_fetcher.get_spot_price()
        
        if self.market_data.empty:
            raise ValueError(f"No market data found for {self.ticker}")
        
        # Print summary
        stats = self.data_fetcher.get_summary_stats(self.market_data)
        print(f"DATA SUMMARY")
        print(f"Spot price: ${stats['spot_price']:.2f}")
        print(f"Total contracts: {stats['total_contracts']} ({stats['call_contracts']} calls, {stats['put_contracts']} puts)")
        print(f"Maturities: {stats['unique_maturities']}")
        print(f"DTE range: {stats['dte_range'][0]} to {stats['dte_range'][1]} days")
        print(f"Strike range: ${stats['strike_range'][0]:.2f} to ${stats['strike_range'][1]:.2f}")
        print(f"Price range: ${stats['price_range'][0]:.2f} to ${stats['price_range'][1]:.2f}")
        
        return self.market_data
    
    def calibrate_heston(self, multi_start: bool = True) -> Tuple[ql.HestonModel, Dict[str, Any]]:
        """Calibrate Heston model to the fetched market data."""
        if self.market_data is None or self.spot_price is None:
            raise ValueError("Must fetch market data before calibration. Call fetch_market_data() first.")
        
        print(f"CALIBRATING HESTON MODEL")
        print("=" * 50)
        
        model, info = self.calibrator.calibrate(
            self.spot_price, self.market_data, multi_start=multi_start
        )
        self.calibration_results = info
        
        return model, info
    
    def run_full_calibration(
        self,
        multi_start: bool = True,
        print_results: bool = True
    ) -> Tuple[ql.HestonModel, Dict[str, Any]]:
        """Run complete Heston calibration process."""
        print(f"HESTON CALIBRATION PIPELINE FOR {self.ticker}")
        print("=" * 60)
        
        self.fetch_market_data()
        model, info = self.calibrate_heston(multi_start=multi_start)
        
        if print_results:
            self.calibrator.print_results(info)
        
        return model, info
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get calibration results summary."""
        if self.calibration_results is None:
            return {}
        
        if not self.calibration_results['success']:
            return {'status': 'failed', 'error': self.calibration_results['error']}
        
        params = self.calibration_results['calibrated_params']
        
        return {
            'status': 'success',
            'ticker': self.ticker,
            'spot_price': self.spot_price,
            'num_options': self.calibration_results['num_helpers'],
            'avg_error': self.calibration_results['average_error'],
            'parameters': {
                'initial_vol': np.sqrt(params['v0']),
                'long_term_vol': np.sqrt(params['theta']),
                'mean_reversion': params['kappa'],
                'vol_of_vol': params['sigma'],
                'correlation': params['rho']
            },
            'feller_satisfied': self.calibration_results['feller_satisfied']
        }
    
    def save_results(self, filepath: str):
        """Save calibration results to CSV."""
        if self.calibration_results is None:
            raise ValueError("No calibration results to save. Run calibration first.")
        
        summary = self.get_calibration_summary()
        
        if summary['status'] == 'failed':
            print(f"Cannot save failed calibration results")
            return
        
        results_data = {
            'ticker': [self.ticker],
            'calibration_date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'spot_price': [summary['spot_price']],
            'num_options': [summary['num_options']],
            'avg_error': [summary['avg_error']],
            'initial_vol': [summary['parameters']['initial_vol']],
            'long_term_vol': [summary['parameters']['long_term_vol']],
            'mean_reversion': [summary['parameters']['mean_reversion']],
            'vol_of_vol': [summary['parameters']['vol_of_vol']],
            'correlation': [summary['parameters']['correlation']],
            'feller_satisfied': [summary['feller_satisfied']],
            'risk_free_rate': [self.r],
            'dividend_yield': [self.q]
        }
        
        df = pd.DataFrame(results_data)
        df.to_csv(filepath, index=False)
        print(f"📁 Results saved to {filepath}")


# Convenience functions
def calibrate_with_expiry_list(
    ticker: str,
    expiry_list: List[str],
    atm_range: float = 0.15,
    r: float = 0.015,
    q: float = 0.0
) -> Tuple[ql.HestonModel, Dict[str, Any]]:
    """Quick calibration with expiry targets."""
    pipeline = HestonCalibrationPipeline(
        ticker=ticker, r=r, q=q, expiry_list=expiry_list, atm_range=atm_range
    )
    return pipeline.run_full_calibration()


def calibrate_with_time_range(
    ticker: str,
    start_time: datetime,
    end_time: datetime,
    atm_range: float = 0.15,
    r: float = 0.015,
    q: float = 0.0
) -> Tuple[ql.HestonModel, Dict[str, Any]]:
    """Quick calibration with time range filtering."""
    pipeline = HestonCalibrationPipeline(
        ticker=ticker, r=r, q=q, start_time=start_time, end_time=end_time, atm_range=atm_range
    )
    return pipeline.run_full_calibration()