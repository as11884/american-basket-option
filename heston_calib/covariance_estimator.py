"""Simple Correlation Matrix Estimation for Multi-Asset Heston Models."""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')


class CovarianceEstimator:
    """Simple correlation matrix estimator."""
    
    def __init__(
        self,
        tickers: List[str],
        lookback_days: int = 252
    ):
        """
        Initialize correlation estimator.
        
        Args:
            tickers: List of stock symbols
            lookback_days: Number of days to look back
        """
        self.tickers = tickers
        self.lookback_days = lookback_days
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=lookback_days + 30)
        
        self.price_data = None
        self.returns = None
        
    def fetch_price_data(self) -> pd.DataFrame:
        """Fetch historical price data."""
        price_data = {}
        
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(
                    start=self.start_date.strftime('%Y-%m-%d'),
                    end=self.end_date.strftime('%Y-%m-%d'),
                    auto_adjust=True
                )
                
                if len(hist) > 50:
                    price_data[ticker] = hist['Close']
                    
            except Exception:
                continue
        
        self.price_data = pd.DataFrame(price_data).dropna()
        
        if len(self.price_data) > self.lookback_days:
            self.price_data = self.price_data.tail(self.lookback_days)
        
        return self.price_data
    
    def calculate_returns(self) -> pd.DataFrame:
        """Calculate daily returns."""
        if self.price_data is None:
            self.fetch_price_data()
        
        self.returns = np.log(self.price_data / self.price_data.shift(1)).dropna()
        return self.returns
    
    def get_correlation_matrix(self) -> np.ndarray:
        """Get correlation matrix from daily returns."""
        if self.returns is None:
            self.calculate_returns()
        
        return self.returns.corr().values
    
    def get_volatilities(self) -> np.ndarray:
        """Get annualized volatilities."""
        if self.returns is None:
            self.calculate_returns()
        
        return self.returns.std().values * np.sqrt(252)

# Simple usage
if __name__ == '__main__':
    estimator = CovarianceEstimator(['AAPL', 'MSFT', 'NVDA'])
    corr_matrix = estimator.get_correlation_matrix()
    print(corr_matrix)
