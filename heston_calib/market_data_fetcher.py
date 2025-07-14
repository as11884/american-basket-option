"""Market Data Fetcher for option calibration."""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Union


class MarketDataFetcher:
    """Fetch and process option data for calibration."""
    
    def __init__(
        self, 
        ticker: str, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        atm_range: float = 0.2,
        expiry_list: Optional[List[str]] = None
    ):
        """
        Initialize fetcher.
        
        Args:
            ticker: Stock symbol (e.g., 'NVDA')
            start_time: Start date (default: today)
            end_time: End date (default: 1 year from start)
            atm_range: ATM range as fraction (0.2 = ±20%)
            expiry_list: Relative times ['1W', '1M', '3M'] or None
        """
        self.ticker = ticker
        self.atm_range = atm_range
        
        if start_time is None:
            start_time = datetime.now()
        if end_time is None:
            end_time = start_time + timedelta(days=90)
            
        self.start_time = start_time
        self.end_time = end_time
        self.expiry_list = self._process_expiry_list(expiry_list)
        self._spot_price = None
        
    def _process_expiry_list(self, expiry_list: Optional[List[str]]) -> Optional[List[datetime]]:
        """Convert relative time strings to actual expiry dates."""
        if expiry_list is None:
            return None
            
        processed_expiries = []
        today = datetime.now()
        
        for expiry in expiry_list:
            if isinstance(expiry, str) and self._is_relative_time(expiry):
                target_date = self._parse_relative_time(expiry, today)
                if target_date:
                    closest_expiry = self._find_closest_expiry_to_date(target_date)
                    if closest_expiry:
                        processed_expiries.append(closest_expiry)
                    else:
                        print(f"Warning: No expiry found close to {expiry} (target: {target_date.strftime('%Y-%m-%d')})")
        
        if processed_expiries:
            processed_expiries = sorted(list(set(processed_expiries)))
            print(f"Using specific expiry dates: {[exp.strftime('%Y-%m-%d') for exp in processed_expiries]}")
            return processed_expiries
        else:
            return None
    
    def _is_relative_time(self, time_str: str) -> bool:
        """Check if string matches pattern like '1M', '3W'."""
        import re
        return bool(re.match(r'^\d+[WMDYH]$', time_str.upper()))
    
    def _parse_relative_time(self, time_str: str, base_date: datetime) -> Optional[datetime]:
        """Parse relative time string to target datetime."""
        import re
        
        time_str = time_str.upper()
        match = re.match(r'^(\d+)([WMDYH])$', time_str)
        
        if not match:
            return None
            
        amount = int(match.group(1))
        unit = match.group(2)
        
        try:
            if unit == 'H':
                return base_date + timedelta(hours=amount)
            elif unit == 'D':
                return base_date + timedelta(days=amount)
            elif unit == 'W':
                return base_date + timedelta(weeks=amount)
            elif unit == 'M':
                return base_date + timedelta(days=amount * 30)
            elif unit == 'Y':
                return base_date + timedelta(days=amount * 365)
        except Exception:
            pass
            
        return None
    
    def _find_closest_expiry_to_date(self, target_date: datetime, tolerance_days: int = 7) -> Optional[datetime]:
        """Find actual option expiry closest to target date."""
        try:
            tk = yf.Ticker(self.ticker)
            available_expiries = tk.options
            
            closest_expiry = None
            min_diff = float('inf')
            
            for exp_str in available_expiries:
                exp_date = datetime.strptime(exp_str, '%Y-%m-%d')
                diff_days = abs((exp_date - target_date).days)
                
                if diff_days <= tolerance_days and diff_days < min_diff:
                    min_diff = diff_days
                    closest_expiry = exp_date
            
            if closest_expiry:
                print(f"Target {target_date.strftime('%Y-%m-%d')}: Found {closest_expiry.strftime('%Y-%m-%d')} ({min_diff} days difference)")
            
            return closest_expiry
            
        except Exception:
            return None

    def get_spot_price(self) -> float:
        """Get current spot price."""
        if self._spot_price is None:
            stock = yf.Ticker(self.ticker)
            hist = stock.history(period='1d', interval='1d', auto_adjust=True)
            self._spot_price = hist['Close'].iloc[-1]
        return self._spot_price
    
    def fetch_raw_option_data(self) -> pd.DataFrame:
        """Fetch raw option data from yfinance."""
        print(f"Fetching option data for {self.ticker}...")
        
        today = pd.Timestamp.now().normalize()
        rows = []
        tk = yf.Ticker(self.ticker)
        
        try:
            if self.expiry_list is not None:
                # Use specific expiry list
                target_expiries = [exp.strftime('%Y-%m-%d') for exp in self.expiry_list]
                print(f"Fetching specific expiries: {target_expiries}")
                
                for exp_str in target_expiries:
                    if exp_str in tk.options:
                        exp_date = pd.to_datetime(exp_str)
                        chain = tk.option_chain(exp_str)
                        
                        df = pd.concat([
                            chain.calls.assign(OptionType='call'),
                            chain.puts.assign(OptionType='put')
                        ], ignore_index=True)
                        
                        df['ExpirationDate'] = exp_date
                        df['DaysToExpiry'] = (df['ExpirationDate'] - today).dt.days
                        df['MarketPrice'] = (df['bid'] + df['ask']) / 2.0
                        df['Strike'] = df['strike']
                        df['Spread'] = df['ask'] - df['bid']
                        df['SpreadPct'] = df['Spread'] / df['MarketPrice'].replace(0, np.nan)
                        df['ImpliedVolatility'] = df['impliedVolatility']  # Yahoo Finance IV
                        
                        rows.append(df)
                        print(f"  Fetched {len(df)} contracts for {exp_str}")
            else:
                # Use time range filtering
                for exp in tk.options:
                    exp_date = pd.to_datetime(exp)
                    
                    if exp_date < pd.Timestamp(self.start_time) or exp_date > pd.Timestamp(self.end_time):
                        continue
                    
                    chain = tk.option_chain(exp)
                    
                    df = pd.concat([
                        chain.calls.assign(OptionType='call'),
                        chain.puts.assign(OptionType='put')
                    ], ignore_index=True)
                    
                    df['ExpirationDate'] = exp_date
                    df['DaysToExpiry'] = (df['ExpirationDate'] - today).dt.days
                    df['MarketPrice'] = (df['bid'] + df['ask']) / 2.0
                    df['Strike'] = df['strike']
                    df['Spread'] = df['ask'] - df['bid']
                    df['SpreadPct'] = df['Spread'] / df['MarketPrice'].replace(0, np.nan)
                    df['ImpliedVolatility'] = df['impliedVolatility']  # Yahoo Finance IV
                    
                    rows.append(df)
                
        except Exception as e:
            print(f"Error fetching option data: {e}")
            return pd.DataFrame()
        
        if not rows:
            return pd.DataFrame()
            
        all_data = pd.concat(rows, ignore_index=True)
        print(f"Fetched {len(all_data)} raw option contracts")
        
        return all_data
    
    def apply_quality_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply quality filters to remove bad option data."""
        if df.empty:
            return df
            
        print("Applying quality filters...")
        initial_count = len(df)
        
        df_filtered = df[
            (df['volume'] > 0) &
            (df['bid'] > 0) & (df['ask'] > 0) &
            (df['DaysToExpiry'] > 0) &
            (df['DaysToExpiry'] <= 365) &
            (df['MarketPrice'] > 0.01) &
            (df['SpreadPct'] <= 0.5) &
            (df['openInterest'] >= 1)
        ].copy()
        
        print(f"Quality filters removed {initial_count - len(df_filtered)} contracts")
        return df_filtered
    
    def apply_atm_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only options near ATM."""
        print("Applying ATM filter...")
        spot = self.get_spot_price()
        initial_count = len(df)
        
        lower_bound = spot * (1 - self.atm_range)
        upper_bound = spot * (1 + self.atm_range)
        
        df_atm = df[
            (df['Strike'] >= lower_bound) &
            (df['Strike'] <= upper_bound)
        ].copy()
        
        print(f"ATM filter (±{self.atm_range:.1%} around ${spot:.2f}) kept {len(df_atm)} of {initial_count} contracts")
        print(f"Strike range: ${df_atm['Strike'].min():.2f} to ${df_atm['Strike'].max():.2f}")
        
        return df_atm
    
    def prepare_market_data(self) -> pd.DataFrame:
        """Fetch and process market data."""
        print(f"Preparing market data for {self.ticker}")
        print(f"Time range: {self.start_time.strftime('%Y-%m-%d')} to {self.end_time.strftime('%Y-%m-%d')}")
        print(f"ATM range: ±{self.atm_range:.1%}")
        
        # Fetch, filter, and process
        raw_data = self.fetch_raw_option_data()
        if raw_data.empty:
            return pd.DataFrame()
        
        quality_filtered = self.apply_quality_filters(raw_data)
        if quality_filtered.empty:
            return pd.DataFrame()
        
        atm_filtered = self.apply_atm_filter(quality_filtered)
        if atm_filtered.empty:
            return pd.DataFrame()
        
        final_df = atm_filtered[['OptionType', 'Strike', 'DaysToExpiry', 'MarketPrice', 'ImpliedVolatility']].copy()
        
        print(f"Final dataset: {len(final_df)} contracts across {final_df['DaysToExpiry'].nunique()} maturities")
        print(f"DTE range: {final_df['DaysToExpiry'].min()} to {final_df['DaysToExpiry'].max()} days")
        
        return final_df
    
    def get_summary_stats(self, df: pd.DataFrame) -> dict:
        """Get summary statistics."""
        if df.empty:
            return {}
            
        return {
            'total_contracts': len(df),
            'call_contracts': len(df[df['OptionType'] == 'call']),
            'put_contracts': len(df[df['OptionType'] == 'put']),
            'unique_maturities': df['DaysToExpiry'].nunique(),
            'dte_range': (df['DaysToExpiry'].min(), df['DaysToExpiry'].max()),
            'strike_range': (df['Strike'].min(), df['Strike'].max()),
            'price_range': (df['MarketPrice'].min(), df['MarketPrice'].max()),
            'spot_price': self.get_spot_price()
        }


# Quick test
if __name__ == '__main__':
    fetcher = MarketDataFetcher(ticker='NVDA', expiry_list=['1M', '3M'], atm_range=0.4)
    data = fetcher.prepare_market_data()
    
    if not data.empty:
        stats = fetcher.get_summary_stats(data)
        print(f"\n📊 Summary: {stats['total_contracts']} contracts, "
              f"{stats['unique_maturities']} maturities, "
              f"DTE: {stats['dte_range'][0]}-{stats['dte_range'][1]} days")
    else:
        print("No data found")
