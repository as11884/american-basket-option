import numpy as np
import pandas as pd
import yfinance as yf
import QuantLib as ql

# ---------------------------------------------------------------------------
#  Fetch and prepare market data
# ---------------------------------------------------------------------------
def prepare_market_data(ticker: str) -> pd.DataFrame:
    """
    Fetches the underlying price and builds a market DataFrame for calibration.

    Returns columns: ['OptionType', 'Strike', 'DaysToExpiry', 'MarketPrice']
    """
    # fetch stock history
    stock = yf.Ticker(ticker).history(period="6mo", interval="1d", auto_adjust=True)
    # today's date for DTE calculation
    today = pd.Timestamp.now().normalize()

    # collect option chains
    rows = []
    tk = yf.Ticker(ticker)
    for exp in tk.options:
        # expiration date as Timestamp
        exp_date = pd.to_datetime(exp)
        # fetch calls + puts once
        chain = tk.option_chain(exp)
        df = pd.concat([
            chain.calls.assign(OptionType='call'),
            chain.puts.assign(OptionType='put')
        ], ignore_index=True)
        # compute days to expiry via datetime64
        df['ExpirationDate'] = exp_date
        df['DaysToExpiry'] = (df['ExpirationDate'] - today).dt.days
        # mid-market price
        df['MarketPrice'] = (df['bid'] + df['ask']) / 2.0
        # rename strike
        df['Strike'] = df['strike']
        
        # Filter for reasonable data
        df = df[
            (df['volume'] > 0) &  # has trading volume
            (df['bid'] > 0) & (df['ask'] > 0) &  # valid bid/ask
            (df['DaysToExpiry'] > 0) &  # positive time to expiry
            (df['DaysToExpiry'] <= 365) &  # not too far out
            (df['MarketPrice'] > 0.01)  # reasonable price
        ]
        
        if not df.empty:
            rows.append(df[['OptionType','Strike','DaysToExpiry','MarketPrice']])

    # concatenate all expiries
    market_data = pd.concat(rows, ignore_index=True)
    
    # Additional filtering for ATM options (better calibration)
    spot = yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1]
    market_data = market_data[
        (market_data['Strike'] >= spot * 0.8) &
        (market_data['Strike'] <= spot * 1.2)
    ]
    
    return market_data

# ---------------------------------------------------------------------------
#  Heston calibration via QuantLib
# ---------------------------------------------------------------------------
def calibrate_heston_quantlib(
    spot: float,
    r: float,
    q: float,
    market_data: pd.DataFrame
) -> ql.HestonModel:
    # set evaluation date
    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    todaysDate = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = todaysDate

    # build yield curves
    rf_curve  = ql.FlatForward(todaysDate, r, ql.Actual365Fixed())
    div_curve = ql.FlatForward(todaysDate, q, ql.Actual365Fixed())
    uquote     = ql.SimpleQuote(spot)

    # Heston process & model with better initial parameters
    # Use realized volatility as starting point
    hist_vol = market_data.groupby('DaysToExpiry')['MarketPrice'].std().mean() / spot
    v0 = min(hist_vol**2, 0.1)  # Cap at reasonable level
    kappa = 2.0  # Mean reversion speed
    theta = v0   # Long-term variance equals initial
    sigma = 0.3  # Vol of vol
    rho = -0.5   # Negative correlation typical for equity
    
    print(f"Initial parameters: v0={v0:.4f}, kappa={kappa}, theta={theta:.4f}, sigma={sigma}, rho={rho}")
    
    process = ql.HestonProcess(
        ql.YieldTermStructureHandle(rf_curve),
        ql.YieldTermStructureHandle(div_curve),
        ql.QuoteHandle(uquote),
        v0, kappa, theta, sigma, rho
    )
    model  = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)

    # helpers
    helpers = []
    for _, row in market_data.iterrows():
        # Skip invalid data
        if pd.isna(row.MarketPrice) or row.MarketPrice <= 0:
            continue
            
        # Create HestonModelHelper with correct arguments
        helper = ql.HestonModelHelper(
            ql.Period(int(row.DaysToExpiry), ql.Days),  # maturity
            calendar,                                    # calendar
            spot,                                        # spot price
            row.Strike,                                  # strike
            ql.QuoteHandle(ql.SimpleQuote(row.MarketPrice)),  # market price
            ql.YieldTermStructureHandle(rf_curve),       # risk-free rate
            ql.YieldTermStructureHandle(div_curve)       # dividend yield
        )
        helper.setPricingEngine(engine)
        helpers.append(helper)

    # calibrate
    optimizer   = ql.LevenbergMarquardt()
    end_criteria = ql.EndCriteria(1000, 100, 1e-8, 1e-8, 1e-8)
    model.calibrate(helpers, optimizer, end_criteria)
    return model

# ---------------------------------------------------------------------------
#  Example usage
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    TICKER = 'NVDA'
    
    try:
        # spot price
        spot = yf.Ticker(TICKER).history(period='1d', interval='1d', auto_adjust=True)['Close'].iloc[-1]
        print(f"Current spot price for {TICKER}: ${spot:.2f}")
        
        # prepare market data
        market_data = prepare_market_data(TICKER)
        print(f"Found {len(market_data)} option contracts")
        print(f"DTE range: {market_data['DaysToExpiry'].min()} to {market_data['DaysToExpiry'].max()} days")
        print(f"Strike range: ${market_data['Strike'].min():.2f} to ${market_data['Strike'].max():.2f}")
        
        if len(market_data) == 0:
            print("No valid market data found!")
            exit(1)
            
        # flat rates
        r, q = 0.015, 0.0
        
        # run calibration
        print("Starting Heston calibration...")
        model = calibrate_heston_quantlib(spot, r, q, market_data)
        
        # Extract calibrated parameters
        params = model.params()
        print('\nCalibrated Heston parameters:')
        print(f"v0 (initial variance): {params[0]:.6f}")
        print(f"kappa (mean reversion): {params[1]:.6f}")
        print(f"theta (long-term variance): {params[2]:.6f}")
        print(f"sigma (vol of vol): {params[3]:.6f}")
        print(f"rho (correlation): {params[4]:.6f}")
        
        # Alternative: get parameters from the process directly
        process = model.process()
        print('\nAlternative parameter extraction:')
        print(f"v0: {process.v0():.6f}")
        print(f"kappa: {process.kappa():.6f}")
        print(f"theta: {process.theta():.6f}")
        print(f"sigma: {process.sigma():.6f}")
        print(f"rho: {process.rho():.6f}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
