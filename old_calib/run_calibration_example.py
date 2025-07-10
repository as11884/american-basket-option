from market_data import prepare_calibration_data
from heston_calibrator import HestonCalibrator, HestonParams
from datetime import datetime

# Encapsulate daily calibration in a function
def calibrate_today(ticker="NVDA"):
    # 1) Fetch today's 3‑month data
    data = prepare_calibration_data(ticker, period="3mo", expiry_days=[30,60,90])
    stock_df = data['stock_data']
    real_vol = data['realized_vol']
    iv_surf = data['iv_surface']
    spot = stock_df['Close'].iloc[-1]

    # 2) Initial guess based on realized vol
    init = HestonParams(
        v0=real_vol**2, theta=real_vol**2,
        kappa=2.0, sigma=0.3, rho=-0.7
    )

    # 3) Filter near‑ATM, short‑dated options
    filt = iv_surf[(iv_surf.Moneyness.between(0.8,1.2)) & (iv_surf.DaysToExpiry <= 90)]

    # 4) Calibrate
    cal = HestonCalibrator(r=0.03)
    fit = cal.calibrate(spot, filt, init, max_iter=100)
    print(f"{datetime.now().date()}: Calibrated {ticker} params: {fit.to_dict()}")
    return fit

# Run if executed directly
if __name__ == "__main__":
    calibrate_today()