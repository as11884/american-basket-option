import matplotlib.pyplot as plt
from datetime import datetime

# Import our calibration modules
from market_data import prepare_calibration_data, filter_options_for_calibration
from heston_calibrator import HestonCalibrator
from calib_utils import HestonParams
from heston_model import HestonModel


def calibrate_nvda():
    ticker = "NVDA"
    print(f"=== NVDA Heston Calibration Example ({ticker}) ===", flush=True)

    # 1) Prepare data
    print("Fetching market data...", flush=True)
    data = prepare_calibration_data(
        ticker_symbol=ticker,
        historical_period="3mo",
        target_expiry_days=[30, 60, 90]
    )
    spot_price = data['spot_price']
    realized_vol = data['realized_vol']
    iv_surface = data['iv_surface']

    print(f"Spot price: {spot_price:.2f}", flush=True)
    print(f"Realized volatility (annualized): {realized_vol:.4f}", flush=True)

    # 2) Filter option data for calibration
    print("Filtering option data...", flush=True)
    filtered_iv = filter_options_for_calibration(
        iv_surface,
        min_moneyness=0.8,
        max_moneyness=1.2,
        min_volume=10,
        max_days_to_expiry=90
    )
    print(f"Using {len(filtered_iv)} options for calibration", flush=True)

    # 3) Initial parameter guess
    print("Setting initial parameters...", flush=True)
    init_params = HestonParams(
        v0=realized_vol**2,
        theta=realized_vol**2,
        kappa=2.0,
        sigma=0.3,
        rho=-0.7
    )

    # 4) Run calibration
    print("Running Heston calibration...", flush=True)
    calibrator = HestonCalibrator(risk_free_rate=0.03)
    calibrated_params = calibrator.calibrate(
        spot_price=spot_price,
        option_data=filtered_iv,
        initial_parameters=init_params
    )
    print("Calibration complete.", flush=True)
    print(f"Calibrated parameters: {calibrated_params}", flush=True)

    # 5) Simulate with calibrated model
    print("Simulating price paths with calibrated model...", flush=True)
    model = HestonModel(
        heston_parameters=calibrated_params,
        risk_free_rate=0.03
    )
    paths = model.generate_paths(
        initial_stock_price=spot_price,
        time_horizon=0.25,  # 3 months
        num_simulation_paths=1000
    )

    # 6) Plot results
    print("Plotting results...", flush=True)
    plt.figure(figsize=(10, 6))
    plt.plot(paths[:, :].T, alpha=0.5)
    plt.title(f"{ticker} Heston Price Paths (Calibrated)")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.show()

if __name__ == "__main__":
    calibrate_nvda()