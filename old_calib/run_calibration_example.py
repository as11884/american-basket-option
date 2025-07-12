import warnings
import matplotlib.pyplot as plt
import numpy as np
from market_data import prepare_dataset
from heston_calibrator import HestonParams, PriceHestonCalibrator, _BS
from scipy.optimize import brentq

warnings.filterwarnings("ignore", category=RuntimeWarning)

def plot_results(spot, iv_surface, fitted_params, calibrator, r):
    fig, ax = plt.subplots(figsize=(12,7))
    cmap = plt.get_cmap("viridis")
    dtes = sorted(iv_surface.DaysToExpiry.unique())

    def to_iv(price, K, T):
        try:
            return brentq(lambda v: _BS.price(spot, K, T, r, v) - price, 1e-6, 3.0)
        except:
            return np.nan

    for i, dte in enumerate(dtes):
        sub = iv_surface[iv_surface.DaysToExpiry==dte].sort_values("Strike")
        T = dte/365.0
        model_prices = [calibrator._pricer.price(spot, K, T, fitted_params, otype=row.OptionType)
                        for K, row in zip(sub.Strike, sub.itertuples())]
        model_ivs = [to_iv(p, k, T) for p, k in zip(model_prices, sub.Strike)]
        color = cmap(i/len(dtes))
        ax.scatter(sub.Strike, sub.ImpliedVolatility, color=color, marker="o",
                   label=f"{dte}d (Market)")
        ax.plot(sub.Strike, model_ivs, color=color, lw=2,
                label=f"{dte}d (Model)")

    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Implied Volatility")
    ax.set_title(f"Heston Fit vs Market IV (r={r:.2%})")
    ax.legend(); ax.grid(True, alpha=0.4, linestyle="--")
    plt.show()

def test_pricing():
    """Test basic pricing functionality"""
    from heston_calibrator import HestonParams, HestonPricer, _BS
    
    # Test parameters
    S = 165.0
    K = 165.0  
    T = 30/365.0
    r = 0.05
    
    # Test Black-Scholes first
    bs_price = _BS.price(S, K, T, r, 0.3, 'call')
    print(f"BS Call Price: ${bs_price:.4f}")
    
    # Test Heston with reasonable parameters
    heston_params = HestonParams(v0=0.09, kappa=2.0, theta=0.09, sigma=0.3, rho=-0.5)
    pricer = HestonPricer(r)
    
    try:
        # Test the internal pricing function directly
        heston_price_internal = pricer._price_internal(S, K, T, heston_params)
        print(f"Heston Internal Price: ${heston_price_internal:.4f}")
        
        heston_price, _ = pricer.price_and_grad(S, K, T, heston_params)
        print(f"Heston Call Price: ${heston_price:.4f}")
        print(f"Difference: ${abs(heston_price - bs_price):.4f}")
        
        # Test if it's always falling back to BS
        if abs(heston_price - bs_price) < 1e-10:
            print("WARNING: Heston is falling back to Black-Scholes!")
            return False
        return True
    except Exception as e:
        print(f"ERROR in Heston pricing: {e}")
        return False

def main():
    TICKER = "NVDA"
    R = 0.05

    # Test pricing first
    print("Testing pricing functions...")
    if not test_pricing():
        print("Pricing test failed. Check Heston implementation.")
        return

    data = prepare_dataset(TICKER)
    price_surface = data["price_surface"]
    stock = data["stock"]
    spot = stock["Close"].iloc[-1]

    if price_surface.empty:
        print("No market prices available. Exiting.")
        return
    print(f"Using {len(price_surface)} prices across {price_surface.DaysToExpiry.nunique()} maturities.")

    # Use more realistic initial parameters
    init = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    calib = PriceHestonCalibrator(r=R)
    fit = calib.calibrate(spot, price_surface, init)

    # If you still have implied vols, fetch iv_surface similarly or drop plotting
    # Here we assume iv_surface was built earlier and stored alongside price_surface
    # iv_surface = data.get("iv_surface")
    # plot_results(spot, iv_surface, fit, calib, R)

if __name__ == "__main__":
    main()