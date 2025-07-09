import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
from numpy.polynomial import Polynomial


class CorrelatedGBM:
    def __init__(self, S0, r, cov, T, step, N, weights=None):
        """
        Initialize the correlated GBM simulator under risk-neutral measure.

        Parameters:
        S0 : np.ndarray : initial stock prices (shape: d,)
        r : float : risk-free rate (used as drift under risk-neutral measure)
        cov : np.ndarray : covariance matrix (shape: d, d)
        T : float : total time period
        step : int : number of time steps
        N : int : number of paths to generate
        weights : np.ndarray, optional : basket weights (shape: d,), defaults to equal weights
        """
        self.S0 = np.array(S0)
        self.r = r
        self.cov = np.array(cov)
        self.T = T
        self.step = step
        self.N = N
        self.d = len(S0)
        self.paths = None
        self.basket_paths = None
        
        # Initialize weights to equal weights if not provided
        if weights is None:
            self.weights = np.full(self.d, 1 / self.d)
        else:
            self.weights = np.array(weights)

    def generate_paths(self):
        """
        Generate multiple correlated geometric Brownian motion paths.

        Returns:
        np.ndarray : array of shape (N, step + 1, d) where d is number of assets
        """
        dt = self.T / self.step
        M = self.step + 1
        paths = np.zeros((self.N, M, self.d))
        paths[:, 0, :] = self.S0

        chol = np.linalg.cholesky(self.cov)

        for i in range(1, M):
            Z = np.random.normal(size=(self.N, self.d))
            correlated_Z = Z @ chol.T
            drift = (self.r - 0.5 * np.diag(self.cov)) * dt
            diffusion = correlated_Z * np.sqrt(dt)
            paths[:, i, :] = paths[:, i - 1, :] * np.exp(drift + diffusion)

        self.paths = paths
        return paths

    def get_basket_paths(self):
        """
        Get the basket price paths using the predefined weights.
        
        Returns:
        np.ndarray : array of shape (N, step + 1) representing basket price paths
        """
        if self.paths is None:
            raise ValueError("No paths generated. Call generate_paths() first.")
            
        if self.basket_paths is None:
            self.basket_paths = self.paths @ self.weights
            
        return self.basket_paths

    def plot_paths(self, basket_only=False):
        """
        Plot the generated geometric Brownian motion paths.
        
        Parameters:
        basket_only : bool : if True, only plot basket paths, otherwise plot all asset paths
        """
        if self.paths is None:
            raise ValueError("No paths generated. Call generate_paths() first.")

        N, M, d = self.paths.shape
        time = np.linspace(0, self.T, M)

        if basket_only:
            # Plot only basket paths
            plt.figure(figsize=(10, 6))
            basket_paths = self.get_basket_paths()
            for j in range(N):
                plt.plot(time, basket_paths[j, :], alpha=0.5)
            plt.title('Basket Price Paths')
            plt.xlabel('Time (years)')
            plt.ylabel('Basket Price')
            plt.grid()
            plt.show()
        else:
            # Plot all individual asset paths
            for i in range(d):
                plt.figure(figsize=(10, 6))
                for j in range(N):
                    plt.plot(time, self.paths[j, :, i], alpha=0.5)
                plt.title(f'Geometric Brownian Motion Paths for Asset {i + 1}')
                plt.xlabel('Time (years)')
                plt.ylabel('Price')
                plt.grid()
                plt.show()

# Base class for regression basis
class RegressionBasis:
    def fit(self, x, y):
        """Fit the basis functions to the data"""
        raise NotImplementedError("Subclasses must implement fit method")
    
    def value(self, x):
        """Evaluate the basis functions at the given points"""
        raise NotImplementedError("Subclasses must implement value method")


class PolynomialBasis(RegressionBasis):
    def __init__(self, degree=3):
        self.degree = degree
        self.coef = None
    
    def fit(self, x, y):
        self.coef = Polynomial.fit(x, y, deg=self.degree).convert().coef
        return self
    
    def value(self, x):
        return np.polyval(self.coef[::-1], x)


# Option pricer class using Longstaff-Schwartz method
class LSMOptionPricer:
    def __init__(self, 
                 S0: np.ndarray,
                 cov: np.ndarray,
                 T: float,
                 step: int,
                 N: int,
                 r: float,
                 K: float,
                 weights: np.ndarray = None,
                 option: Literal["call", "put"] = "call"):
        """
        Initialize the Longstaff-Schwartz option pricer.
        
        Parameters
        ----------
        S0 : np.ndarray
            Initial stock prices (shape: d,)
        cov : np.ndarray
            Covariance matrix (shape: d, d)
        T : float
            Total time period
        step : int
            Number of time steps
        N : int
            Number of paths to generate
        r : float
            Risk-free rate (continuously compounded) - used as drift under risk-neutral measure
        K : float
            Strike of the basket option
        weights : np.ndarray, optional
            Basket weights (shape d,).  Defaults to equal weights.
        option : {"call", "put"}
            Pay-off type.
        """
        self.S0 = S0
        self.cov = cov
        self.T = T
        self.step = step
        self.N = N
        self.r = r
        self.K = K
        self.weights = weights
        self.option = option
        self.sim = None
        self.basket_paths = None
        self.payoff = None
        self.dt = T / step
        self.disc = np.exp(-r * self.dt)
        
    def simulate_paths(self):
        """
        Generate price paths for the basket option under the risk-neutral measure.
        
        Uses the risk-free rate for drift under risk-neutral pricing.
        """
        # Create GBM simulator with risk-free rate (risk-neutral measure)
        self.sim = CorrelatedGBM(self.S0, self.r, self.cov, self.T, self.step, self.N, self.weights)
        self.sim.generate_paths()
        self.basket_paths = self.sim.get_basket_paths()
        
        # Calculate payoff at each time point
        if self.option == "call":
            self.payoff = np.maximum(self.basket_paths - self.K, 0.0)
        else:
            self.payoff = np.maximum(self.K - self.basket_paths, 0.0)
        
        return self.basket_paths
    
    def price(self, 
              basis=None, 
              exercise_grid=None):
        """
        Price the option using Longstaff-Schwartz method.
        
        Parameters
        ----------
        basis : RegressionBasis, optional
            Regression basis to use. Defaults to 3rd degree polynomial.
        exercise_grid : tuple[int], optional
            Indices (1 … step−1) of time points where early exercise is allowed.
            Default = every step (true American).
            
        Returns
        -------
        price : float
            Monte-Carlo estimate of the option value at t = 0.
        """
        if self.basket_paths is None:
            self.simulate_paths()
            
        if basis is None:
            basis = PolynomialBasis(degree=3)
            
        Npaths, M = self.basket_paths.shape
        
        # Default exercise = every step except t=0, T (American)
        if exercise_grid is None:
            exercise_grid = tuple(range(1, M))
            
        # Start backward induction from maturity
        cashflow = self.payoff[:, -1].copy()
        
        for t in reversed(exercise_grid):
            itm = self.payoff[:, t] > 0  # in-the-money paths
            
            if np.any(itm):
                # Regression of discounted CF on basis functions
                X = self.basket_paths[itm, t]
                Y = cashflow[itm] * self.disc  # discount one step
                basis.fit(X, Y)
                cont_val = basis.value(X)
                
                # Optimal exercise decision
                exercise = self.payoff[itm, t] >= cont_val
                cashflow[itm] = np.where(exercise, self.payoff[itm, t], cashflow[itm] * self.disc)
            else:
                # Nobody is ITM → just discount
                cashflow *= self.disc
                
        # Take expectation and discount final step to t=0
        price = np.mean(cashflow) * self.disc
        return price


# Example usage
if __name__ == "__main__":
    # Parameters for 30 assets
    n_assets = 30
    S0 = [100] * n_assets
    r = 0.03  # Risk-free rate (used for both drift and discounting in risk-neutral pricing)
    sigma = [0.2 + i * 0.01 for i in range(n_assets)]  # Different volatilities

    # Create correlation matrix (block correlation structure)
    base_corr = np.eye(n_assets)
    # First block: high correlation (0.7)
    base_corr[:10, :10] = 0.7
    np.fill_diagonal(base_corr[:10, :10], 1.0)
    # Second block: medium correlation (0.4)
    base_corr[10:20, 10:20] = 0.4
    np.fill_diagonal(base_corr[10:20, 10:20], 1.0)
    # Third block: low correlation (0.2)
    base_corr[20:, 20:] = 0.2
    np.fill_diagonal(base_corr[20:, 20:], 1.0)

    # Build covariance matrix
    cov = np.zeros((n_assets, n_assets))
    for i in range(n_assets):
        for j in range(n_assets):
            cov[i, j] = base_corr[i, j] * sigma[i] * sigma[j]

    T = 3 / 12  # 3 months in years
    step = 90  # 90 steps over 3 months
    N = 500  # Number of simulations

    print(f"Simulating {n_assets} assets over {T*12} months with {step} steps")
    
    # Create simulator with equal weights
    weights = np.ones(n_assets) / n_assets
    
    # Create risk-neutral simulation
    print("\n--- Risk-Neutral GBM Simulation ---")
    
    # Create simulator and generate paths
    gbm_sim = CorrelatedGBM(S0, r, cov, T, step, N, weights)
    gbm_sim.generate_paths()
    
    # Calculate average final basket price
    risk_neutral_paths = gbm_sim.get_basket_paths()
    
    print(f"Average final basket price (risk-neutral, r = {r}): {np.mean(risk_neutral_paths[:, -1]):.2f}")
    print(f"Expected risk-neutral price: {S0[0] * np.exp(r * T):.2f}")
    
    # Plot the risk-neutral basket path
    print("\nPlotting basket path (risk-neutral)...")
    gbm_sim.plot_paths(basket_only=True)
    
    # Price options using the LSMOptionPricer class
    print("\n--- Using the LSMOptionPricer class ---")
    
    # Create the pricer object
    pricer = LSMOptionPricer(
        S0=S0, cov=cov, T=T, step=step, N=N,
        r=r, K=100, weights=weights, option="put"
    )
    
    # Generate paths and price American option
    pricer.simulate_paths()
    american_price_cls = pricer.price(basis=PolynomialBasis(degree=3))
    print(f"American basket put price (using class): {american_price_cls:.4f}")
    
    # Price European option (exercise only at maturity)
    european_price_cls = pricer.price(exercise_grid=(step,))
    print(f"European basket put price (using class): {european_price_cls:.4f}")
    print(f"Early exercise premium: {american_price_cls - european_price_cls:.4f}")
