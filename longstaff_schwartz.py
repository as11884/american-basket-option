import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
from numpy.polynomial import Polynomial


class CorrelatedHeston:
    def __init__(self, S0, r, v0, theta, kappa, sigma, rho_sv, corr_matrix, T, step, N, weights=None, var_corr_matrix=None):
        """
        Initialize the correlated Heston model simulator under risk-neutral measure.

        Parameters:
        S0 : np.ndarray : initial stock prices (shape: d,)
        r : float : risk-free rate
        v0 : np.ndarray : initial variance for each asset (shape: d,)
        theta : np.ndarray : long-term variance for each asset (shape: d,)
        kappa : np.ndarray : mean reversion speed for each asset (shape: d,)
        sigma : np.ndarray : volatility of volatility for each asset (shape: d,)
        rho_sv : np.ndarray : correlation between stock and variance for each asset (shape: d,)
        corr_matrix : np.ndarray : correlation matrix between STOCK prices (shape: d, d)
        T : float : total time period
        step : int : number of time steps
        N : int : number of paths to generate
        weights : np.ndarray, optional : basket weights (shape: d,), defaults to equal weights
        var_corr_matrix : np.ndarray, optional : correlation matrix between VARIANCE processes (shape: d, d)
                         If None, defaults to identity matrix (no variance-to-variance correlations)
                         This assumes variance processes are independent across assets
        """
        self.S0 = np.array(S0)
        self.r = r
        self.v0 = np.array(v0)
        self.theta = np.array(theta)
        self.kappa = np.array(kappa)
        self.sigma = np.array(sigma)
        self.rho_sv = np.array(rho_sv)
        self.corr_matrix = np.array(corr_matrix)
        self.T = T
        self.step = step
        self.N = N
        self.d = len(S0)
        self.paths = None
        self.variance_paths = None
        self.basket_paths = None
        
        # Store variance correlation matrix (default to identity = no correlations)
        if var_corr_matrix is None:
            self.var_corr_matrix = np.eye(self.d)  # No variance-to-variance correlations
        else:
            self.var_corr_matrix = np.array(var_corr_matrix)
        
        # Initialize weights to equal weights if not provided
        if weights is None:
            self.weights = np.full(self.d, 1 / self.d)
        else:
            self.weights = np.array(weights)

    def generate_paths(self):
        """
        Generate multiple correlated Heston model paths using Euler discretization.

        Returns:
        np.ndarray : array of shape (N, step + 1, d) where d is number of assets
        """
        dt = self.T / self.step
        M = self.step + 1
        
        # Initialize arrays
        S_paths = np.zeros((self.N, M, self.d))
        V_paths = np.zeros((self.N, M, self.d))
        
        # Set initial conditions
        S_paths[:, 0, :] = self.S0
        V_paths[:, 0, :] = self.v0
        
        # Cholesky decomposition for stock correlations
        L_stock = np.linalg.cholesky(self.corr_matrix)
        
        # Cholesky decomposition for variance correlations (default = identity = no correlations)
        L_var = np.linalg.cholesky(self.var_corr_matrix)
        
        # Create cross-correlation matrix between stock and variance
        # This incorporates the stock-variance correlation (rho_sv) for each asset
        sqrt_dt = np.sqrt(dt)
        
        for i in range(1, M):
            # Generate independent normal random variables
            Z_S = np.random.normal(size=(self.N, self.d))
            Z_V = np.random.normal(size=(self.N, self.d))
            
            # Apply stock correlations
            W_S = Z_S @ L_stock.T
            
            # Apply variance correlations and stock-variance correlation
            # Create correlated variance shocks
            W_V = np.zeros_like(Z_V)
            for j in range(self.d):
                # Correlated component with stock
                rho_j = self.rho_sv[j]
                W_V[:, j] = rho_j * W_S[:, j] + np.sqrt(1 - rho_j**2) * Z_V[:, j]
            
            # Apply variance cross-correlations
            W_V = W_V @ L_var.T
            
            # Euler discretization for each asset
            for j in range(self.d):
                # Variance process (with reflection at zero)
                V_prev = np.maximum(V_paths[:, i-1, j], 0)
                dV = self.kappa[j] * (self.theta[j] - V_prev) * dt + \
                     self.sigma[j] * np.sqrt(V_prev) * W_V[:, j] * sqrt_dt
                V_paths[:, i, j] = np.maximum(V_prev + dV, 0)  # Reflection at zero
                
                # Stock process
                S_prev = S_paths[:, i-1, j]
                sqrt_V = np.sqrt(np.maximum(V_prev, 0))
                dS = self.r * S_prev * dt + sqrt_V * S_prev * W_S[:, j] * sqrt_dt
                S_paths[:, i, j] = S_prev + dS

        self.paths = S_paths
        self.variance_paths = V_paths
        return S_paths

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

    def get_variance_paths(self):
        """
        Get the variance paths for all assets.
        
        Returns:
        np.ndarray : array of shape (N, step + 1, d) representing variance paths
        """
        if self.variance_paths is None:
            raise ValueError("No paths generated. Call generate_paths() first.")
            
        return self.variance_paths

    def plot_paths(self, basket_only=False, plot_variance=False):
        """
        Plot the generated Heston model paths.
        
        Parameters:
        basket_only : bool : if True, only plot basket paths, otherwise plot all asset paths
        plot_variance : bool : if True, also plot variance paths
        """
        if self.paths is None:
            raise ValueError("No paths generated. Call generate_paths() first.")

        N, M, d = self.paths.shape
        time = np.linspace(0, self.T, M)

        if basket_only:
            # Plot only basket paths
            plt.figure(figsize=(12, 8))
            
            if plot_variance:
                plt.subplot(2, 1, 1)
            
            basket_paths = self.get_basket_paths()
            for j in range(min(N, 50)):  # Plot max 50 paths for clarity
                plt.plot(time, basket_paths[j, :], alpha=0.5)
            plt.title('Basket Price Paths (Heston Model)')
            plt.xlabel('Time (years)')
            plt.ylabel('Basket Price')
            plt.grid()
            
            if plot_variance:
                plt.subplot(2, 1, 2)
                # Plot average variance across all assets
                avg_variance = np.mean(self.variance_paths, axis=2)
                for j in range(min(N, 50)):
                    plt.plot(time, avg_variance[j, :], alpha=0.5)
                plt.title('Average Variance Paths')
                plt.xlabel('Time (years)')
                plt.ylabel('Variance')
                plt.grid()
                
            plt.tight_layout()
            plt.show()
        else:
            # Plot all individual asset paths
            for i in range(d):
                plt.figure(figsize=(12, 8))
                
                if plot_variance:
                    plt.subplot(2, 1, 1)
                
                for j in range(min(N, 20)):  # Plot max 20 paths per asset
                    plt.plot(time, self.paths[j, :, i], alpha=0.5)
                plt.title(f'Heston Price Paths for Asset {i + 1}')
                plt.xlabel('Time (years)')
                plt.ylabel('Price')
                plt.grid()
                
                if plot_variance:
                    plt.subplot(2, 1, 2)
                    for j in range(min(N, 20)):
                        plt.plot(time, self.variance_paths[j, :, i], alpha=0.5)
                    plt.title(f'Variance Paths for Asset {i + 1}')
                    plt.xlabel('Time (years)')
                    plt.ylabel('Variance')
                    plt.grid()
                
                plt.tight_layout()
                plt.show()


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
                 T: float,
                 step: int,
                 N: int,
                 r: float,
                 K: float,
                 weights: np.ndarray = None,
                 option: Literal["call", "put"] = "call",
                 model: Literal["gbm", "heston"] = "gbm",
                 # GBM parameters
                 cov: np.ndarray = None,
                 # Heston parameters
                 v0: np.ndarray = None,
                 theta: np.ndarray = None,
                 kappa: np.ndarray = None,
                 sigma: np.ndarray = None,
                 rho_sv: np.ndarray = None,
                 corr_matrix: np.ndarray = None):
        """
        Initialize the Longstaff-Schwartz option pricer.
        
        Parameters
        ----------
        S0 : np.ndarray
            Initial stock prices (shape: d,)
        T : float
            Total time period
        step : int
            Number of time steps
        N : int
            Number of paths to generate
        r : float
            Risk-free rate (continuously compounded)
        K : float
            Strike of the basket option
        weights : np.ndarray, optional
            Basket weights (shape d,). Defaults to equal weights.
        option : {"call", "put"}
            Pay-off type.
        model : {"gbm", "heston"}
            Underlying model to use for simulation.
        
        GBM Parameters
        --------------
        cov : np.ndarray
            Covariance matrix (shape: d, d) - required for GBM model
            
        Heston Parameters
        -----------------
        v0 : np.ndarray
            Initial variance for each asset (shape: d,) - required for Heston
        theta : np.ndarray
            Long-term variance for each asset (shape: d,) - required for Heston
        kappa : np.ndarray
            Mean reversion speed for each asset (shape: d,) - required for Heston
        sigma : np.ndarray
            Volatility of volatility for each asset (shape: d,) - required for Heston
        rho_sv : np.ndarray
            Correlation between stock and variance for each asset (shape: d,) - required for Heston
        corr_matrix : np.ndarray
            Correlation matrix between assets (shape: d, d) - required for Heston
        """
        self.S0 = S0
        self.T = T
        self.step = step
        self.N = N
        self.r = r
        self.K = K
        self.weights = weights
        self.option = option
        self.model = model
        
        # Model-specific parameters
        self.cov = cov
        self.v0 = v0
        self.theta = theta
        self.kappa = kappa
        self.sigma = sigma
        self.rho_sv = rho_sv
        self.corr_matrix = corr_matrix
        
        self.sim = None
        self.basket_paths = None
        self.payoff = None
        self.dt = T / step
        self.disc = np.exp(-r * self.dt)
        
        # Validate parameters based on model choice
        if model == "gbm" and cov is None:
            raise ValueError("Covariance matrix 'cov' is required for GBM model")
        elif model == "heston":
            heston_params = [v0, theta, kappa, sigma, rho_sv, corr_matrix]
            if any(param is None for param in heston_params):
                raise ValueError("All Heston parameters (v0, theta, kappa, sigma, rho_sv, corr_matrix) are required for Heston model")
        
    def simulate_paths(self):
        """
        Generate price paths for the basket option under the risk-neutral measure.
        
        Uses either GBM or Heston model based on the model parameter.
        """
        if self.model == "gbm":
            # Create GBM simulator with risk-free rate (risk-neutral measure)
            self.sim = CorrelatedGBM(self.S0, self.r, self.cov, self.T, self.step, self.N, self.weights)
        elif self.model == "heston":
            # Create Heston simulator
            self.sim = CorrelatedHeston(
                self.S0, self.r, self.v0, self.theta, self.kappa, 
                self.sigma, self.rho_sv, self.corr_matrix, 
                self.T, self.step, self.N, self.weights
            )
        else:
            raise ValueError(f"Unknown model: {self.model}")
            
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
    r = 0.03  # Risk-free rate
    
    # GBM parameters
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

    # Build covariance matrix for GBM
    cov = np.zeros((n_assets, n_assets))
    for i in range(n_assets):
        for j in range(n_assets):
            cov[i, j] = base_corr[i, j] * sigma[i] * sigma[j]

    # Heston parameters (different for each asset)
    v0 = np.array([sigma[i]**2 for i in range(n_assets)])  # Initial variance = sigma^2
    theta = np.array([0.04 + i * 0.002 for i in range(n_assets)])  # Long-term variance
    kappa = np.array([2.0 + i * 0.1 for i in range(n_assets)])  # Mean reversion speed
    sigma_v = np.array([0.3 + i * 0.01 for i in range(n_assets)])  # Vol of vol
    rho_sv = np.array([-0.7 + i * 0.02 for i in range(n_assets)])  # Stock-vol correlation

    T = 3 / 12  # 3 months in years
    step = 90  # 90 steps over 3 months
    N = 500  # Number of simulations

    print(f"Simulating {n_assets} assets over {T*12} months with {step} steps")
    
    # Create simulator with equal weights
    weights = np.ones(n_assets) / n_assets
    
    # ===============================
    # GBM Simulation
    # ===============================
    print("\n" + "="*50)
    print("GBM SIMULATION")
    print("="*50)
    
    # Create GBM simulator and generate paths
    gbm_sim = CorrelatedGBM(S0, r, cov, T, step, N, weights)
    gbm_sim.generate_paths()
    
    # Calculate average final basket price
    gbm_paths = gbm_sim.get_basket_paths()
    
    print(f"Average final basket price (GBM): {np.mean(gbm_paths[:, -1]):.2f}")
    print(f"Expected theoretical price: {S0[0] * np.exp(r * T):.2f}")
    
    # ===============================
    # Heston Simulation
    # ===============================
    print("\n" + "="*50)
    print("HESTON SIMULATION")
    print("="*50)
    
    # Create Heston simulator and generate paths
    heston_sim = CorrelatedHeston(
        S0, r, v0, theta, kappa, sigma_v, rho_sv, base_corr, T, step, N, weights
    )
    heston_sim.generate_paths()
    
    # Calculate average final basket price
    heston_paths = heston_sim.get_basket_paths()
    
    print(f"Average final basket price (Heston): {np.mean(heston_paths[:, -1]):.2f}")
    print(f"Average final variance: {np.mean(heston_sim.get_variance_paths()[:, -1, :]):.4f}")
    
    # ===============================
    # Option Pricing Comparison
    # ===============================
    print("\n" + "="*50)
    print("OPTION PRICING COMPARISON")
    print("="*50)
    
    # Price options using GBM
    print("\n--- GBM Option Pricing ---")
    gbm_pricer = LSMOptionPricer(
        S0=S0, T=T, step=step, N=N, r=r, K=100, 
        weights=weights, option="put", model="gbm", cov=cov
    )
    
    gbm_pricer.simulate_paths()
    gbm_american_price = gbm_pricer.price(basis=PolynomialBasis(degree=3))
    gbm_european_price = gbm_pricer.price(exercise_grid=(step,))
    
    print(f"American basket put price (GBM): {gbm_american_price:.4f}")
    print(f"European basket put price (GBM): {gbm_european_price:.4f}")
    print(f"Early exercise premium (GBM): {gbm_american_price - gbm_european_price:.4f}")
    
    # Price options using Heston
    print("\n--- Heston Option Pricing ---")
    heston_pricer = LSMOptionPricer(
        S0=S0, T=T, step=step, N=N, r=r, K=100, 
        weights=weights, option="put", model="heston",
        v0=v0, theta=theta, kappa=kappa, sigma=sigma_v, 
        rho_sv=rho_sv, corr_matrix=base_corr
    )
    
    heston_pricer.simulate_paths()
    heston_american_price = heston_pricer.price(basis=PolynomialBasis(degree=3))
    heston_european_price = heston_pricer.price(exercise_grid=(step,))
    
    print(f"American basket put price (Heston): {heston_american_price:.4f}")
    print(f"European basket put price (Heston): {heston_european_price:.4f}")
    print(f"Early exercise premium (Heston): {heston_american_price - heston_european_price:.4f}")
    
    # Compare models
    print("\n--- Model Comparison ---")
    print(f"Price difference (Heston - GBM): {heston_american_price - gbm_american_price:.4f}")
    print(f"Relative difference: {((heston_american_price - gbm_american_price) / gbm_american_price * 100):.2f}%")
    
    # Plot comparison
    print("\n--- Plotting Path Comparison ---")
    
    # Plot basket paths comparison
    plt.figure(figsize=(15, 5))
    time = np.linspace(0, T, step + 1)
    
    plt.subplot(1, 3, 1)
    for j in range(min(N, 50)):
        plt.plot(time, gbm_paths[j, :], alpha=0.3, color='blue')
    plt.title('GBM Basket Paths')
    plt.xlabel('Time (years)')
    plt.ylabel('Basket Price')
    plt.grid()
    
    plt.subplot(1, 3, 2)
    for j in range(min(N, 50)):
        plt.plot(time, heston_paths[j, :], alpha=0.3, color='red')
    plt.title('Heston Basket Paths')
    plt.xlabel('Time (years)')
    plt.ylabel('Basket Price')
    plt.grid()
    
    plt.subplot(1, 3, 3)
    # Plot average variance for Heston
    avg_variance = np.mean(heston_sim.get_variance_paths(), axis=2)
    for j in range(min(N, 50)):
        plt.plot(time, avg_variance[j, :], alpha=0.3, color='green')
    plt.title('Heston Average Variance Paths')
    plt.xlabel('Time (years)')
    plt.ylabel('Variance')
    plt.grid()
    
    plt.tight_layout()
    plt.show()
    
    print("\nSimulation complete!")
    print(f"Heston model shows {'higher' if heston_american_price > gbm_american_price else 'lower'} option prices due to stochastic volatility effects.")
