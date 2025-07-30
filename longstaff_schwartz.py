import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Dict, Any, Tuple, Optional
from numpy.polynomial import Polynomial
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# ---------------------------
# Regression basis
# ---------------------------

class RegressionBasis:
    def fit(self, x, y):
        raise NotImplementedError("Subclasses must implement fit")
    def value(self, x):
        raise NotImplementedError("Subclasses must implement value")


class PolynomialBasis(RegressionBasis):
    def __init__(self, degree=4):
        self.degree = degree
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.model = LinearRegression()

    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        self.model.fit(X_poly, y)
        return self

    def value(self, X):
        X_poly = self.poly.transform(X)
        return self.model.predict(X_poly)


# ---------------------------
# Utilities
# ---------------------------

def _chol_block_correlation(corr_S: np.ndarray,
                            corr_V: np.ndarray,
                            rho_sv: np.ndarray,
                            jitter: float = 1e-12) -> np.ndarray:
    """
    Build 2d x 2d correlation matrix and return its Cholesky factor.
    C = [[C_SS, C_SV],
         [C_SV^T, C_VV]],
    where C_SV is diagonal with rho_sv.
    """
    d = corr_S.shape[0]
    C_SS = np.array(corr_S, dtype=float)
    C_VV = np.array(corr_V, dtype=float)
    C_SV = np.diag(rho_sv.astype(float))

    top = np.hstack([C_SS, C_SV])
    bot = np.hstack([C_SV.T, C_VV])
    C = np.vstack([top, bot])

    # Ensure symmetry and PD (add a tiny jitter on diag if needed)
    C = 0.5 * (C + C.T)
    # Jitter loop if ill-conditioned
    k = 0
    while True:
        try:
            L = np.linalg.cholesky(C)
            return L
        except np.linalg.LinAlgError:
            # Add jitter and retry
            C = C + (jitter * (10 ** k)) * np.eye(2 * d)
            k += 1
            if k > 6:
                raise ValueError("Block correlation is not positive definite even after jitter.")


def _rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------
# Simulators
# ---------------------------

class CorrelatedHeston:
    def __init__(self, S0, r, v0, theta, kappa, sigma, rho_sv, corr_matrix, T, step, N, weights=None, var_corr_matrix=None):
        """
        Multi-asset Heston under risk-neutral measure using per-asset stock–variance coupling.

        Construction (per time step):
          1) Generate stock shocks W_S with Corr(W_S) = corr_matrix.
          2) For each asset j, set W_V,j = rho_j * W_S,j + sqrt(1 - rho_j^2) * Z_V,j
             where Z_V is i.i.d. standard normal and independent of W_S.
        This guarantees Corr(W_S,j, W_V,j) = rho_j and Corr(W_S) = corr_matrix.
        There is NO cross-asset mixing of W_V (i.e., no variance–variance correlation imposed).

        Parameters:
            S0 : (d,) initial stock prices
            r  : risk-free rate (cont. comp.)
            v0, theta, kappa, sigma, rho_sv : (d,) Heston parameters (rho_sv are per-asset S–v correlations)
            corr_matrix : (d,d) stock–stock correlation matrix
            T : total time
            step : number of time steps
            N : number of paths
            weights : optional basket weights (defaults to equal)
            var_corr_matrix : kept for API compatibility but UNUSED in this construction
        """
        self.S0 = np.array(S0)
        self.r = float(r)
        self.v0 = np.array(v0)
        self.theta = np.array(theta)
        self.kappa = np.array(kappa)
        self.sigma = np.array(sigma)
        self.rho_sv = np.array(rho_sv)
        self.corr_matrix = np.array(corr_matrix)
        self.T = float(T)
        self.step = int(step)
        self.N = int(N)
        self.d = len(S0)
        self.paths = None
        self.variance_paths = None
        self.basket_paths = None

        # NOTE: var_corr_matrix is intentionally ignored in this implementation
        self.var_corr_matrix = var_corr_matrix

        # Initialize weights (equal-weight basket if not provided)
        if weights is None:
            self.weights = np.full(self.d, 1.0 / self.d)
        else:
            self.weights = np.array(weights)

    def generate_paths(self, seed: int | None = None):
        """
        Generate paths using Euler discretization with full truncation for variance.
        """
        rng = np.random.default_rng(seed)
        dt = self.T / self.step
        M = self.step + 1
        sqrt_dt = np.sqrt(dt)

        S_paths = np.zeros((self.N, M, self.d))
        V_paths = np.zeros((self.N, M, self.d))

        S_paths[:, 0, :] = self.S0
        V_paths[:, 0, :] = self.v0

        # Cholesky for stock correlations only
        L_stock = np.linalg.cholesky(self.corr_matrix)

        for i in range(1, M):
            # Independent normals
            Z_S = rng.standard_normal(size=(self.N, self.d))
            Z_V = rng.standard_normal(size=(self.N, self.d))

            # Stock shocks with desired cross-asset correlation
            W_S = Z_S @ L_stock.T  # shape (N, d)

            # Per-asset variance shocks correlated with own stock shock
            # W_V[:, j] = rho_j * W_S[:, j] + sqrt(1 - rho_j^2) * Z_V[:, j]
            W_V = np.empty_like(W_S)
            for j in range(self.d):
                rho_j = self.rho_sv[j]
                W_V[:, j] = rho_j * W_S[:, j] + np.sqrt(max(0.0, 1.0 - rho_j**2)) * Z_V[:, j]

            # Euler updates (full truncation on v)
            for j in range(self.d):
                V_prev = V_paths[:, i - 1, j]
                V_pos = np.maximum(V_prev, 0.0)
                dV = self.kappa[j] * (self.theta[j] - V_pos) * dt + self.sigma[j] * np.sqrt(V_pos) * W_V[:, j] * sqrt_dt
                V_new = np.maximum(V_prev + dV, 0.0)
                V_paths[:, i, j] = V_new

                S_prev = S_paths[:, i - 1, j]
                sqrt_V = np.sqrt(np.maximum(V_pos, 0.0))
                dS = self.r * S_prev * dt + sqrt_V * S_prev * W_S[:, j] * sqrt_dt
                S_paths[:, i, j] = S_prev + dS

        self.paths = S_paths
        self.variance_paths = V_paths
        return S_paths

    def get_basket_paths(self, kind: Literal["geometric", "arithmetic"] = "geometric"):
        """
        Basket paths with configurable averaging method.
        
        Parameters:
            kind: "geometric" or "arithmetic" averaging method
        """
        if self.paths is None:
            raise ValueError("No paths generated. Call generate_paths() first.")

        if self.basket_paths is None or getattr(self, "_basket_kind", None) != kind:
            w = self.weights / np.sum(self.weights)
            if kind == "geometric":
                # Weighted geometric mean: exp(sum_i w_i log S_i)
                log_prices = np.log(self.paths)            # (N, M, d)
                weighted_mean_log = np.tensordot(log_prices, w, axes=([2], [0]))  # (N, M)
                self.basket_paths = np.exp(weighted_mean_log)
            elif kind == "arithmetic":
                # Weighted arithmetic mean: sum_i w_i S_i
                self.basket_paths = np.tensordot(self.paths, w, axes=([2], [0]))
            else:
                raise ValueError("Unknown basket kind. Use 'geometric' or 'arithmetic'.")
            self._basket_kind = kind

        return self.basket_paths

    def get_variance_paths(self):
        if self.variance_paths is None:
            raise ValueError("No paths generated. Call generate_paths() first.")
        return self.variance_paths

    def plot_paths(self, basket_only=False, plot_variance=False):
        if self.paths is None:
            raise ValueError("No paths generated. Call generate_paths() first.")

        N, M, d = self.paths.shape
        time = np.linspace(0, self.T, M)

        if basket_only:
            plt.figure(figsize=(12, 8))
            if plot_variance:
                plt.subplot(2, 1, 1)

            basket_paths = self.get_basket_paths()
            for j in range(min(N, 50)):
                plt.plot(time, basket_paths[j, :], alpha=0.5)
            plt.title('Geometric Average Basket Paths (Heston)')
            plt.xlabel('Time (years)')
            plt.ylabel('Basket Price')
            plt.grid()

            if plot_variance:
                plt.subplot(2, 1, 2)
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
            for i in range(d):
                plt.figure(figsize=(12, 8))
                if plot_variance:
                    plt.subplot(2, 1, 1)
                for j in range(min(N, 20)):
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
        Multi-asset GBM under risk-neutral measure.
        cov: instantaneous covariance matrix of log-returns (d x d).
        """
        self.S0 = np.array(S0, dtype=float)
        self.r = float(r)
        self.cov = np.array(cov, dtype=float)
        self.T = float(T)
        self.step = int(step)
        self.N = int(N)
        self.d = len(self.S0)

        if weights is None:
            self.weights = np.full(self.d, 1.0 / self.d)
        else:
            self.weights = np.array(weights, dtype=float)

        self.paths: Optional[np.ndarray] = None
        self.basket_paths: Optional[np.ndarray] = None

    def generate_paths(self, seed: Optional[int] = None):
        dt = self.T / self.step
        M = self.step + 1
        paths = np.zeros((self.N, M, self.d), dtype=float)
        paths[:, 0, :] = self.S0

        chol = np.linalg.cholesky(self.cov)
        rng = _rng(seed)

        for i in range(1, M):
            Z = rng.standard_normal(size=(self.N, self.d))
            correlated_Z = Z @ chol.T
            drift = (self.r - 0.5 * np.diag(self.cov)) * dt  # (d,)
            diffusion = correlated_Z * np.sqrt(dt)           # (N,d)
            paths[:, i, :] = paths[:, i - 1, :] * np.exp(drift + diffusion)

        self.paths = paths
        self.basket_paths = None
        return paths

    def get_basket_paths(self,
                         kind: Literal["geometric", "arithmetic"] = "geometric"):
        if self.paths is None:
            raise ValueError("No paths generated. Call generate_paths() first.")

        if self.basket_paths is None or getattr(self, "_basket_kind", None) != kind:
            w = self.weights / np.sum(self.weights)
            if kind == "geometric":
                logS = np.log(self.paths)
                weighted_mean_log = np.tensordot(logS, w, axes=([2], [0]))
                self.basket_paths = np.exp(weighted_mean_log)
            elif kind == "arithmetic":
                self.basket_paths = np.tensordot(self.paths, w, axes=([2], [0]))
            else:
                raise ValueError("Unknown basket kind.")
            self._basket_kind = kind
        return self.basket_paths


# ---------------------------
# LSM pricer
# ---------------------------

class LSMOptionPricer:
    def __init__(self,
                 S0: np.ndarray,
                 T: float,
                 step: int,
                 N: int,
                 r: float,
                 K: float,
                 weights: np.ndarray = None,
                 option: Literal["call", "put"] = "put",
                 model: Literal["gbm", "heston"] = "gbm",
                 basket_kind: Literal["geometric", "arithmetic"] = "geometric",
                 include_variance_state: bool = False,
                 degree: int = 4,  # Polynomial degree for LSM regression
                 # GBM
                 cov: np.ndarray = None,
                 # Heston
                 v0: np.ndarray = None,
                 theta: np.ndarray = None,
                 kappa: np.ndarray = None,
                 sigma: np.ndarray = None,
                 rho_sv: np.ndarray = None,
                 corr_matrix: np.ndarray = None,
                 var_corr_matrix: np.ndarray = None):

        self.S0 = np.array(S0, dtype=float)
        self.T = float(T)
        self.step = int(step)
        self.N = int(N)
        self.r = float(r)
        self.K = float(K)
        self.weights = np.array(weights, dtype=float) if weights is not None else None
        self.option = option
        self.model = model
        self.basket_kind = basket_kind
        self.include_variance_state = include_variance_state
        self.degree = int(degree)  # Store polynomial degree as instance variable

        # Model params
        self.cov = cov
        self.v0 = v0
        self.theta = theta
        self.kappa = kappa
        self.sigma = sigma
        self.rho_sv = rho_sv
        self.corr_matrix = corr_matrix
        self.var_corr_matrix = var_corr_matrix

        # Validations
        if model == "gbm":
            if cov is None:
                raise ValueError("GBM requires 'cov'.")
        elif model == "heston":
            for name, param in dict(v0=v0, theta=theta, kappa=kappa, sigma=sigma, rho_sv=rho_sv, corr_matrix=corr_matrix).items():
                if param is None:
                    raise ValueError(f"Heston requires '{name}'.")
        else:
            raise ValueError("Unknown model.")

        self.sim = None
        self.paths = None
        self.basket_paths = None
        self.variance_paths = None
        self.payoff = None

        self.dt = self.T / self.step
        self.disc = np.exp(-self.r * self.dt)

        # For trained models
        self.trained_models: Dict[int, RegressionBasis] = {}
        self.exercise_grid = None
        self.use_individual_prices = True
        self._trained_flags = {}

    def _make_simulator(self):
        if self.model == "gbm":
            sim = CorrelatedGBM(self.S0,
                                self.r,
                                self.cov,
                                self.T,
                                self.step,
                                self.N,
                                weights=self.weights)
        else:
            sim = CorrelatedHeston(self.S0,
                                   self.r,
                                   self.v0, self.theta, self.kappa, self.sigma, self.rho_sv,
                                   self.corr_matrix,
                                   self.T, self.step, self.N,
                                   weights=self.weights,
                                   var_corr_matrix=self.var_corr_matrix)
        return sim

    def simulate_paths(self, seed: Optional[int] = None):
        self.sim = self._make_simulator()
        self.sim.generate_paths(seed=seed)
        self.paths = self.sim.paths
        self.variance_paths = getattr(self.sim, "variance_paths", None)
        self.basket_paths = self.sim.get_basket_paths(kind=self.basket_kind)

        # Pathwise payoff at each time slice
        if self.option == "call":
            self.payoff = np.maximum(self.basket_paths - self.K, 0.0)
        else:
            self.payoff = np.maximum(self.K - self.basket_paths, 0.0)
        return self.basket_paths

    def _state_at(self, t: int, itm_mask: np.ndarray, use_individual_prices: bool):
        """Return regression features X at time t for in-the-money paths."""
        if use_individual_prices:
            X = self.paths[itm_mask, t, :]  # (n_itm, d)
            if self.include_variance_state and self.model == "heston" and self.variance_paths is not None:
                V = self.variance_paths[itm_mask, t, :]  # (n_itm, d)
                X = np.concatenate([X, V], axis=1)       # include vol state
            return X
        else:
            return self.basket_paths[itm_mask, t]        # (n_itm,)

    def _lsm_core(self,
                  exercise_grid: Optional[Tuple[int, ...]],
                  use_individual_prices: bool):
        """
        Run LSM backward induction on current self.paths / self.payoff.
        Returns price, std, se, exercise_time, trained_models
        """
        if self.basket_paths is None or self.payoff is None:
            self.simulate_paths()

        Npaths, M = self.basket_paths.shape

        # Default: American, decisions at 1..M-2 (exclude t=0 and maturity)
        if exercise_grid is None:
            exercise_grid = tuple(range(1, M - 1))

        # Backward induction
        cashflow = self.payoff[:, -1].copy()
        exercise_time = np.full(Npaths, M - 1, dtype=int)
        trained_models: Dict[int, RegressionBasis] = {}

        for t in reversed(exercise_grid):
            itm = self.payoff[:, t] > 0.0
            if not np.any(itm):
                continue

            X = self._state_at(t, itm, use_individual_prices)

            # Discount current realized cashflows back to time t
            discount_steps = exercise_time[itm] - t
            Y = cashflow[itm] * (self.disc ** discount_steps)

            # Fit a fresh basis at time t
            time_basis = PolynomialBasis(degree=self.degree)
            time_basis.fit(X, Y)
            trained_models[t] = time_basis

            # Predict continuation values and decide
            cont_val = time_basis.value(X)
            immediate = self.payoff[itm, t]
            exercise = immediate >= cont_val

            itm_idx = np.where(itm)[0]
            exercise_indices = itm_idx[exercise]
            cashflow[exercise_indices] = immediate[exercise]
            exercise_time[exercise_indices] = t

        # Discount final cashflows to t=0
        final_discount = self.disc ** exercise_time
        final_prices = cashflow * final_discount
        price = float(np.mean(final_prices))
        std = float(np.std(final_prices, ddof=1))
        se = std / np.sqrt(Npaths)
        return price, std, se, exercise_time, trained_models, exercise_grid, use_individual_prices

    # --- Public APIs

    def price(self,
              basis: Optional[RegressionBasis] = None,
              exercise_grid: Optional[Tuple[int, ...]] = None,
              use_individual_prices: bool = True,
              include_variance_state: Optional[bool] = None,
              seed: Optional[int] = None,
              return_exercise_times: bool = False):
        """
        One-shot LSM price (no model persistence).
        """
        if include_variance_state is not None:
            self.include_variance_state = include_variance_state

        self.simulate_paths(seed=seed)
        price, std, se, ex_times, _, _, _ = self._lsm_core(exercise_grid, use_individual_prices)

        if return_exercise_times:
            return price, std, se, ex_times
        return price, std, se

    def train(self,
              exercise_grid: Optional[Tuple[int, ...]] = None,
              use_individual_prices: bool = True,
              include_variance_state: Optional[bool] = None,
              seed: Optional[int] = None,
              return_exercise_times: bool = False):
        """
        Fit continuation regressions and store them (in-sample).
        """
        if include_variance_state is not None:
            self.include_variance_state = include_variance_state

        self.simulate_paths(seed=seed)
        price, std, se, ex_times, models, ex_grid, uip = self._lsm_core(exercise_grid, use_individual_prices)

        # Persist models/flags for testing
        self.trained_models = models
        self.exercise_grid = ex_grid
        self.use_individual_prices = uip
        self._trained_flags = {
            "basis_degree": self.degree,
            "include_variance_state": self.include_variance_state,
            "basket_kind": self.basket_kind,
            "option": self.option,
        }

        if return_exercise_times:
            return price, std, se, ex_times
        return price, std, se

    def test(self, n_test_paths: Optional[int] = None, seed: Optional[int] = None):
        """
        Apply stored models to new, independent paths (out-of-sample).
        """
        if not self.trained_models:
            raise ValueError("No trained models. Call train() first.")

        original_N = self.N
        if n_test_paths is not None:
            self.N = int(n_test_paths)

        # Generate NEW paths and calculate NEW payoffs for testing
        self.simulate_paths(seed=seed)
        
        # Calculate payoff on the NEW test paths
        if self.option == "call":
            test_payoff = np.maximum(self.basket_paths - self.K, 0.0)
        else:
            test_payoff = np.maximum(self.K - self.basket_paths, 0.0)

        Npaths, M = self.basket_paths.shape
        cashflow = test_payoff[:, -1].copy()  # Use NEW payoff at maturity
        exercise_time = np.full(Npaths, M - 1, dtype=int)

        # Apply trained models to NEW test data
        for t in reversed(self.exercise_grid):
            itm = test_payoff[:, t] > 0.0  # Use NEW payoff for ITM check
            if not np.any(itm):
                continue
            if t not in self.trained_models:
                continue

            X_test = self._state_at(t, itm, self.use_individual_prices)
            cont_val = self.trained_models[t].value(X_test)  # Apply TRAINED model
            immediate = test_payoff[itm, t]  # Use NEW payoff for immediate exercise
            exercise = immediate >= cont_val

            itm_idx = np.where(itm)[0]
            exercise_indices = itm_idx[exercise]
            cashflow[exercise_indices] = immediate[exercise]
            exercise_time[exercise_indices] = t

        final_discount = self.disc ** exercise_time
        final_prices = cashflow * final_discount

        test_price = float(np.mean(final_prices))
        test_std = float(np.std(final_prices, ddof=1))
        test_se = test_std / np.sqrt(Npaths)

        # restore N
        self.N = original_N
        return test_price, test_std, test_se


# ---------------------------
# Demo / diagnostic
# ---------------------------

if __name__ == "__main__":
    # Parameters
    n_assets = 5
    S0 = np.full(n_assets, 100.0)
    r = 0.1
    T = 0.5
    step = 126
    N_train = 100000
    N_test = 100000
    K = 100.0
    weights = np.full(n_assets, 1.0 / n_assets)

    # GBM covariance with uniform corr
    sig = 0.2
    corr = 0.30
    cov = np.full((n_assets, n_assets), corr * sig * sig)
    np.fill_diagonal(cov, sig * sig)

    print("Out-of-Sample LSM Testing for American PUT (geometric basket)")
    print(f"S0={S0[0]}, K={K}, T={T}, r={r}, Training paths: {N_train}, Test paths per batch: {N_test}")
    print("=" * 72)

    pricer = LSMOptionPricer(
        S0=S0, T=T, step=step, N=N_train, r=r, K=K,
        weights=weights,
        option="put",
        model="gbm",
        basket_kind="geometric",
        include_variance_state=False,   # (ignored for GBM)
        degree=2,  # Use degree 2 polynomial for this demo
        cov=cov
    )

    # Train (use a fixed seed for reproducibility)
    train_seed = 12345
    training_price, training_std, training_se = pricer.train(
        use_individual_prices=True,
        seed=train_seed
    )
    print(f"Training:  {training_price:.4f}  ± {training_se:.4f}  (SE),  STD={training_std:.4f}")
    
    # European option price for comparison (same paths)
    euro_payoff = pricer.payoff[:, -1]  # Payoff at maturity
    euro_prices = euro_payoff * np.exp(-r * T)  # Discount to t=0
    euro_price = float(np.mean(euro_prices))
    print(f"European:  {euro_price:.4f}")

    # Out-of-sample batches with distinct seeds
    seeds = [101, 202, 303, 404, 505]
    exceed_2se = 0
    test_prices = []

    print("Out-of-sample batches:")
    for i, sd in enumerate(seeds, 1):
        tp, tstd, tse = pricer.test(n_test_paths=N_test, seed=sd)
        test_prices.append(tp)
        diff = tp - training_price
        flag = abs(diff) > 2.0 * np.sqrt(training_se**2 + tse**2)
        exceed_2se += int(flag)
        mark = " **" if flag else ""
        print(f"  Batch {i}: {tp:.4f}  ± {tse:.4f} (SE),  Δ={diff:+.4f}{mark}")

    mean_test = float(np.mean(test_prices))
    bias = mean_test - training_price
    print("-" * 72)
    print(f"Mean OOS: {mean_test:.4f}, Bias (mean_oos - train): {bias:+.4f}")
    print(f"Share of batches where |Δ| > 2 × SE_test: {exceed_2se}/{len(seeds)}")