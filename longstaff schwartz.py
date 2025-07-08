import numpy as np
import matplotlib.pyplot as plt

class CorrelatedGBM:
    def __init__(self, S0, mu, cov, T, step, N):
        """
        Initialize the correlated GBM simulator.

        Parameters:
        S0 : np.ndarray : initial stock prices (shape: d,)
        mu : np.ndarray : drift coefficients (shape: d,)
        cov : np.ndarray : covariance matrix (shape: d, d)
        T : float : total time period
        step : int : number of time steps
        N : int : number of paths to generate
        """
        self.S0 = np.array(S0)
        self.mu = np.array(mu)
        self.cov = np.array(cov)
        self.T = T
        self.step = step
        self.N = N
        self.d = len(S0)
        self.paths = None

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
            drift = (self.mu - 0.5 * np.diag(self.cov)) * dt
            diffusion = correlated_Z * np.sqrt(dt)
            paths[:, i, :] = paths[:, i - 1, :] * np.exp(drift + diffusion)

        self.paths = paths
        return paths

    def plot_paths(self):
        """
        Plot the generated geometric Brownian motion paths.
        """
        if self.paths is None:
            raise ValueError("No paths generated. Call generate_paths() first.")

        N, M, d = self.paths.shape
        time = np.linspace(0, self.T, M)

        for i in range(d):
            plt.figure(figsize=(10, 6))
            for j in range(N):
                plt.plot(time, self.paths[j, :, i], alpha=0.5)
            plt.title(f'Geometric Brownian Motion Paths for Asset {i + 1}')
            plt.xlabel('Time (years)')
            plt.ylabel('Price')
            plt.grid()
            plt.show()

# Example usage
if __name__ == "__main__":
    # Parameters for two assets
    S0 = [100, 100]
    mu = [0.05, 0.05]
    sigma = [0.2, 0.2]

    # Covariance matrix with 0.7 correlation
    corr = 0.7
    cov = [
        [sigma[0]**2, corr * sigma[0] * sigma[1]],
        [corr * sigma[0] * sigma[1], sigma[1]**2]
    ]

    T = 3 / 12  # 3 months in years
    step = int(21 * 3)  # 21 trading days per month * 3 months = 63 steps
    N = 1000  # Number of simulations

    gbm_sim = CorrelatedGBM(S0, mu, cov, T, step, N)
    paths = gbm_sim.generate_paths()
    gbm_sim.plot_paths()
    print("Generated 1000 correlated GBM paths for 2 assets over 3 months.")
