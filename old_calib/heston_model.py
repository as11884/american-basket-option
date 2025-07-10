import numpy as np

class HestonModel:
    def __init__(self, params, r=0.03, dt=1/252):
        self.p = params; self.r = r; self.dt = dt

    def generate_paths(self, S0, T, n_paths=1000):
        v0, kappa, theta, sigma, rho = (
            self.p.v0, self.p.kappa, self.p.theta, self.p.sigma, self.p.rho)
        n_steps = int(T/self.dt) + 1
        S = np.zeros((n_paths,n_steps)); v = np.zeros_like(S)
        S[:,0] = S0; v[:,0] = v0
        sqrt_dt = np.sqrt(self.dt)
        Z1 = np.random.normal(size=(n_paths,n_steps-1))
        Z2 = np.random.normal(size=(n_paths,n_steps-1))
        Z2c = rho*Z1 + np.sqrt(1-rho**2)*Z2
        for i in range(n_steps-1):
            vp = np.maximum(v[:,i],0)
            v[:,i+1] = v[:,i] + kappa*(theta-vp)*self.dt + sigma*np.sqrt(vp)*Z1[:,i]*sqrt_dt
            S[:,i+1] = S[:,i]*np.exp((self.r-0.5*vp)*self.dt + np.sqrt(vp)*Z2c[:,i]*sqrt_dt)
        return S