import numpy as np
from scipy.optimize import minimize
from math import log, sqrt


class HestonParams:
    def __init__(self, v0, theta, kappa, sigma, rho):
        self.v0, self.theta, self.kappa, self.sigma, self.rho = v0, theta, kappa, sigma, rho
    def to_dict(self):
        return {'v0': self.v0, 'theta': self.theta, 'kappa': self.kappa,
                'sigma': self.sigma, 'rho': self.rho}


class HestonCalibrator:
    def __init__(self, r=0.03, param_bounds=None, use_fft=True):
        self.r = r; self.use_fft = use_fft
        self.param_bounds = param_bounds or {
            'v0': (1e-4, 1.0), 'kappa': (0.1, 10), 'theta': (1e-4, 1.0),
            'sigma': (1e-3, 2.0), 'rho': (-0.99, 0.99)
        }

    def _cf(self, u, S0, K, tau, p):
        v0, kappa, theta, sigma, rho = p.v0, p.kappa, p.theta, p.sigma, p.rho
        x = log(S0/K)
        alpha, beta, gamma = -0.5, -0.5 - 1j*rho*sigma*u, 0.5*sigma*sigma
        d = np.sqrt(beta**2 - 4*alpha*gamma*u*(u+1j))
        g = (beta - d)/(beta + d)
        exp1 = np.exp(-d*kappa*tau)
        term1 = g*(1-exp1)/(1 - g*exp1)
        term2 = np.log((1 - g*exp1)/(1 - g)) - kappa*tau
        return np.exp(1j*u*x + 1j*u*(self.r - 0.5*v0)*tau
                      + v0*kappa*theta*term2/(sigma*sigma)
                      + v0*term1/(sigma*sigma))

    def price_fft(self, S0, K, tau, p, option_type='call'):
        # minimal FFT implementation
        N, B = 2**10, 500
        dx = B/N; dv = 2*np.pi/(N*dx)
        v = np.arange(N)*dv
        cf = self._cf(v - 0.5j, S0, K, tau, p)
        mult = np.exp(1j*v*(-0.5*N*dx))/(v*v+0.25)
        inp = cf*mult*dv; inp[0] = cf[0]*dv*0.5
        out = np.fft.fft(inp).real
        k = log(K); x0 = -0.5*N*dx
        idx = int((k - x0)/dx)
        call = np.exp(-k)*out[idx]/np.pi if 0<=idx<N else max(0, S0-K*np.exp(-self.r*tau))
        return call if option_type=='call' else call + K*np.exp(-self.r*tau) - S0

    def _model_iv(self, S0, K, tau, p, option_type):
        price = self.price_fft(S0, K, tau, p, option_type)
        from scipy.stats import norm
        def bs_err(s):
            d1 = (log(S0/K)+(self.r+0.5*s*s)*tau)/(s*sqrt(tau))
            d2 = d1 - s*sqrt(tau)
            bs = (S0*norm.cdf(d1) - K*np.exp(-self.r*tau)*norm.cdf(d2))
            return (bs - price)**2
        res = minimize(bs_err, sqrt(p.v0), bounds=[(1e-4,3.0)])
        return res.x[0] if res.success else np.nan

    def _obj(self, x, mkt, S0):
        p = HestonParams(*x)
        errs = []
        for _, r in mkt.iterrows():
            tau = r.DaysToExpiry/365.0
            iv = self._model_iv(S0, r.Strike, tau, p, r.OptionType)
            if not np.isnan(iv): errs.append((iv - r.ImpliedVolatility)**2)
        return np.mean(errs) if errs else 1e6

    def calibrate(self, spot, option_data, initial_params, max_iter=100):
        x0 = [initial_params.v0, initial_params.kappa,
              initial_params.theta, initial_params.sigma, initial_params.rho]
        bounds = [self.param_bounds[k] for k in ['v0','kappa','theta','sigma','rho']]
        res = minimize(self._obj, x0, args=(option_data, spot), bounds=bounds,
                       options={'maxiter': max_iter})
        return HestonParams(*res.x)